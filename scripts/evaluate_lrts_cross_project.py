import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'processed' / 'lrts_v1'
TEST_PATH = DATA_DIR / 'lrts_test_modeling_v1.csv.gz'
AUDIT_PATH = DATA_DIR / 'lrts_run_audit_v1.csv.gz'
FEATURES = [
    'change_data_available', 'changed_file_count', 'additions', 'deletions',
    'changed_test_file_count', 'path_token_overlap_count', 'path_token_overlap_ratio',
    'test_file_changed', 'path_prefix_overlap_depth', 'prior_runs', 'prior_fails',
    'fail_rate', 'fails_last5', 'fails_last20', 'last_outcome', 'runs_since_fail',
    'last_duration', 'mean_duration',
]
RUN_KEY = ['project', 'stage_id', 'run_idx']
SPLIT_KEY = ['project', 'stage_id']
EVAL_FRACTIONS = (0.05, 0.10, 0.20)
METHODS = ('random', 'history_combo', 'path_overlap', 'model', 'model_per_prior_second')


def load_split_plan(audit):
    required = set(RUN_KEY + ['pr_name', 'build_timestamp', 'test_rows_match_metadata'])
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f'Run audit is missing columns: {sorted(missing)}')
    if audit.duplicated(RUN_KEY).any():
        raise ValueError('Run audit contains duplicate project/stage/run rows')

    cutoffs = {}
    test_prs = defaultdict(set)
    groups = []
    split_rows = []
    for (project, stage), group in audit.groupby(SPLIT_KEY, sort=False, observed=True):
        group = group.sort_values('run_idx', kind='stable')
        if not group['build_timestamp'].is_monotonic_increasing:
            raise ValueError(f'run_idx is not chronological for {project}/{stage}')
        if len(group) < 2:
            groups.append((project, stage, group, None))
            continue

        cutoff_position = min(max(int(len(group) * 0.70), 1), len(group) - 1)
        cutoff = int(group.iloc[cutoff_position]['run_idx'])
        cutoffs[(project, stage)] = cutoff
        test_group = group[group['run_idx'] >= cutoff]
        test_prs[project].update(test_group['pr_name'].astype(str).unique())
        groups.append((project, stage, group, cutoff))

    for project, stage, group, cutoff in groups:
        if cutoff is None:
            split_rows.append({
                'project': project, 'stage_id': stage, 'runs': len(group),
                'cutoff_run_idx': None, 'train_runs_pre_purge': 0,
                'purged_train_run_stages': 0, 'purged_pr_names': 0,
                'train_runs_after_purge': 0, 'test_runs': 0,
            })
            continue

        test_group = group[group['run_idx'] >= cutoff]
        train_group = group[group['run_idx'] < cutoff]
        purged_mask = train_group['pr_name'].astype(str).isin(test_prs[project])
        train_after_purge = train_group[~purged_mask]
        purged_pr_names = set(train_group.loc[purged_mask, 'pr_name'].astype(str))
        split_rows.append({
            'project': project, 'stage_id': stage, 'runs': len(group),
            'cutoff_run_idx': cutoff, 'train_runs_pre_purge': len(train_group),
            'test_runs': len(test_group),
            'purged_train_run_stages': int(purged_mask.sum()),
            'purged_pr_names': len(purged_pr_names),
            'train_runs_after_purge': len(train_after_purge),
        })

    return cutoffs, test_prs, pd.DataFrame(split_rows)


def split_masks(chunk, cutoffs, test_prs):
    train_mask = np.zeros(len(chunk), dtype=bool)
    test_mask = np.zeros(len(chunk), dtype=bool)
    for key, positions in chunk.groupby(SPLIT_KEY, sort=False, observed=True).indices.items():
        cutoff = cutoffs.get(key)
        if cutoff is None:
            continue
        local = chunk.iloc[positions]
        run_idx = local['run_idx'].to_numpy()
        after_cut = run_idx >= cutoff
        test_mask[positions] = after_cut
        if key[0] != 'tvm':
            seen_in_test_pr = local['pr_name'].astype(str).isin(test_prs[key[0]]).to_numpy()
            train_mask[positions] = (run_idx < cutoff) & ~seen_in_test_pr
    return train_mask, test_mask


def empty_metric_state():
    return {
        'test_rows': 0,
        'run_stages': 0,
        'failure_rows': 0,
        'failing_run_stages': 0,
        'apfd_sum': 0.0,
        'apfd_run_count': 0,
        'recall_sum': {fraction: 0.0 for fraction in EVAL_FRACTIONS},
        'caught': {fraction: 0 for fraction in EVAL_FRACTIONS},
        'run_hits': {fraction: 0 for fraction in EVAL_FRACTIONS},
        'total_duration': 0.0,
        'selected_duration': {fraction: 0.0 for fraction in EVAL_FRACTIONS},
    }


def record_run(stats, evaluation_group, test_ids, outcomes, durations, scores):
    n = len(outcomes)
    failure_count = int(outcomes.sum())
    duration_total = float(np.nan_to_num(durations, nan=0.0).sum())
    for method, score in scores.items():
        state = stats[(evaluation_group, method)]
        state['test_rows'] += n
        state['run_stages'] += 1
        state['failure_rows'] += failure_count
        state['failing_run_stages'] += int(failure_count > 0)
        state['total_duration'] += duration_total

        order = np.lexsort((test_ids, -score))
        if failure_count:
            positions = np.flatnonzero(outcomes[order] == 1) + 1
            state['apfd_sum'] += 1 - positions.sum() / (n * failure_count) + 1 / (2 * n)
            state['apfd_run_count'] += 1
            for fraction in EVAL_FRACTIONS:
                k = max(1, int(np.ceil(n * fraction)))
                recall = float((positions <= k).mean())
                state['recall_sum'][fraction] += recall

        for fraction in EVAL_FRACTIONS:
            k = max(1, int(np.ceil(n * fraction)))
            selected = order[:k]
            caught = int(outcomes[selected].sum())
            state['caught'][fraction] += caught
            state['run_hits'][fraction] += int(caught > 0)
            state['selected_duration'][fraction] += float(
                np.nan_to_num(durations[selected], nan=0.0).sum()
            )


def finish_run(frame, stats, random_rng):
    if frame.empty:
        return
    evaluation_group = str(frame['evaluation_group'].iloc[0])
    test_ids = frame['test_id'].astype(str).to_numpy()
    outcomes = frame['outcome'].to_numpy(dtype=np.int8)
    durations = frame['duration'].to_numpy(dtype=np.float64)
    model_score = frame['model_score'].to_numpy(dtype=np.float64)
    history_score = (
        frame['fails_last5'].to_numpy(dtype=np.float64) * 10.0
        + frame['fails_last20'].to_numpy(dtype=np.float64)
        + frame['fail_rate'].to_numpy(dtype=np.float64)
    )
    overlap_score = frame['path_token_overlap_ratio'].to_numpy(dtype=np.float64)
    prior_duration = np.maximum(frame['mean_duration'].to_numpy(dtype=np.float64), 0.1)
    scores = {
        'random': random_rng.random(len(frame)),
        'history_combo': history_score,
        'path_overlap': overlap_score,
        'model': model_score,
        'model_per_prior_second': model_score / prior_duration,
    }
    record_run(stats, evaluation_group, test_ids, outcomes, durations, scores)


def evaluation_chunk(frame, stats, random_rng, pending=None, final=False):
    if frame.empty and pending is None:
        return None
    if pending is not None:
        frame = pd.concat([pending, frame], ignore_index=True) if not frame.empty else pending
    groups = list(frame.groupby(RUN_KEY, sort=False, observed=True))
    if not groups:
        return None
    if final:
        for _, group in groups:
            finish_run(group, stats, random_rng)
        return None
    for _, group in groups[:-1]:
        finish_run(group, stats, random_rng)
    return groups[-1][1].copy()


def metric_rows(stats, audit, cutoffs, test_prs):
    count_status = {}
    for evaluation_group, group in audit.groupby('evaluation_group', sort=False):
        test_run_mask = np.zeros(len(group), dtype=bool)
        for key, positions in group.groupby(SPLIT_KEY, sort=False, observed=True).indices.items():
            cutoff = cutoffs.get(key)
            if cutoff is not None:
                test_run_mask[positions] = group.iloc[positions]['run_idx'].to_numpy() >= cutoff
        test_group = group.iloc[np.flatnonzero(test_run_mask)]
        count_status[evaluation_group] = (
            float(test_group['test_rows_match_metadata'].mean()) if len(test_group) else float('nan')
        )

    rows = []
    for (evaluation_group, method), state in stats.items():
        scope = 'observed_rows_only' if evaluation_group == 'tvm_mismatched' else 'metadata_reconciled'
        row = {
            'project': evaluation_group,
            'evaluation_scope': scope,
            'count_reconciliation_rate': count_status.get(evaluation_group, float('nan')),
            'method': method,
            'test_rows': state['test_rows'],
            'run_stages': state['run_stages'],
            'failing_run_stages': state['failing_run_stages'],
            'failure_rows': state['failure_rows'],
            'apfd': state['apfd_sum'] / state['apfd_run_count'] if state['apfd_run_count'] else float('nan'),
        }
        for fraction in EVAL_FRACTIONS:
            label = int(fraction * 100)
            row[f'recall_at_{label}pct'] = (
                state['recall_sum'][fraction] / state['apfd_run_count']
                if state['apfd_run_count'] else float('nan')
            )
            row[f'failures_caught_at_{label}pct'] = (
                100 * state['caught'][fraction] / state['failure_rows']
                if state['failure_rows'] else float('nan')
            )
            row[f'failing_runs_hit_at_{label}pct'] = (
                100 * state['run_hits'][fraction] / state['failing_run_stages']
                if state['failing_run_stages'] else float('nan')
            )
            row[f'time_saved_at_{label}pct'] = (
                100 * (1 - state['selected_duration'][fraction] / state['total_duration'])
                if state['total_duration'] else float('nan')
            )
        rows.append(row)

    pooled_projects = [project for project in count_status
                       if project not in {'tvm_reconciled', 'tvm_mismatched'}]
    for method in METHODS:
        components = [stats[(project, method)] for project in pooled_projects if (project, method) in stats]
        if not components:
            continue
        pooled = empty_metric_state()
        for component in components:
            for key in ('test_rows', 'run_stages', 'failure_rows', 'failing_run_stages', 'apfd_sum',
                        'apfd_run_count', 'total_duration'):
                pooled[key] += component[key]
            for fraction in EVAL_FRACTIONS:
                for key in ('recall_sum', 'caught', 'run_hits', 'selected_duration'):
                    pooled[key][fraction] += component[key][fraction]
        pooled_row = {
            'project': 'all_nine_non_tvm',
            'evaluation_scope': 'metadata_reconciled',
            'count_reconciliation_rate': 1.0,
            'method': method,
            'test_rows': pooled['test_rows'],
            'run_stages': pooled['run_stages'],
            'failing_run_stages': pooled['failing_run_stages'],
            'failure_rows': pooled['failure_rows'],
            'apfd': pooled['apfd_sum'] / pooled['apfd_run_count'] if pooled['apfd_run_count'] else float('nan'),
        }
        for fraction in EVAL_FRACTIONS:
            label = int(fraction * 100)
            pooled_row[f'recall_at_{label}pct'] = (
                pooled['recall_sum'][fraction] / pooled['apfd_run_count']
                if pooled['apfd_run_count'] else float('nan')
            )
            pooled_row[f'failures_caught_at_{label}pct'] = (
                100 * pooled['caught'][fraction] / pooled['failure_rows']
                if pooled['failure_rows'] else float('nan')
            )
            pooled_row[f'failing_runs_hit_at_{label}pct'] = (
                100 * pooled['run_hits'][fraction] / pooled['failing_run_stages']
                if pooled['failing_run_stages'] else float('nan')
            )
            pooled_row[f'time_saved_at_{label}pct'] = (
                100 * (1 - pooled['selected_duration'][fraction] / pooled['total_duration'])
                if pooled['total_duration'] else float('nan')
            )
        rows.append(pooled_row)

    return pd.DataFrame(rows), count_status


def main():
    parser = argparse.ArgumentParser(description='Time-split, PR-purged cross-project LRTS model evaluation.')
    parser.add_argument('--negative-sample-rate', type=float, default=0.005)
    parser.add_argument('--chunksize', type=int, default=250000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--plan-only', action='store_true', help='Print split/TVM diagnostics without fitting or scanning the test table.')
    args = parser.parse_args()
    if not 0 < args.negative_sample_rate <= 1:
        parser.error('--negative-sample-rate must be in (0, 1]')
    if args.chunksize < 1:
        parser.error('--chunksize must be positive')
    if not TEST_PATH.is_file() or not AUDIT_PATH.is_file():
        raise FileNotFoundError('Finalized LRTS v1 table and run audit must exist before evaluation')
    output_path = DATA_DIR / 'cross_project_evaluation_tvm_split.csv'
    diagnostics_path = DATA_DIR / 'tvm_stage_count_reconciliation.csv'
    split_path = DATA_DIR / 'cross_project_time_split_plan.csv'
    manifest_path = DATA_DIR / 'cross_project_evaluation_manifest.json'
    model_path = DATA_DIR / 'cross_project_model_tvm_split.joblib'
    outputs = [output_path, diagnostics_path, split_path, manifest_path, model_path]
    if not args.plan_only and any(path.exists() for path in outputs):
        raise FileExistsError('Evaluation outputs already exist: '
                              + ', '.join(str(path) for path in outputs if path.exists()))

    audit = pd.read_csv(AUDIT_PATH, dtype={'project': 'string', 'stage_id': 'string', 'pr_name': 'string'})
    audit['evaluation_group'] = audit['project'].astype(str)
    tvm_mask = audit['project'].eq('tvm')
    audit.loc[tvm_mask & audit['test_rows_match_metadata'].eq(1), 'evaluation_group'] = 'tvm_reconciled'
    audit.loc[tvm_mask & audit['test_rows_match_metadata'].eq(0), 'evaluation_group'] = 'tvm_mismatched'
    cutoffs, test_prs, split_plan = load_split_plan(audit)
    split_plan['test_pr_names_in_project'] = split_plan['project'].map(
        lambda project: len(test_prs.get(project, set()))
    )
    count_diagnostics = (audit.groupby(['project', 'stage_id'], observed=True)
                        .agg(run_stages=('run_idx', 'size'),
                             count_reconciliation_rate=('test_rows_match_metadata', 'mean'),
                             median_metadata_tests=('metadata_test_count', 'median'),
                             median_raw_test_rows=('raw_test_rows', 'median'),
                             median_normalized_tests=('normalized_test_count', 'median'))
                        .reset_index())
    print('Count reconciliation by project:')
    print(audit.groupby('project')['test_rows_match_metadata'].agg(['size', 'mean']).to_string())
    print('TVM stage count reconciliation:')
    print(count_diagnostics[count_diagnostics['project'].eq('tvm')].to_string(index=False))
    print('Time split plan:')
    print(split_plan.to_string(index=False))
    if args.plan_only:
        return

    train_x, train_y, train_weight = [], [], []
    rng = np.random.default_rng(args.seed)
    train_population_rows = 0
    train_positive_rows = 0
    train_pr_purged_rows = 0
    usecols = list(dict.fromkeys(
        ['project', 'stage_id', 'run_idx', 'pr_name', 'test_id', 'outcome', 'duration'] + FEATURES
    ))
    for chunk in pd.read_csv(TEST_PATH, usecols=usecols, chunksize=args.chunksize,
                             dtype={'project': 'string', 'stage_id': 'string', 'pr_name': 'string', 'test_id': 'string'}):
        eligible, _ = split_masks(chunk, cutoffs, test_prs)
        eligible &= chunk['project'].ne('tvm').to_numpy()
        candidate_rows = int(eligible.sum())
        train_population_rows += candidate_rows
        eligible_y = chunk.loc[eligible, 'outcome'].to_numpy(dtype=np.int8)
        positive = eligible_y == 1
        train_positive_rows += int(positive.sum())
        negative_positions = np.flatnonzero(~positive)
        sampled_negative = np.zeros(len(eligible_y), dtype=bool)
        if len(negative_positions):
            sampled_negative[negative_positions] = rng.random(len(negative_positions)) < args.negative_sample_rate
        keep = positive | sampled_negative
        if not keep.any():
            continue
        local = chunk.loc[eligible].iloc[np.flatnonzero(keep)]
        if local[FEATURES].isna().any().any():
            missing_features = local[FEATURES].columns[local[FEATURES].isna().any()].tolist()
            raise ValueError(f'Missing model feature values in training rows: {missing_features}')
        x = local[FEATURES].to_numpy(dtype=np.float32, copy=True)
        y = local['outcome'].to_numpy(dtype=np.int8, copy=True)
        weights = np.where(y == 1, 1.0, 1.0 / args.negative_sample_rate).astype(np.float32)
        train_x.append(x)
        train_y.append(y)
        train_weight.append(weights)

    if not train_x:
        raise ValueError('No training rows were selected')
    x_train = np.concatenate(train_x)
    y_train = np.concatenate(train_y)
    sample_weight = np.concatenate(train_weight)
    train_sample_rows = int(len(y_train))
    if np.unique(y_train).size != 2:
        raise ValueError('Training sample does not contain both pass and failure labels')
    print(f'Train population rows after time split/PR purge: {train_population_rows:,}')
    print(f'Train sample rows: {len(y_train):,}; failures: {int(y_train.sum()):,}; '
          f'negative sample rate: {args.negative_sample_rate:g}')

    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=args.seed,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    del x_train, y_train, sample_weight, train_x, train_y, train_weight

    rng_eval = np.random.default_rng(args.seed)
    stats = defaultdict(empty_metric_state)
    pending = None
    for chunk in pd.read_csv(TEST_PATH, usecols=usecols, chunksize=args.chunksize,
                             dtype={'project': 'string', 'stage_id': 'string', 'pr_name': 'string', 'test_id': 'string'}):
        _, in_test = split_masks(chunk, cutoffs, test_prs)
        if not in_test.any():
            continue
        evaluation = chunk.loc[in_test].copy()
        if evaluation[FEATURES].isna().any().any():
            missing_features = evaluation[FEATURES].columns[evaluation[FEATURES].isna().any()].tolist()
            raise ValueError(f'Missing model feature values in evaluation rows: {missing_features}')
        evaluation['model_score'] = model.predict_proba(
            evaluation[FEATURES].to_numpy(dtype=np.float32, copy=False)
        )[:, 1]
        evaluation = evaluation.merge(
            audit[RUN_KEY + ['evaluation_group']], on=RUN_KEY, how='left', validate='many_to_one'
        )
        if evaluation['evaluation_group'].isna().any():
            raise ValueError('Some held-out rows could not be assigned an evaluation group')
        pending = evaluation_chunk(evaluation, stats, rng_eval, pending=pending, final=False)
    if pending is not None:
        evaluation_chunk(pd.DataFrame(), stats, rng_eval, pending=pending, final=True)

    results, count_status = metric_rows(stats, audit, cutoffs, test_prs)
    results.to_csv(output_path, index=False)
    count_diagnostics.to_csv(diagnostics_path, index=False)
    split_plan.to_csv(split_path, index=False)
    joblib.dump(model, model_path)
    evaluation_manifest = {
        'evaluation_version': 'cross_project_tvm_split',
        'dataset_manifest': 'data/processed/lrts_v1/lrts_manifest_v1.json',
        'split': 'chronological per project/stage, 70/30; training PRs purged if PR appears in any test stage for that project',
        'training_projects': sorted(project for project in audit['project'].unique() if project != 'tvm'),
        'tvm_evaluation': {
            'training': 'TVM excluded from training; model trained on the nine non-TVM projects',
            'reconciled_runs': 'reported separately as tvm_reconciled using runs where raw test rows equal pass+fail metadata',
            'mismatched_runs': 'reported separately as tvm_mismatched using observed rows only; not complete-suite ground truth',
        },
        'negative_sample_rate': args.negative_sample_rate,
        'negative_sample_weight': 1.0 / args.negative_sample_rate,
        'seed': args.seed,
        'features': FEATURES,
        'model': 'HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=31)',
        'metrics': ['APFD', 'recall@5/10/20%', 'failures_caught@5/10/20%',
                    'failing_run_stages_hit@5/10/20%', 'duration-weighted time_saved@5/10/20%'],
        'outputs': [str(path.relative_to(ROOT)) for path in outputs],
        'train_population_rows_after_split_and_pr_purge': train_population_rows,
        'train_sample_rows': train_sample_rows,
        'train_positive_rows': train_positive_rows,
    }
    manifest_path.write_text(json.dumps(evaluation_manifest, indent=2), encoding='utf-8')
    print('\nPer-project and pooled evaluation metrics:')
    print(results.to_string(index=False))
    print('\nSaved evaluation artifacts:')
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
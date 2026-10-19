"""Evaluate non-TVM model variants without replacing the saved model artifact.

The experiment uses the established chronological, PR-purged split. Its
uncertainty estimates are paired percentile bootstraps that resample held-out
PR clusters within project; every method receives the same resampled PRs.
"""
import hashlib
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from evaluate_model import (  # noqa: E402
    AUDIT_PATH, BASE_FEATURES, FILE_LINK_FEATURES, FILE_LINK_PATH, ROOT,
    TABLE_FEATURES, TABLE_HAS_FILE_LINKS, TEST_PATH, attach_file_link_features,
    filter_non_tvm, load_file_link_sidecar, load_split_plan, split_masks,
)

SEED, NEGATIVE_RATE, CHUNK, BOOTSTRAP_REPS = 0, 0.005, 250_000, 2_000
LINK_FEATURES = FILE_LINK_FEATURES if TABLE_HAS_FILE_LINKS or FILE_LINK_PATH.is_file() else []
FEATURES = BASE_FEATURES + LINK_FEATURES
HISTORY = ['prior_runs', 'prior_fails', 'fail_rate', 'fails_last5', 'fails_last20',
           'last_outcome', 'runs_since_fail', 'last_duration', 'mean_duration']
CODE = [column for column in BASE_FEATURES if column not in HISTORY]
SUBSETS = (
    'all', 'change_bearing', 'quiet', 'never_failed', 'quiet_and_never_failed',
    'quiet_change_bearing', 'never_failed_change_bearing',
    'quiet_and_never_failed_change_bearing',
)
METRICS = ('apfd', 'recall_10pct')
MODELS = {
    'rf_history': (RandomForestClassifier, HISTORY),
    'rf_code': (RandomForestClassifier, CODE),
    'rf_both': (RandomForestClassifier, BASE_FEATURES),
    'hgb_history': (HistGradientBoostingClassifier, HISTORY),
    'hgb_code': (HistGradientBoostingClassifier, CODE),
    'hgb_both': (HistGradientBoostingClassifier, BASE_FEATURES),
}
if LINK_FEATURES:
    MODELS.update({
        'rf_code_plus_filelink': (RandomForestClassifier, CODE + LINK_FEATURES),
        'rf_both_plus_filelink': (RandomForestClassifier, FEATURES),
        'hgb_code_plus_filelink': (HistGradientBoostingClassifier, CODE + LINK_FEATURES),
        'hgb_both_plus_filelink': (HistGradientBoostingClassifier, FEATURES),
    })
METHODS = tuple(MODELS) + ('history_combo', 'random')
BASE_COMPARISONS = (
    ('rf_both', 'rf_history'), ('hgb_both', 'hgb_history'),
    ('rf_both', 'history_combo'), ('hgb_both', 'history_combo'),
    ('rf_both', 'hgb_both'),
)
FILE_LINK_COMPARISONS = (
    ('rf_code_plus_filelink', 'rf_code'), ('hgb_code_plus_filelink', 'hgb_code'),
    ('rf_both_plus_filelink', 'rf_both'), ('hgb_both_plus_filelink', 'hgb_both'),
) if LINK_FEATURES else ()
COMPARISONS = BASE_COMPARISONS + FILE_LINK_COMPARISONS


def estimator(cls):
    if cls is RandomForestClassifier:
        return cls(n_estimators=48, max_depth=18, min_samples_leaf=10,
                   max_features='sqrt', n_jobs=1, random_state=SEED)
    return cls(max_iter=100, max_leaf_nodes=31, learning_rate=0.08,
               l2_regularization=1.0, random_state=SEED)


def fit_models(cutoffs, test_prs, file_link_sidecar):
    rng = np.random.default_rng(SEED)
    samples = {name: [[], [], []] for name in MODELS}
    population = positives = 0
    usecols = list(dict.fromkeys(['project', 'stage_id', 'run_idx', 'pr_name', 'outcome'] + TABLE_FEATURES))
    for chunk in pd.read_csv(TEST_PATH, usecols=usecols, chunksize=CHUNK,
                             dtype={'project': 'string', 'stage_id': 'string', 'pr_name': 'string'}):
        chunk = attach_file_link_features(chunk, file_link_sidecar)
        train, _ = split_masks(chunk, cutoffs, test_prs)
        train &= chunk.project.ne('tvm').to_numpy()
        local = chunk.loc[train]
        y = local.outcome.to_numpy(np.int8)
        population += len(y)
        positives += int(y.sum())
        negatives = np.flatnonzero(y == 0)
        keep = y == 1
        if len(negatives):
            keep[negatives] = rng.random(len(negatives)) < NEGATIVE_RATE
        local, y = local.iloc[np.flatnonzero(keep)], y[keep]
        if not len(y):
            continue
        weight = np.where(y == 1, 1.0, 1.0 / NEGATIVE_RATE).astype(np.float32)
        for name, (_, columns) in MODELS.items():
            samples[name][0].append(local[columns].to_numpy(np.float32, copy=True))
            samples[name][1].append(y.copy())
            samples[name][2].append(weight.copy())
    fitted = {}
    for name, (cls, columns) in MODELS.items():
        x = np.concatenate(samples[name][0])
        y = np.concatenate(samples[name][1])
        weight = np.concatenate(samples[name][2])
        fitted[name] = (estimator(cls).fit(x, y, sample_weight=weight), columns)
        print(f'{name}: fitted {len(y):,} rows ({int(y.sum()):,} failures)')
    return fitted, population, positives


def stable_seed(*parts):
    payload = '|'.join(map(str, (SEED,) + parts)).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little')


def per_run_metrics(frame, scores, tie_break):
    outcomes = frame.outcome.to_numpy(np.int8)
    if not len(outcomes) or not int(outcomes.sum()):
        return {}
    masks = {
        'all': np.ones(len(frame), dtype=bool),
        'quiet': frame.fails_last20.to_numpy() == 0,
        'never_failed': frame.prior_fails.to_numpy() == 0,
    }
    masks['quiet_and_never_failed'] = masks['quiet'] & masks['never_failed']
    change_bearing = ((frame.change_data_available.to_numpy() == 1)
                      & (frame.changed_file_count.to_numpy() > 0))
    masks.update({
        'change_bearing': change_bearing,
        'quiet_change_bearing': change_bearing & masks['quiet'],
        'never_failed_change_bearing': change_bearing & masks['never_failed'],
        'quiet_and_never_failed_change_bearing': change_bearing & masks['quiet'] & masks['never_failed'],
    })
    rows = {}
    for subset, mask in masks.items():
        y = outcomes[mask]
        test_count, failure_count = len(y), int(y.sum())
        if not test_count or not failure_count:
            continue
        for method, score in scores.items():
            # The secondary key is fixed pseudorandom noise for this run and is
            # shared by every method. It replaces alphabetical test-id ordering.
            order = np.lexsort((tie_break[mask], -score[mask]))
            positions = np.flatnonzero(y[order] == 1) + 1
            rows[(subset, method, 'apfd')] = float(
                1 - positions.sum() / (test_count * failure_count) + 1 / (2 * test_count)
            )
            k = max(1, int(np.ceil(test_count * 0.10)))
            rows[(subset, method, 'recall_10pct')] = float((positions <= k).mean())
    return rows


def aggregate_rows(run_rows):
    totals = defaultdict(lambda: [0.0, 0])
    for row in run_rows:
        for (subset, method, metric), value in row['metrics'].items():
            total = totals[(row['project'], subset, method, metric)]
            total[0] += value
            total[1] += 1
    projects = []
    for (project, subset, method, metric), (total, count) in totals.items():
        projects.append({'project': project, 'subset': subset, 'method': method, 'metric': metric,
                         'failing_run_stages': count, 'mean': total / count})
    project_frame = pd.DataFrame(projects)
    aggregate = []
    for (subset, method, metric), group in project_frame.groupby(['subset', 'method', 'metric'], sort=True):
        weighted_count = int(group.failing_run_stages.sum())
        weighted_mean = float(np.average(group['mean'], weights=group.failing_run_stages))
        aggregate.extend([
            {'aggregation': 'failing_run_stage_weighted', 'subset': subset, 'method': method,
             'metric': metric, 'projects': int(group.project.nunique()),
             'failing_run_stages': weighted_count, 'mean': weighted_mean},
            {'aggregation': 'macro_project', 'subset': subset, 'method': method,
             'metric': metric, 'projects': int(group.project.nunique()),
             'failing_run_stages': weighted_count, 'mean': float(group['mean'].mean())},
        ])
    return project_frame, pd.DataFrame(aggregate)


def paired_pr_bootstrap(run_rows):
    compared_methods = tuple(dict.fromkeys(method for pair in COMPARISONS for method in pair))
    method_index = {method: index for index, method in enumerate(compared_methods)}
    result = []
    for subset in SUBSETS:
        for metric in METRICS:
            clusters = defaultdict(lambda: defaultdict(list))
            for row in run_rows:
                values = row['metrics']
                vector = [values.get((subset, method, metric)) for method in compared_methods]
                if any(value is None for value in vector):
                    continue
                clusters[row['project']][row['pr_name']].append(vector)
            if not clusters:
                continue
            project_data = {}
            for project, per_pr in clusters.items():
                sums = np.asarray([np.asarray(values, dtype=float).sum(axis=0)
                                   for values in per_pr.values()])
                counts = np.asarray([len(values) for values in per_pr.values()], dtype=np.int64)
                project_data[project] = (sums, counts)
            projects = sorted(project_data)
            point_project = {}
            total_sum = np.zeros(len(compared_methods), dtype=float)
            total_count = 0
            for project, (sums, counts) in project_data.items():
                point_project[project] = sums.sum(axis=0) / counts.sum()
                total_sum += sums.sum(axis=0)
                total_count += int(counts.sum())
            point_weighted = total_sum / total_count
            point_macro = np.mean(np.stack(list(point_project.values())), axis=0)
            draws_weighted = np.empty((BOOTSTRAP_REPS, len(compared_methods)), dtype=float)
            draws_macro = np.empty_like(draws_weighted)
            rng = np.random.default_rng(stable_seed(subset, metric, 'pr-cluster-bootstrap'))
            for replicate in range(BOOTSTRAP_REPS):
                sampled_sum = np.zeros(len(compared_methods), dtype=float)
                sampled_count = 0
                per_project_means = []
                for project in projects:
                    sums, counts = project_data[project]
                    indices = rng.integers(0, len(counts), size=len(counts))
                    current_sum = sums[indices].sum(axis=0)
                    current_count = int(counts[indices].sum())
                    sampled_sum += current_sum
                    sampled_count += current_count
                    per_project_means.append(current_sum / current_count)
                draws_weighted[replicate] = sampled_sum / sampled_count
                draws_macro[replicate] = np.mean(np.stack(per_project_means), axis=0)
            for left, right in COMPARISONS:
                left_index, right_index = method_index[left], method_index[right]
                project_diffs = np.asarray([values[left_index] - values[right_index]
                                            for values in point_project.values()])
                wins_left = int((project_diffs > 0).sum())
                wins_right = int((project_diffs < 0).sum())
                ties = int((project_diffs == 0).sum())
                for aggregation, point, draws in (
                    ('failing_run_stage_weighted', point_weighted, draws_weighted),
                    ('macro_project', point_macro, draws_macro),
                ):
                    differences = draws[:, left_index] - draws[:, right_index]
                    lo, hi = np.quantile(differences, [0.025, 0.975])
                    result.append({
                        'subset': subset, 'metric': metric, 'aggregation': aggregation,
                        'left_method': left, 'right_method': right,
                        'difference_left_minus_right': float(point[left_index] - point[right_index]),
                        'ci95_low': float(lo), 'ci95_high': float(hi),
                        'projects': len(projects), 'pr_clusters': int(sum(len(item[1]) for item in project_data.values())),
                        'failing_run_stages': total_count, 'left_project_wins': wins_left,
                        'right_project_wins': wins_right, 'project_ties': ties,
                    })
    return pd.DataFrame(result)


def f4(value):
    return f'{float(value):.4f}'


def interval(row):
    return f"{f4(row.difference_left_minus_right)} [{f4(row.ci95_low)}, {f4(row.ci95_high)}]"


def write_report(summary, differences, population, positives):
    path = ROOT / 'docs' / 'non_tvm_model_experiments.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Non-TVM model experiments', '', '## Measured protocol', '',
        'TVM was excluded before split construction and from fitting and scoring. Results use the established chronological 70/30 project/stage split with PR purge, seed 0, all positive training rows, and a deterministic 0.5% negative sample with inverse-probability weights.', '',
        f'The corpus was rebuilt with `scripts/build_dataset.py --overwrite`. Training scanned {population:,} non-TVM rows with {positives:,} failures. The saved deployment artifact was not replaced.', '',
        f'Each metric is a mean over failing held-out run-stages. Ties use a seeded pseudorandom order shared by all methods within a run. Intervals are {BOOTSTRAP_REPS:,} percentile paired bootstraps that resample PR clusters within project; each replicate uses the same sampled PRs for every method. `failing_run_stage_weighted` weights projects by their failing run-stage counts; `macro_project` averages the nine project means.', '',
        '## Aggregate metrics', '',
        '| Model | Aggregation | APFD | Recall at 10% | Projects | Failing run-stages |',
        '|---|---|---:|---:|---:|---:|',
    ]
    selected = summary[(summary.subset == 'all') & (summary.method != 'random')]
    for method in list(MODELS) + ['history_combo']:
        for aggregation in ('failing_run_stage_weighted', 'macro_project'):
            rows = selected[(selected.method == method) & (selected.aggregation == aggregation)]
            apfd = rows.loc[rows.metric == 'apfd'].iloc[0]
            recall = rows.loc[rows.metric == 'recall_10pct'].iloc[0]
            lines.append(f'| {method} | {aggregation} | {f4(apfd.mean)} | {f4(recall.mean)} | {int(apfd.projects)} | {int(apfd.failing_run_stages):,} |')
    lines += ['', '## Requested paired differences', '',
              'Difference is left method minus right method. Win counts use each project’s observed mean; zero-difference projects are ties.', '',
              '| Slice | Metric | Comparison | Aggregation | Difference [95% CI] | Left/right wins/ties |', '|---|---|---|---|---:|---:|']
    requested = differences[(differences.subset == 'all') & differences.apply(lambda row: (row.left_method, row.right_method) in BASE_COMPARISONS, axis=1)]
    for _, row in requested.sort_values(['metric', 'left_method', 'right_method', 'aggregation']).iterrows():
        lines.append(f"| all | {row.metric} | {row.left_method} − {row.right_method} | {row.aggregation} | {interval(row)} | {int(row.left_project_wins)}/{int(row.right_project_wins)}/{int(row.project_ties)} |")
    lines += ['', '## Quiet and never-failed paired differences', '',
              '| Slice | Metric | Comparison | Aggregation | Difference [95% CI] | Left/right wins/ties |', '|---|---|---|---|---:|---:|']
    history_pairs = {('rf_both', 'rf_history'), ('hgb_both', 'hgb_history')}
    focused = differences[differences.subset.isin(['quiet', 'never_failed']) & differences.apply(lambda row: (row.left_method, row.right_method) in history_pairs, axis=1)]
    for _, row in focused.sort_values(['subset', 'metric', 'left_method', 'aggregation']).iterrows():
        lines.append(f"| {row.subset} | {row.metric} | {row.left_method} − {row.right_method} | {row.aggregation} | {interval(row)} | {int(row.left_project_wins)}/{int(row.right_project_wins)}/{int(row.project_ties)} |")
    lines += ['', '## File-link feature effects on every slice', '',
              'Each row is a paired difference. An interval excluding zero is reported as such; no direction is described as an improvement here.', '',
              '| Slice | Metric | Comparison | Aggregation | Difference [95% CI] | Interval status |', '|---|---|---|---|---:|---|']
    links = differences[differences.apply(lambda row: (row.left_method, row.right_method) in FILE_LINK_COMPARISONS, axis=1)]
    for _, row in links.sort_values(['subset', 'metric', 'left_method', 'aggregation']).iterrows():
        status = 'excludes zero' if row.ci95_low > 0 or row.ci95_high < 0 else 'includes zero'
        lines.append(f'| {row.subset} | {row.metric} | {row.left_method} − {row.right_method} | {row.aggregation} | {interval(row)} | {status} |')
    lines += ['', '## Artifacts', '',
              '- `artifacts/evaluation/non_tvm_model_experiment_summary.csv` contains weighted and macro estimates.',
              '- `artifacts/evaluation/non_tvm_model_paired_differences.csv` contains all paired PR-clustered intervals and project win counts.',
              '- `artifacts/evaluation/non_tvm_model_experiments.csv` contains per-project estimates.',
              '- `scripts/experiments/run_non_tvm_model_experiments.py` reproduces this report.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def record_run(run_rows, run_name, frame, predictions):
    history_score = (frame.fails_last5.to_numpy(float) * 10.0
                     + frame.fails_last20.to_numpy(float) + frame.fail_rate.to_numpy(float))
    rng = np.random.default_rng(stable_seed(run_name, 'ties'))
    scores = {**predictions, 'history_combo': history_score, 'random': rng.random(len(frame))}
    metrics = per_run_metrics(frame, scores, rng.random(len(frame)))
    if metrics:
        run_rows.append({'project': str(frame.project.iloc[0]), 'pr_name': str(frame.pr_name.iloc[0]),
                         'run': run_name, 'metrics': metrics})


def evaluate_runs(fitted, cutoffs, test_prs, file_link_sidecar):
    usecols = list(dict.fromkeys(['project', 'stage_id', 'run_idx', 'pr_name', 'test_id', 'outcome', 'fails_last20', 'prior_fails'] + TABLE_FEATURES))
    run_rows, pending = [], None
    for chunk in pd.read_csv(TEST_PATH, usecols=usecols, chunksize=CHUNK,
                             dtype={'project': 'string', 'stage_id': 'string', 'pr_name': 'string', 'test_id': 'string'}):
        chunk = attach_file_link_features(chunk, file_link_sidecar)
        _, held = split_masks(chunk, cutoffs, test_prs)
        frame = chunk.loc[held & chunk.project.ne('tvm').to_numpy()].copy()
        if frame.empty:
            continue
        frame.reset_index(drop=True, inplace=True)
        predictions = {name: model.predict_proba(frame[columns].to_numpy(np.float32, copy=False))[:, 1] for name, (model, columns) in fitted.items()}
        frame['_run'] = frame.project.astype(str) + '|' + frame.stage_id.astype(str) + '|' + frame.run_idx.astype(int).astype(str)
        if pending is not None:
            frame = pd.concat([pending[0], frame], ignore_index=True)
            predictions = {name: np.concatenate([pending[1][name], values]) for name, values in predictions.items()}
        groups = list(frame.groupby('_run', sort=False))
        for run_name, group in groups[:-1]:
            indexes = group.index.to_numpy()
            record_run(run_rows, run_name, group, {name: values[indexes] for name, values in predictions.items()})
        if groups:
            _, last = groups[-1]
            indexes = last.index.to_numpy()
            pending = (last.copy(), {name: values[indexes].copy() for name, values in predictions.items()})
    if pending is not None:
        record_run(run_rows, pending[0]['_run'].iloc[0], pending[0], pending[1])
    return run_rows


def main():
    audit = filter_non_tvm(pd.read_csv(AUDIT_PATH, dtype={'project': 'string', 'stage_id': 'string', 'pr_name': 'string'}))
    cutoffs, test_prs, split_plan = load_split_plan(audit)
    if audit.project.astype(str).nunique() != 9:
        raise ValueError('Expected exactly nine non-TVM projects')
    print('Training experiment-only models on non-TVM rows; saved model artifact is untouched.')
    sidecar = load_file_link_sidecar()
    fitted, population, positives = fit_models(cutoffs, test_prs, sidecar)
    run_rows = evaluate_runs(fitted, cutoffs, test_prs, sidecar)
    projects, summary = aggregate_rows(run_rows)
    differences = paired_pr_bootstrap(run_rows)
    out = ROOT / 'artifacts' / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    projects.to_csv(out / 'non_tvm_model_experiments.csv', index=False)
    summary.to_csv(out / 'non_tvm_model_experiment_summary.csv', index=False)
    differences.to_csv(out / 'non_tvm_model_paired_differences.csv', index=False)
    split_plan.to_csv(out / 'non_tvm_experiment_split.csv', index=False)
    manifest = {
        'excluded_projects': ['tvm'], 'included_projects': sorted(audit.project.astype(str).unique()),
        'train_population_rows': population, 'train_failures': positives,
        'negative_sample_rate': NEGATIVE_RATE, 'bootstrap_unit': 'held-out PR cluster within project',
        'bootstrap_replicates': BOOTSTRAP_REPS, 'paired_methods_share_resampled_prs': True,
        'tie_break': 'seeded pseudorandom within run, shared by methods', 'features': FEATURES,
        'history_features': HISTORY, 'code_features': CODE, 'file_link_features': LINK_FEATURES,
        'estimators': {name: cls.__name__ for name, (cls, _) in MODELS.items()},
        'saved_model_artifact_replaced': False,
        'causality_note': 'History-constrained slices do not establish that a code change caused a failure.',
    }
    (out / 'non_tvm_model_experiments.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    write_report(summary, differences, population, positives)
    print(f'Saved {len(projects):,} project estimates and {len(differences):,} paired differences.')


if __name__ == '__main__':
    main()

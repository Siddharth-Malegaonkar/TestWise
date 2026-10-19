"""Chronological non-TVM offline replay of the cost-aware contextual bandit."""
import json
import hashlib
import math
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml.bandit.contextual_bandit import CostAwareContextualBandit

DATA = ROOT / 'data' / 'processed' / 'lrts_v1'
ARTIFACTS = ROOT / 'artifacts'
MODEL_PATH = ARTIFACTS / 'models' / 'testwise_lrts_non_tvm.joblib'
SPLIT_PATH = ARTIFACTS / 'evaluation' / 'non_tvm_time_split_plan.csv'
OUT_PATH = ARTIFACTS / 'evaluation' / 'non_tvm_bandit_replay.csv'
REPORT_PATH = ROOT / 'docs' / 'bandit_replay.md'
RUN_KEY = ['project', 'stage_id', 'run_idx']
FRACTION = 0.10
EPSILON = 0.1
COST_WEIGHT = 0.1
SEED = 0


def compute_cutoffs(split_plan):
    return {(r.project, r.stage_id): int(r.cutoff_run_idx)
            for r in split_plan.itertuples() if pd.notna(r.cutoff_run_idx)}


def make_policy(key, n_features):
    seed = zlib.crc32('|'.join(map(str, key)).encode('utf-8')) ^ SEED
    return CostAwareContextualBandit(
        n_features, epsilon=EPSILON, learning_rate=0.001,
        l2_penalty=1e-3, cost_weight=COST_WEIGHT, random_state=seed,
    )


def add_run(group, policies, last_run, rngs, states):
    project = str(group['project'].iloc[0])
    stage = str(group['stage_id'].iloc[0])
    key = (project, stage)
    run_idx = int(group['run_idx'].iloc[0])
    if run_idx <= last_run.get(key, -1):
        raise ValueError(f'Bandit replay is not chronological for {project}/{stage}: {run_idx}')
    last_run[key] = run_idx
    if project.lower() == 'tvm':
        raise AssertionError('TVM rows reached bandit replay')
    features = group.attrs.get('features')
    if features is None:
        raise ValueError('Feature list missing from replay group')
    X = group[features].to_numpy(dtype=np.float32, copy=False)
    y = group['outcome'].to_numpy(dtype=np.int8)
    duration = group['duration'].to_numpy(dtype=np.float64)
    prior_duration = group['mean_duration'].to_numpy(dtype=np.float64)
    probabilities = group['model_score'].to_numpy(dtype=np.float64)
    ids = group['test_id'].astype(str).to_numpy()
    n = len(group)
    if not n:
        return
    k = min(n, max(1, int(math.ceil(n * FRACTION))))
    order_model = np.lexsort((ids, -probabilities))
    history = (group['fails_last5'].to_numpy(dtype=float) * 10
               + group['fails_last20'].to_numpy(dtype=float)
               + group['fail_rate'].to_numpy(dtype=float))
    order_history = np.lexsort((ids, -history))
    random_order = rngs[key].permutation(n)
    policy = policies.setdefault(key, make_policy(key, X.shape[1]))
    bandit_selected = policy.select(X, probabilities, k)
    selections = {
        'random': random_order[:k],
        'history_combo': order_history[:k],
        'model': order_model[:k],
        'bandit': bandit_selected,
    }
    fail_count = int(y.sum())
    for method, indices in selections.items():
        state = states[(project, method)]
        state['test_rows'] += n
        state['run_stages'] += 1
        state['failure_rows'] += fail_count
        state['failing_run_stages'] += int(fail_count > 0)
        state['selected_tests'] += len(indices)
        caught = int(y[indices].sum())
        state['failures_caught'] += caught
        state['failing_run_stages_hit'] += int(caught > 0)
        state['total_duration'] += float(np.nan_to_num(duration, nan=0).sum())
        state['selected_duration'] += float(np.nan_to_num(duration[indices], nan=0).sum())

    # Feedback becomes available only after this run's selection and execution.
    policy.update(X[bandit_selected], probabilities[bandit_selected], y[bandit_selected],
                  duration[bandit_selected], prior_duration[bandit_selected])


def consume_chunk(chunk, pending, cutoffs, model, features, policies, last_run, rngs, states, final=False):
    if pending is not None:
        chunk = pd.concat([pending, chunk], ignore_index=True) if not chunk.empty else pending
    if chunk.empty:
        return None
    groups = list(chunk.groupby(RUN_KEY, sort=False, observed=True))
    if not groups:
        return None
    completed = groups if final else groups[:-1]
    for _, source in completed:
        key = (str(source['project'].iloc[0]), str(source['stage_id'].iloc[0]))
        cutoff = cutoffs.get(key)
        if cutoff is None or int(source['run_idx'].iloc[0]) < cutoff:
            continue
        source = source.copy()
        if source[features].isna().any().any():
            raise ValueError(f'Missing model features in replay candidates: {key}, run {source.run_idx.iloc[0]}')
        if source['duration'].isna().any():
            raise ValueError(f'Missing observed duration in bandit feedback: {key}, run {source.run_idx.iloc[0]}')
        source['model_score'] = model.predict_proba(source[features].to_numpy(dtype=np.float32))[:, 1]
        source.attrs['features'] = features
        rngs.setdefault(key, np.random.default_rng(zlib.crc32('|'.join(key).encode('utf-8')) ^ SEED))
        add_run(source, policies, last_run, rngs, states)
    return None if final else groups[-1][1].copy()


def summarize(states):
    rows = []
    for (project, method), state in states.items():
        caught = state['failures_caught']
        failures = state['failure_rows']
        selected = state['selected_tests']
        rows.append({
            'project': project, 'method': method, **state,
            'selection_pct': 100 * selected / state['test_rows'] if state['test_rows'] else float('nan'),
            'failure_recall_pct': 100 * caught / failures if failures else float('nan'),
            'precision_pct': 100 * caught / selected if selected else float('nan'),
            'failing_run_stages_hit_pct': 100 * state['failing_run_stages_hit'] / state['failing_run_stages'] if state['failing_run_stages'] else float('nan'),
            'duration_savings_proxy_pct': 100 * (1 - state['selected_duration'] / state['total_duration']) if state['total_duration'] else float('nan'),
        })
    frame = pd.DataFrame(rows)
    pooled_rows = []
    for method, group in frame.groupby('method', sort=False):
        summed = {column: group[column].sum() for column in (
            'test_rows', 'run_stages', 'failure_rows', 'failing_run_stages',
            'selected_tests', 'failures_caught', 'failing_run_stages_hit',
            'total_duration', 'selected_duration')}
        pooled_rows.append({'project': 'all_non_tvm', 'method': method, **summed})
    pooled = pd.DataFrame(pooled_rows)
    for row in pooled.to_dict('records'):
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    for idx, row in frame.iterrows():
        frame.loc[idx, 'selection_pct'] = 100 * row.selected_tests / row.test_rows if row.test_rows else float('nan')
        frame.loc[idx, 'failure_recall_pct'] = 100 * row.failures_caught / row.failure_rows if row.failure_rows else float('nan')
        frame.loc[idx, 'precision_pct'] = 100 * row.failures_caught / row.selected_tests if row.selected_tests else float('nan')
        frame.loc[idx, 'failing_run_stages_hit_pct'] = 100 * row.failing_run_stages_hit / row.failing_run_stages if row.failing_run_stages else float('nan')
        frame.loc[idx, 'duration_savings_proxy_pct'] = 100 * (1 - row.selected_duration / row.total_duration) if row.total_duration else float('nan')
    return frame


def write_report(results, model_payload):
    pooled = results.loc[results['project'].eq('all_non_tvm')].set_index('method')
    lines = [
        '# Non-TVM contextual bandit offline replay', '',
        'This is a chronological offline replay, not online deployment or evidence of real-world improvement. TVM was excluded before replay.', '',
        'Each action selects the top 10% of candidates within one project/stage run. The supervised model probability is the base score; an epsilon-greedy linear contextual correction (epsilon=0.1) updates after each run using feedback only from selected tests.', '',
        "Reward for selected test i is `failure_i - 0.1 * min(observed_duration_i / max(prior_mean_duration_i, 1s), 5) / 5`. The test outcome and duration are revealed only after selection. The scaler is fit using the first run's candidate contexts per project/stage and then frozen.", '',
        '## Pooled held-out results', '',
        '| Method | Failures caught (%) | Precision (%) | Failing stages hit (%) | Duration savings proxy (%) |',
        '|---|---:|---:|---:|---:|',
    ]
    for method in ('random', 'history_combo', 'model', 'bandit'):
        row = pooled.loc[method]
        lines.append(f"| {method} | {row.failure_recall_pct:.2f} | {row.precision_pct:.2f} | {row.failing_run_stages_hit_pct:.2f} | {row.duration_savings_proxy_pct:.2f} |")
    bandit = pooled.loc['bandit']
    model = pooled.loc['model']
    lines += ['', f"The replay selected {int(bandit.selected_tests):,} of {int(bandit.test_rows):,} held-out candidates ({bandit.selection_pct:.2f}%). Model training excluded project `tvm`; model artifact version is `{model_payload['version']}`.", '', 'The replay simulates feedback from historical outcomes and durations. Real CI integration, delayed/partial feedback behavior, exploration safeguards, and online monitoring have not been validated.', '']
    REPORT_PATH.write_text('\n'.join(lines), encoding='utf-8')


def main():
    if not MODEL_PATH.is_file() or not SPLIT_PATH.is_file():
        raise FileNotFoundError('Train and evaluate the non-TVM supervised model before bandit replay')
    payload = joblib.load(MODEL_PATH)
    if 'tvm' in {p.lower() for p in payload['training_projects']}:
        raise AssertionError('TVM appears in supervised model training projects')
    model, features = payload['model'], payload['features']
    splits = pd.read_csv(SPLIT_PATH, dtype={'project': 'string', 'stage_id': 'string'})
    cutoffs = compute_cutoffs(splits)
    usecols = list(dict.fromkeys(RUN_KEY + ['test_id', 'outcome', 'duration'] + features))
    policies, last_run, rngs = {}, {}, {}
    states = defaultdict(lambda: {
        'test_rows': 0, 'run_stages': 0, 'failure_rows': 0, 'failing_run_stages': 0,
        'selected_tests': 0, 'failures_caught': 0, 'failing_run_stages_hit': 0,
        'total_duration': 0.0, 'selected_duration': 0.0,
    })
    pending = None
    for chunk in pd.read_csv(DATA / 'lrts_test_modeling_v1.csv.gz', usecols=usecols,
                             chunksize=250_000,
                             dtype={'project': 'string', 'stage_id': 'string', 'test_id': 'string'}):
        if chunk['project'].astype(str).str.lower().eq('tvm').all():
            continue
        chunk = chunk.loc[chunk['project'].astype(str).str.lower().ne('tvm')].copy()
        chunk['_cutoff'] = [cutoffs.get((str(project), str(stage)), np.inf)
                            for project, stage in zip(chunk['project'], chunk['stage_id'])]
        chunk = chunk.loc[chunk['run_idx'].to_numpy() >= chunk['_cutoff'].to_numpy()]
        chunk = chunk.drop(columns='_cutoff')
        pending = consume_chunk(chunk, pending, cutoffs, model, features, policies, last_run, rngs, states, final=False)
    if pending is not None:
        consume_chunk(pd.DataFrame(), pending, cutoffs, model, features, policies, last_run, rngs, states, final=True)
    results = summarize(states)
    results['model_artifact_sha256'] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_PATH, index=False)
    write_report(results, payload)
    if results['project'].astype(str).str.lower().eq('tvm').any():
        raise AssertionError('TVM metrics found in bandit replay output')
    print(results.loc[results['project'].eq('all_non_tvm')].to_string(index=False))
    print(f'Saved {OUT_PATH} and {REPORT_PATH}')


if __name__ == '__main__':
    main()

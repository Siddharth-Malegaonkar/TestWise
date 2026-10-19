"""Add selection precision summaries and write the final non-TVM report."""
import json
import math
import hashlib
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
EVAL = ARTIFACTS / 'evaluation'
DATA = ROOT / 'data' / 'processed' / 'lrts_v1'


def main():
    results_path = EVAL / 'lrts_non_tvm_results.csv'
    audit_path = DATA / 'lrts_run_audit_v1.csv.gz'
    manifest_path = EVAL / 'non_tvm_evaluation_manifest.json'
    if not results_path.is_file():
        raise FileNotFoundError(f'Run scripts/evaluate_model.py first: {results_path}')
    results = pd.read_csv(results_path)
    audit = pd.read_csv(audit_path, dtype={'project': 'string', 'stage_id': 'string'})
    audit = audit.loc[audit['project'].astype(str).str.lower().ne('tvm')].copy()
    splits = pd.read_csv(EVAL / 'non_tvm_time_split_plan.csv', dtype={'project': 'string', 'stage_id': 'string'})
    cutoffs = {(r.project, r.stage_id): r.cutoff_run_idx for r in splits.itertuples() if pd.notna(r.cutoff_run_idx)}
    selected_counts = {fraction: {} for fraction in (5, 10, 20)}
    eligible_counts = {}
    for (project, stage), group in audit.groupby(['project', 'stage_id'], observed=True):
        cutoff = cutoffs.get((project, stage))
        if cutoff is None:
            continue
        heldout = group.loc[group['run_idx'] >= cutoff]
        project = str(project)
        eligible_counts[project] = eligible_counts.get(project, 0) + int(heldout['normalized_test_count'].sum())
        for pct in selected_counts:
            count = sum(max(1, math.ceil(int(n) * pct / 100)) for n in heldout['normalized_test_count'])
            selected_counts[pct][project] = selected_counts[pct].get(project, 0) + count

    for row in results.itertuples():
        project = str(row.project)
        if project == 'all_non_tvm':
            eligible = sum(eligible_counts.values())
            selected = {pct: sum(per_project.values()) for pct, per_project in selected_counts.items()}
        else:
            eligible = eligible_counts.get(project, 0)
            selected = {pct: selected_counts[pct].get(project, 0) for pct in selected_counts}
        if eligible != int(row.test_rows):
            raise ValueError(f'Held-out row-count mismatch for {project}: audit={eligible}, metrics={row.test_rows}')
        index = results.index[results['project'].astype(str).eq(project) & results['method'].eq(row.method)]
        if not len(index):
            continue
        idx = index[0]
        results.loc[idx, 'full_suite_selected_pct'] = 100.0
        results.loc[idx, 'full_suite_failures_caught_pct'] = 100.0 if row.failure_rows else float('nan')
        for pct in selected:
            caught_pct = getattr(row, f'failures_caught_at_{pct}pct')
            caught = row.failure_rows * caught_pct / 100 if pd.notna(caught_pct) else 0.0
            results.loc[idx, f'missed_failures_at_{pct}pct'] = max(0.0, row.failure_rows - caught)
            precision = caught / selected[pct] if selected[pct] else 0.0
            recall = caught / row.failure_rows if row.failure_rows else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            results.loc[idx, f'precision_at_{pct}pct'] = 100 * precision
            results.loc[idx, f'global_recall_at_{pct}pct'] = 100 * recall
            results.loc[idx, f'f1_at_{pct}pct'] = 100 * f1
            results.loc[idx, f'selection_pct_at_{pct}pct'] = 100 * selected[pct] / eligible if eligible else 0.0

    results.to_csv(results_path, index=False)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    data_manifest = json.loads((DATA / 'lrts_manifest_v1.json').read_text(encoding='utf-8'))
    model_path = ARTIFACTS / 'models' / 'testwise_lrts_non_tvm.joblib'
    model_payload = joblib.load(model_path)
    model_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model_metadata = {
        key: model_payload[key] for key in (
            'version', 'target', 'features', 'training_projects', 'excluded_projects',
            'training_rows', 'training_positive_rows', 'negative_sample_rate', 'seed', 'model_config',
        )
    }
    model_metadata['training_data'] = 'data/processed/lrts_v1/lrts_test_modeling_v1.csv.gz'
    model_metadata['library_versions'] = {
        'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__,
        'joblib': joblib.__version__,
    }
    model_metadata['evaluation'] = {
        'protocol': manifest['split'],
        'evaluation_manifest': 'artifacts/evaluation/non_tvm_evaluation_manifest.json',
        'results_csv': 'artifacts/evaluation/lrts_non_tvm_results.csv',
    }
    metadata_path = ARTIFACTS / 'models' / 'testwise_lrts_non_tvm.metadata.json'
    metadata_path.write_text(json.dumps(model_metadata, indent=2), encoding='utf-8')
    projects = [p for p in data_manifest['project_summary'] if p != 'tvm']
    included_rows = sum(data_manifest['project_summary'][p]['test_rows'] for p in projects)
    included_failures = sum(data_manifest['project_summary'][p]['failures'] for p in projects)
    pooled = results.loc[(results['project'] == 'all_non_tvm') & (results['method'].isin(['random', 'history_combo', 'path_overlap', 'model', 'model_per_prior_second']))]
    report = [
        '# TestWise non-TVM evaluation',
        '',
        '## Scope and objective',
        '',
        'The model scores the probability of an observed per-test `outcome` failure. It does not estimate code-change relevance and it does not execute tests.',
        '',
        f"Included projects: {', '.join(projects)}. The source audit has 10 projects before filtering and 9 after filtering; TVM is excluded before split construction, training, and evaluation.",
        '',
        f"The non-TVM source population contains {included_rows:,} test rows and {included_failures:,} failures ({100 * included_failures / included_rows:.3f}% failure rate).",
        '',
        f"Training used {manifest['train_population_rows_after_split_and_pr_purge']:,} eligible non-TVM historical rows; the sampled fit contained {manifest['train_sample_rows']:,} rows ({manifest['train_positive_rows']:,} failures) with inverse-probability weights for sampled negatives.",
        '',
        '## Evaluation protocol',
        '',
        'Chronological 70/30 split within each project and stage by run index. Training PRs that also appear in any held-out stage for that project are purged. All nine included projects have complete test-count reconciliation in the source run audit. The final estimator is a single-worker RandomForestClassifier (48 trees, depth capped at 18, minimum leaf size 10); the existing HistGradientBoosting implementation could not fit under this Windows sandbox because scikit-learn thread-pool creation was denied.',
        '',
        'Ranking results are aggregated from held-out runs. Precision/F1 and selection percentage are derived from measured failure catches and the exact test counts selected at each per-run budget. Full-suite execution is a reference baseline: 100% of tests selected and 100% of recorded failures observed by definition; it has no test-time saving.',
        '',
        'Row-wise average precision/PR-AUC, fixed-threshold classification metrics, and probability calibration were not computed; the executed evaluation focuses on within-run ranking and budgeted failure detection.',
        '',
        '## Pooled held-out results',
        '',
        '| Method | APFD | Recall @ 5% (mean/run) | Failures caught @ 5% | Precision @ 5% | F1 @ 5% | Failures caught @ 10% | Failures caught @ 20% | Time saved proxy @ 10% |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in pooled.itertuples():
        fmt = lambda x: '—' if pd.isna(x) else f'{x:.2f}'
        report.append(f'| {row.method} | {fmt(row.apfd)} | {fmt(row.recall_at_5pct)} | {fmt(row.failures_caught_at_5pct)}% | {fmt(row.precision_at_5pct)}% | {fmt(row.f1_at_5pct)}% | {fmt(row.failures_caught_at_10pct)}% | {fmt(row.failures_caught_at_20pct)}% | {fmt(row.time_saved_at_10pct)}% |')
    model = next((row for row in pooled.itertuples() if row.method == 'model'), None)
    best_baseline = max((row for row in pooled.itertuples() if row.method != 'model'), key=lambda r: r.apfd if pd.notna(r.apfd) else -1, default=None)
    report += ['', '## Interpretation and limitations', '']
    if model and best_baseline and pd.notna(model.apfd) and pd.notna(best_baseline.apfd):
        direction = 'higher' if model.apfd > best_baseline.apfd else 'not higher than'
        report.append(f"The model APFD is {direction} the strongest listed non-model baseline ({best_baseline.method}: {best_baseline.apfd:.3f}; model: {model.apfd:.3f}). At 5% selection, the model catches {model.failures_caught_at_5pct:.2f}% of failures versus {best_baseline.failures_caught_at_5pct:.2f}% for that baseline. The model's 10% duration-savings proxy ({model.time_saved_at_10pct:.2f}%) is lower than history_combo ({next((r.time_saved_at_10pct for r in pooled.itertuples() if r.method == 'history_combo'), float('nan')):.2f}%). The APFD gap is small and no uncertainty interval was computed.")
    bandit_path = EVAL / 'non_tvm_bandit_replay.csv'
    if bandit_path.is_file():
        bandit_results = pd.read_csv(bandit_path)
        if 'model_artifact_sha256' in bandit_results and bandit_results['model_artifact_sha256'].eq(model_digest).all():
            bandit_row = bandit_results.loc[(bandit_results['project'] == 'all_non_tvm') & (bandit_results['method'] == 'bandit')].iloc[0]
            model_row = bandit_results.loc[(bandit_results['project'] == 'all_non_tvm') & (bandit_results['method'] == 'model')].iloc[0]
            report.append(f"The separate 10% chronological bandit replay caught {bandit_row.failure_recall_pct:.2f}% of failures and had {bandit_row.duration_savings_proxy_pct:.2f}% duration savings proxy, versus {model_row.failure_recall_pct:.2f}% and {model_row.duration_savings_proxy_pct:.2f}% for the supervised model. It trades failure catch for lower selected duration and is not selected as the default policy.")
        else:
            report.append('A prior bandit replay exists but does not match the current model artifact; rerun `scripts/replay_bandit_evaluation.py` before using it.')
    report += [
        'The duration-based measure is a selected-duration savings proxy, not an execution-order simulator. Partial execution can miss failures; the full-suite baseline remains the complete test run.',
        'The fitted model is validated only on these nine LRTS projects. Historical features require real prior CI results. New repositories need a test identifier mapping, current change features, and a policy for absent history; the prediction module rejects missing feature values rather than inventing them.',
        'The contextual-bandit result is an offline replay only. No online learning or real-world improvement is claimed.',
        '',
        '## Artifacts',
        '',
        '- Model: `artifacts/models/testwise_lrts_non_tvm.joblib`',
        '- Model metadata: `artifacts/models/testwise_lrts_non_tvm.metadata.json`',
        '- Result table: `artifacts/evaluation/lrts_non_tvm_results.csv`',
        '- Split plan: `artifacts/evaluation/non_tvm_time_split_plan.csv`',
        '- Evaluation manifest: `artifacts/evaluation/non_tvm_evaluation_manifest.json`',
        '- Bandit replay: `artifacts/evaluation/non_tvm_bandit_replay.csv` and `docs/bandit_replay.md`',
        '- Prediction API: `ml/prediction.py`',
        '',
    ]
    (ROOT / 'docs' / 'final_evaluation.md').write_text('\n'.join(report), encoding='utf-8')
    print(f'Updated {results_path} and wrote docs/final_evaluation.md')


if __name__ == '__main__':
    main()

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROJ = sys.argv[1] if len(sys.argv) > 1 else 'activemq'
ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / 'data' / 'processed' / f'features_{PROJ}.csv.gz'
MODEL_PATH = ROOT / 'models' / 'model.joblib'
RANKINGS_PATH = ROOT / 'data' / 'processed' / f'ranked_tests_{PROJ}.csv'

# Keep the same feature sets used by the baseline evaluation script.
HIST = ['last_outcome', 'prior_runs', 'prior_fails', 'fail_rate', 'fails_last5',
        'fails_last20', 'runs_since_fail', 'last_duration', 'mean_duration']
CODE = ['test_class_changed', 'same_pkg', 'pkg_overlap_depth', 'name_match',
        'n_files', 'additions', 'deletions', 'n_java', 'n_test_files', 'n_pom']
FEATURES = HIST + CODE


def split_by_build_order(df):
    # Older builds train the model; newer builds are used for evaluation.
    runs = df[['run_idx', 'pr_name']].drop_duplicates().sort_values('run_idx')
    if len(runs) < 2:
        raise ValueError('Need at least two build orders to create a train/test split.')

    split_idx = runs.run_idx.iloc[int(len(runs) * 0.7)]
    test = df[df.run_idx >= split_idx].reset_index(drop=True)
    train = df[(df.run_idx < split_idx) & (~df.pr_name.isin(set(test.pr_name)))].copy()
    return train, test


def evaluate_rankings(frame, score, seed=0):
    # This matches the APFD / recall@k metric used by evaluate_baselines.py.
    rng = np.random.default_rng(seed)
    d = frame[['run_idx', 'outcome']].copy()
    d['score'] = score
    d['rnd'] = rng.random(len(d))

    rows = []
    for _, group in d.groupby('run_idx', sort=False):
        fail_count = int(group.outcome.sum())
        if fail_count == 0:
            continue

        ordered = group.sort_values(['score', 'rnd'], ascending=[False, True])
        positions = np.flatnonzero(ordered.outcome.values == 1) + 1
        n = len(ordered)
        row = {'apfd': 1 - positions.sum() / (n * fail_count) + 1 / (2 * n)}
        for pct in (5, 10, 20):
            row[f'recall@{pct}%'] = (positions <= np.ceil(n * pct / 100)).mean()
        rows.append(row)

    if not rows:
        return pd.Series({
            'apfd': 0.0,
            'recall@5%': 0.0,
            'recall@10%': 0.0,
            'recall@20%': 0.0,
        })

    return pd.DataFrame(rows).mean()


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f'Feature table not found: {FEATURES_PATH}')

    df = pd.read_csv(FEATURES_PATH)
    train, test = split_by_build_order(df)

    # Failures are rare, so use a balanced forest to avoid overfitting to the majority class.
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=0,
        class_weight='balanced',
        n_jobs=-1,
    )
    model.fit(train[FEATURES], train.outcome)

    test = test.copy()
    test['failure_probability'] = model.predict_proba(test[FEATURES])[:, 1]

    # Rank tests in each build by likelihood of failure.
    ranked = []
    for _, group in test.groupby('run_idx', sort=False):
        ordered = group.sort_values(['failure_probability', 'testclass'], ascending=[False, True]).reset_index(drop=True)
        ordered['rank_in_build'] = np.arange(1, len(ordered) + 1)
        ranked.append(ordered)
    ranked_df = pd.concat(ranked, ignore_index=True)

    # Save the ranking table for inspection and downstream analysis.
    RANKINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranked_df.to_csv(RANKINGS_PATH, index=False)

    # Compare the model with a trivial random baseline using the same metric as the baselines.
    rng = np.random.default_rng(0)
    metrics = pd.DataFrame({
        'random': evaluate_rankings(test, rng.random(len(test)), seed=1),
        'model': evaluate_rankings(test, test.failure_probability.values, seed=1),
    }).T.round(3)
    print('\nmean over failing test runs:')
    print(metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f'\nSaved model to {MODEL_PATH}')


if __name__ == '__main__':
    main()

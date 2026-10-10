import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJ = sys.argv[1] if len(sys.argv) > 1 else 'activemq'
ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / 'data' / 'processed' / f'features_{PROJ}.csv.gz'
MODEL_PATH = ROOT / 'models' / 'model.joblib'


def split_test_builds(df):
    # Match the time-based split used by the baselines: older runs train, newer runs test.
    runs = df[['run_idx', 'pr_name']].drop_duplicates().sort_values('run_idx')
    cut = runs.run_idx.iloc[int(len(runs) * 0.7)]
    test = df[df.run_idx >= cut].reset_index(drop=True)
    return test


def evaluate_rankings(frame, score, seed=0):
    # Same metric as evaluate_baselines.py: APFD and recall@5/10/20 over failing test runs.
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
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Saved model not found: {MODEL_PATH}. Run scripts/train_model.py first.')

    print(f'Loading features from {FEATURES_PATH}')
    print(f'Loading model from {MODEL_PATH}')
    df = pd.read_csv(FEATURES_PATH)
    test = split_test_builds(df)

    model = joblib.load(MODEL_PATH)
    feature_names = list(model.feature_names_in_)

    # Score each test build with the same baselines and the trained model.
    # The processed feature table does not include the raw per-test duration column,
    # so the fastest-test baseline uses the available duration proxy instead.
    random_scores = np.random.default_rng(0).random(len(test))
    recent_failure_scores = test.fails_last5.values
    if 'duration' in test.columns:
        fastest_test_scores = -test['duration'].values
    elif 'last_duration' in test.columns:
        fastest_test_scores = -test['last_duration'].values
    elif 'mean_duration' in test.columns:
        fastest_test_scores = -test['mean_duration'].values
    else:
        raise KeyError('No duration-like feature available for the fastest-test baseline.')
    model_scores = model.predict_proba(test[feature_names])[:, 1]

    scores = {
        'random': random_scores,
        'recent_failure': recent_failure_scores,
        'fastest_test': fastest_test_scores,
        'model': model_scores,
    }

    table = pd.DataFrame({
        name: evaluate_rankings(test, score, seed=1)
        for name, score in scores.items()
    }).T.round(3)

    # Print one compact table so all methods can be compared directly.
    print('\nComparison of ranking methods on the same test builds')
    print(table[['apfd', 'recall@5%', 'recall@10%', 'recall@20%']])


if __name__ == '__main__':
    main()

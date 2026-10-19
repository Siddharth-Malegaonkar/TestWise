import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from bandit import EpsilonGreedyBandit

PROJ = sys.argv[1] if len(sys.argv) > 1 else 'activemq'
ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / 'data' / 'processed' / f'features_{PROJ}.csv.gz'
MODEL_PATH = ROOT / 'models' / 'model.joblib'
RESULTS_PATH = ROOT / 'data' / 'processed' / 'replay_results.csv'
N_SEEDS = 3


def split_test_builds(df):
    # Match the same time-based split used by the model-building workflow.
    runs = df[['run_idx', 'pr_name']].drop_duplicates().sort_values('run_idx')
    cut = runs.run_idx.iloc[int(len(runs) * 0.7)]
    return df[df.run_idx >= cut].reset_index(drop=True)


def choose_top_k(scores, k):
    # Rank tests by score, highest first, and keep the best k tests.
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return np.array([], dtype=int)
    k = min(max(1, int(np.ceil(k))), len(scores))
    return np.argsort(scores)[-k:][::-1]


def choose_random_k(n_tests, k, rng):
    # Random subset of size k without replacement.
    if n_tests == 0:
        return np.array([], dtype=int)
    k = min(max(1, int(np.ceil(k))), n_tests)
    return rng.choice(n_tests, size=k, replace=False)


def mask_from_indices(n_tests, indices):
    mask = np.zeros(n_tests, dtype=bool)
    if len(indices) > 0:
        mask[np.asarray(indices, dtype=int)] = True
    return mask


def compute_pct(caught, total):
    return 0.0 if total == 0 else caught / total


def summarize_by_quartile(df):
    # Report performance on the first and last 25% of failing builds in time order.
    if df.empty:
        return 0.0, 0.0
    q = max(1, int(np.ceil(len(df) * 0.25)))
    first = df.head(q)
    last = df.tail(q)
    first_pct = first['failures_caught'].sum() / first['failures_total'].sum() if first['failures_total'].sum() else 0.0
    last_pct = last['failures_caught'].sum() / last['failures_total'].sum() if last['failures_total'].sum() else 0.0
    return first_pct, last_pct


def prepare_builds(test, model, feature_names):
    # Run the expensive model prediction once per build; reuse it for every replay method.
    builds = []
    for run_idx, group in test.groupby('run_idx', sort=True):
        X = group[feature_names].to_numpy(dtype=float)
        y = group['outcome'].to_numpy(dtype=int)
        model_probs = model.predict_proba(X)[:, 1]
        fails_last5 = group['fails_last5'].fillna(0).to_numpy(dtype=float)
        builds.append({
            'run_idx': run_idx,
            'X': X,
            'y': y,
            'model_probs': model_probs,
            'fails_last5': fails_last5,
        })
    return builds


def replay_method(method_name, prepared_builds, rng, bandit=None, k_fraction=0.10):
    # Replay one approach across all test builds and return per-build results.
    results = []
    for build_num, build in enumerate(prepared_builds, start=1):
        X = build['X']
        y = build['y']
        model_probs = build['model_probs']
        failures_total = int(y.sum())

        if failures_total == 0:
            if bandit is not None:
                k = max(1, int(np.ceil(k_fraction * len(X))))
                selected = bandit.select(X, k, model_probabilities=model_probs)
                bandit.update(X[selected], y[selected], model_probabilities=model_probs[selected])
            continue

        k = max(1, int(np.ceil(k_fraction * len(X))))
        if method_name == 'random':
            idxs = choose_random_k(len(X), k, rng)
        elif method_name == 'recent_failure':
            idxs = choose_top_k(build['fails_last5'], k)
        elif method_name == 'model':
            idxs = choose_top_k(model_probs, k)
        elif bandit is not None:
            idxs = bandit.select(X, k, model_probabilities=model_probs)
        else:
            raise ValueError(f'Unknown method: {method_name}')

        selected_mask = mask_from_indices(len(X), idxs)
        caught = int((selected_mask & (y == 1)).sum())
        results.append({
            'build_num': build_num,
            'run_idx': build['run_idx'],
            'method': method_name,
            'failures_total': failures_total,
            'failures_caught': caught,
            'caught_pct': compute_pct(caught, failures_total),
        })

        if bandit is not None:
            bandit.update(X[idxs], y[idxs], model_probabilities=model_probs[idxs])

    return pd.DataFrame(results)


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
    feature_names = [c for c in model.feature_names_in_ if c in test.columns]
    if not feature_names:
        raise ValueError('No model feature columns found in the feature table.')

    prepared_builds = prepare_builds(test, model, feature_names)

    rng = np.random.default_rng(0)
    configs = [
        {'method': 'bandit_lr0_eps0', 'learning_rate': 0.0, 'epsilon': 0.0},
        {'method': 'bandit_lr0_eps0.1', 'learning_rate': 0.0, 'epsilon': 0.1},
        {'method': 'bandit_lr0.001_eps0.1', 'learning_rate': 0.001, 'epsilon': 0.1},
        {'method': 'bandit_lr0.01_eps0.1', 'learning_rate': 0.01, 'epsilon': 0.1},
    ]

    all_results = []
    methods = ['random', 'recent_failure', 'model']
    for method_name in methods:
        out = replay_method(method_name, prepared_builds, rng)
        all_results.append(out)

    for cfg in configs:
        bandit = EpsilonGreedyBandit(
            n_features=len(feature_names),
            epsilon=cfg['epsilon'],
            learning_rate=cfg['learning_rate'],
            random_state=0,
        )
        out = replay_method(cfg['method'], prepared_builds, rng, bandit=bandit)
        all_results.append(out)

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(columns=[
        'build_num', 'run_idx', 'method', 'failures_total', 'failures_caught', 'caught_pct'])

    if results_df.empty:
        print('No failing builds were found in the replay window.')
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(RESULTS_PATH, index=False)
        return

    method_order = ['random', 'recent_failure', 'model',
                    'bandit_lr0_eps0', 'bandit_lr0_eps0.1',
                    'bandit_lr0.001_eps0.1', 'bandit_lr0.01_eps0.1']
    summary_rows = []
    for method in method_order:
        subset = results_df[results_df['method'] == method].copy()
        if subset.empty:
            summary_rows.append({'method': method, 'total_pct': 0.0, 'first_25_pct': 0.0, 'last_25_pct': 0.0})
            continue
        total_caught = int(subset['failures_caught'].sum())
        total_failures = int(subset['failures_total'].sum())
        first_q, last_q = summarize_by_quartile(subset[['run_idx', 'failures_total', 'failures_caught']].copy().assign(
            caught=subset['failures_caught'],
            total=subset['failures_total'],
        ))
        summary_rows.append({
            'method': method,
            'total_pct': compute_pct(total_caught, total_failures) * 100,
            'first_25_pct': first_q * 100,
            'last_25_pct': last_q * 100,
        })

    summary = pd.DataFrame(summary_rows).set_index('method').reindex(method_order)
    print('Failure catch rate (%) by method')
    print(summary[['total_pct', 'first_25_pct', 'last_25_pct']].round(3))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f'\nSaved per-build replay results to {RESULTS_PATH}')

    # Bandit seed sweep after the main table, using only the requested settings.
    bandit_settings = [
        {'method': 'bandit_lr0_eps0.1', 'learning_rate': 0.0, 'epsilon': 0.1},
        {'method': 'bandit_lr0.001_eps0.1', 'learning_rate': 0.001, 'epsilon': 0.1},
    ]
    print('\nBandit seed sweep')
    model_only = replay_method('model', prepared_builds, np.random.default_rng(0))
    model_total = model_only['failures_caught'].sum() / model_only['failures_total'].sum()
    print(f'model_only: {model_total * 100:.3f}%')

    for cfg in bandit_settings:
        vals = []
        for seed in range(N_SEEDS):
            start = time.perf_counter()
            bandit = EpsilonGreedyBandit(
                n_features=len(feature_names),
                epsilon=cfg['epsilon'],
                learning_rate=cfg['learning_rate'],
                random_state=seed,
            )
            out = replay_method(cfg['method'], prepared_builds, np.random.default_rng(seed), bandit=bandit)
            total_caught = int(out['failures_caught'].sum())
            total_failures = int(out['failures_total'].sum())
            val = compute_pct(total_caught, total_failures) * 100
            vals.append(val)
            elapsed = time.perf_counter() - start
            print(f"  seed {seed + 1}/{N_SEEDS}: {cfg['method']} -> {val:.3f}% | seconds={elapsed:.2f}")
        print(f"  {cfg['method']}: avg={np.mean(vals):.3f}%, min={np.min(vals):.3f}%, max={np.max(vals):.3f}%")

    # Quick K-sweep for the main comparison rows requested in the handoff.
    k_values = [0.05, 0.10, 0.20]
    bandit_configs = [
        {'method': 'model_only', 'learning_rate': None, 'epsilon': None},
        {'method': 'bandit_lr0.001_eps0.1', 'learning_rate': 0.001, 'epsilon': 0.1},
        {'method': 'bandit_lr0.1_eps0.1', 'learning_rate': 0.1, 'epsilon': 0.1},
        {'method': 'bandit_lr1.0_eps0.1', 'learning_rate': 1.0, 'epsilon': 0.1},
        {'method': 'bandit_lr0.001_eps0.3', 'learning_rate': 0.001, 'epsilon': 0.3},
    ]
    summary_rows = []
    for cfg in bandit_configs:
        method_name = cfg['method']
        if method_name == 'model_only':
            row = {'method': method_name}
            for k_fraction in k_values:
                out = replay_method('model', prepared_builds, np.random.default_rng(0), k_fraction=k_fraction)
                total_caught = int(out['failures_caught'].sum())
                total_failures = int(out['failures_total'].sum())
                row[f'k{int(k_fraction * 100)}_pct'] = compute_pct(total_caught, total_failures) * 100
                if np.isclose(k_fraction, 0.10):
                    row['k10_failures_caught'] = total_caught
                    row['k10_failures_total'] = total_failures
            summary_rows.append(row)
            continue

        k_to_values = {}
        k10_caught = []
        k10_total = []
        for k_fraction in k_values:
            k_to_values[k_fraction] = []
        for seed in range(5):
            bandit = EpsilonGreedyBandit(
                n_features=len(feature_names),
                epsilon=cfg['epsilon'],
                learning_rate=cfg['learning_rate'],
                random_state=seed,
            )
            for k_fraction in k_values:
                out = replay_method('bandit', prepared_builds, np.random.default_rng(seed), bandit=bandit, k_fraction=k_fraction)
                total_caught = int(out['failures_caught'].sum())
                total_failures = int(out['failures_total'].sum())
                k_to_values[k_fraction].append(compute_pct(total_caught, total_failures) * 100)
                if np.isclose(k_fraction, 0.10):
                    k10_caught.append(total_caught)
                    k10_total.append(total_failures)

        row = {'method': method_name}
        for k_fraction in k_values:
            row[f'k{int(k_fraction * 100)}_pct'] = float(np.mean(k_to_values[k_fraction]))
        row['k10_failures_caught'] = int(np.mean(k10_caught))
        row['k10_failures_total'] = int(np.mean(k10_total))
        summary_rows.append(row)

    table = pd.DataFrame(summary_rows)
    print('\nK-sweep summary (mean over 5 seeds)')
    print(table[['method', 'k5_pct', 'k10_pct', 'k20_pct', 'k10_failures_caught', 'k10_failures_total']].to_string(index=False))


if __name__ == '__main__':
    main()

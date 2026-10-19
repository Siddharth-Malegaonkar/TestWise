import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

PROJ = sys.argv[1] if len(sys.argv) > 1 else 'activemq'
df = pd.read_csv(f'data/processed/features_{PROJ}.csv.gz')

HIST = ['last_outcome', 'prior_runs', 'prior_fails', 'fail_rate', 'fails_last5',
        'fails_last20', 'runs_since_fail', 'last_duration', 'mean_duration']
CODE = ['test_class_changed', 'same_pkg', 'pkg_overlap_depth', 'name_match',
        'n_files', 'additions', 'deletions', 'n_java', 'n_test_files', 'n_pom']

# time split, purging PRs that appear on both sides
runs = df[['run_idx', 'pr_name']].drop_duplicates().sort_values('run_idx')
cut = runs.run_idx.iloc[int(len(runs) * 0.7)]
test = df[df.run_idx >= cut]
train = df[(df.run_idx < cut) & (~df.pr_name.isin(set(test.pr_name)))]

# keep only "new" failures: drop failures of classes that failed in their last 20 appearances
def drop_repeats(d):
    return d[~((d.outcome == 1) & (d.fails_last20 > 0))].reset_index(drop=True)
train, test = drop_repeats(train), drop_repeats(test)
for name, d in (('train', train), ('test', test)):
    print(f'{name}: {len(d)} rows, {int(d.outcome.sum())} new failures in '
          f'{d[d.outcome == 1].run_idx.nunique()} runs')

scores = {
    'random': np.zeros(len(test)),
    'history_combo': (test.fails_last5 * 10 + test.fails_last20 + test.fail_rate).values,
    'code_rule': (test.test_class_changed * 3 + test.name_match * 2 + test.same_pkg).values,
}
w = np.where(train.outcome == 1, 20, 1)
for name, feats in (('model_history_only', HIST), ('model_code_only', CODE), ('model_all', HIST + CODE)):
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=0)
    m.fit(train[feats], train.outcome, sample_weight=w)
    scores[name] = m.predict_proba(test[feats])[:, 1]

def evaluate(score, seed=0):
    rng = np.random.default_rng(seed)
    d = test[['run_idx', 'outcome']].copy()
    d['score'] = score
    d['rnd'] = rng.random(len(d))
    out = []
    for _, g in d.groupby('run_idx', sort=False):
        m = int(g.outcome.sum())
        if m == 0:
            continue
        g = g.sort_values(['score', 'rnd'], ascending=[False, True])
        pos = np.flatnonzero(g.outcome.values == 1) + 1
        n = len(g)
        row = {'apfd': 1 - pos.sum() / (n * m) + 1 / (2 * n)}
        for p in (5, 10, 20):
            row[f'recall@{p}%'] = (pos <= np.ceil(n * p / 100)).mean()
        out.append(row)
    return pd.DataFrame(out).mean()

print('\nmean over test runs with a new failure:')
print(pd.DataFrame({k: evaluate(v) for k, v in scores.items()}).T.round(3))
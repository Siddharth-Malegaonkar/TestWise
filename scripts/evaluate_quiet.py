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

runs = df[['run_idx', 'pr_name']].drop_duplicates().sort_values('run_idx')
cut = runs.run_idx.iloc[int(len(runs) * 0.7)]
test = df[df.run_idx >= cut].reset_index(drop=True)
train = df[(df.run_idx < cut) & (~df.pr_name.isin(set(test.pr_name)))]   # no filtering on outcome

w = np.where(train.outcome == 1, 20, 1)
test['random'] = 0.0
test['fail_rate_only'] = test.fail_rate
test['code_rule'] = test.test_class_changed * 3 + test.name_match * 2 + test.same_pkg
for name, feats in (('model_history_only', HIST), ('model_code_only', CODE), ('model_all', HIST + CODE)):
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, random_state=0)
    m.fit(train[feats], train.outcome, sample_weight=w)
    test[name] = m.predict_proba(test[feats])[:, 1]
COLS = ['random', 'fail_rate_only', 'code_rule', 'model_history_only', 'model_code_only', 'model_all']

def evaluate(d, col, seed=0):
    rng = np.random.default_rng(seed)
    g_all = d[['run_idx', 'outcome', col]].copy()
    g_all['rnd'] = rng.random(len(g_all))
    out = []
    for _, g in g_all.groupby('run_idx', sort=False):
        m = int(g.outcome.sum())
        if m == 0:
            continue
        g = g.sort_values([col, 'rnd'], ascending=[False, True])
        pos = np.flatnonzero(g.outcome.values == 1) + 1
        n = len(g)
        row = {'apfd': 1 - pos.sum() / (n * m) + 1 / (2 * n)}
        for p in (5, 10, 20):
            row[f'recall@{p}%'] = (pos <= np.ceil(n * p / 100)).mean()
        out.append(row)
    return pd.DataFrame(out).mean()

for title, mask in (('A: classes with no failure in last 20 appearances', test.fails_last20 == 0),
                    ('B: classes that have never failed before', test.prior_fails == 0)):
    d = test[mask]
    f = d[d.outcome == 1]
    print(f'\n=== {title} ===')
    print(f'{len(d)} rows | {len(f)} failures in {f.run_idx.nunique()} runs')
    if len(f):
        print(pd.DataFrame({c: evaluate(d, c) for c in COLS}).T.round(3))
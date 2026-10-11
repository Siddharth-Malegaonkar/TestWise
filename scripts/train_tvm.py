import io, json, re, sys
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

STAGE = sys.argv[1] if len(sys.argv) > 1 else 'tvm-cpu'
ROOT = 'long_running_test_suites/'
z = ZipFile('data/raw/long_running_test_suites.zip')

# ---- 1. runs, oldest first
df = pd.read_csv(z.open(ROOT + 'dataset.csv'))
runs = df[(df.project == 'tvm') & (df.stage_id.astype(str) == STAGE)].copy()
runs = runs.sort_values('build_timestamp', kind='stable').reset_index(drop=True)
runs['run_idx'] = runs.index
runs['run_key'] = runs.pr_name + '_build' + runs.build_id.astype(str)
print(f'{STAGE}: {len(runs)} runs, {(runs.num_fail_class > 0).sum()} with failures')

# ---- 2. test results: merge ctypes./cython. variants into one test file
ids, frames, dropped = {}, [], 0
for r in runs.itertuples():
    name = f'{ROOT}processed_test_result/tvm/{r.run_key}/stage_{r.stage_id}/test_class.csv.zip'
    x = pd.read_csv(ZipFile(io.BytesIO(z.read(name))).open('test_class.csv'),
                    usecols=['testclass', 'duration', 'outcome'])
    parts = x.testclass.str.split('.', n=1, expand=True)
    if parts.shape[1] < 2:
        parts[1] = None
    ok = parts[0].isin(['ctypes', 'cython']) & parts[1].notna()
    dropped += int((~ok).sum())
    x = pd.DataFrame({'cid': [ids.setdefault(c, len(ids)) for c in parts.loc[ok, 1]],
                      'outcome': x.outcome[ok].to_numpy(),
                      'duration': x.duration[ok].to_numpy()})
    x = x.groupby('cid', as_index=False).agg(outcome=('outcome', 'max'), duration=('duration', 'sum'))
    x['run_idx'] = r.run_idx
    frames.append(x)
res = pd.concat(frames, ignore_index=True)
names = list(ids)
print(f'rows: {len(res)} | test files: {len(names)} | failures: {int(res.outcome.sum())} '
      f'| rate: {res.outcome.mean():.5f} | dropped odd rows: {dropped}')

# ---- 3. changed files per run
by_run = {}
for n in z.namelist():
    if '/shadata/tvm/compare_commits/' in n and n.endswith('.json'):
        by_run.setdefault(n.split('/')[4], []).append(n)

GENERIC = {'py', 'cc', 'h', 'c', 'cpp', 'hpp', 'cu', 'rs', 'ts', 'sh', 'md', 'txt', 'json', 'yml', 'yaml',
           'cmake', 'tvm', 'python', 'src', 'include', 'tests', 'test', 'unittest', 'init', 'cython',
           'ctypes', 'lib', 'utils', 'util', 'common', 'base'}
CPP = ('.cc', '.h', '.c', '.cpp', '.hpp', '.cu', '.cuh')

def toks(s):
    return {w for w in re.split(r'[^a-z0-9]+', s.lower()) if w and w not in GENERIC and not w.isdigit()}

changes, rl = {}, []
for rk in set(runs.run_key):
    ct, cm = set(), set()
    n_files = adds = dels = n_py = n_cpp = n_tf = 0
    for f in by_run.get(rk, []):
        for x in json.loads(z.read(f)).get('files', []):
            p = x['filename']
            n_files += 1; adds += x['additions']; dels += x['deletions']
            ct |= toks(p)
            n_py += p.endswith('.py'); n_cpp += p.endswith(CPP); n_tf += p.startswith('tests/')
            if p.endswith('.py'):
                cm.add(p[:-3].replace('/', '.'))
    changes[rk] = (ct, cm)
    rl.append(dict(run_key=rk, n_files=n_files, additions=adds, deletions=dels,
                   n_py=n_py, n_cpp=n_cpp, n_test_files=n_tf))
rl = runs[['run_idx', 'run_key', 'pr_name']].merge(pd.DataFrame(rl), on='run_key', how='left')

# ---- 4. code-link features (word overlap, test file itself changed)
tok_c = [toks(u) for u in names]
res = res.sort_values(['run_idx', 'cid']).reset_index(drop=True)
rk_of = runs.run_key.to_numpy()
r_arr, c_arr = res.run_idx.to_numpy(), res.cid.to_numpy()
overlap = np.zeros(len(res), np.float32)
overlap_cnt = np.zeros(len(res), np.float32)
test_changed = np.zeros(len(res), np.int8)
starts = np.flatnonzero(np.r_[True, r_arr[1:] != r_arr[:-1]])
ends = np.r_[starts[1:], len(res)]
for s, e in zip(starts, ends):
    ct, cm = changes[rk_of[r_arr[s]]]
    seg = c_arr[s:e]
    inter = [len(tok_c[c] & ct) for c in seg]
    overlap_cnt[s:e] = inter
    overlap[s:e] = [i / max(len(tok_c[c]), 1) for i, c in zip(inter, seg)]
    test_changed[s:e] = [names[c] in cm for c in seg]
res['overlap'], res['overlap_cnt'], res['test_changed'] = overlap, overlap_cnt, test_changed
res = res.merge(rl.drop(columns='run_key'), on='run_idx', how='left')

# ---- 5. history features: same-PR previous build, then per-test history
res = res.sort_values(['pr_name', 'cid', 'run_idx']).reset_index(drop=True)
prev_pr = res.groupby(['pr_name', 'cid']).outcome.shift(1)
res['pr_prev_seen'] = prev_pr.notna().astype(int)
res['pr_prev_fail'] = prev_pr.fillna(0)

res = res.sort_values(['cid', 'run_idx']).reset_index(drop=True)
g = res.groupby('cid', sort=False)
idx = g.cumcount()
res['prior_runs'] = idx
res['prior_fails'] = g.outcome.cumsum() - res.outcome
res['fail_rate'] = (res.prior_fails / idx.where(idx > 0)).fillna(0)
for w in (5, 20):
    res[f'f{w}'] = res.prior_fails - res.prior_fails.groupby(res.cid).shift(w).fillna(0)
res['last_outcome'] = g.outcome.shift(1).fillna(0)
last_fail = idx.where(res.outcome == 1).groupby(res.cid).shift(1).groupby(res.cid).ffill()
res['runs_since_fail'] = (idx - last_fail).fillna(999)
res['last_duration'] = g.duration.shift(1).fillna(0)
res['mean_duration'] = ((g.duration.cumsum() - res.duration) / idx.where(idx > 0)).fillna(0)
res['history_combo'] = res.f5 * 10 + res.f20 + res.fail_rate
res['hist_plus_overlap'] = res.history_combo + 0.5 * res.overlap
res['random'] = 0.0

HIST = ['prior_runs', 'prior_fails', 'fail_rate', 'f5', 'f20', 'runs_since_fail', 'last_outcome',
        'last_duration', 'mean_duration', 'pr_prev_fail', 'pr_prev_seen']
CODE = ['overlap', 'overlap_cnt', 'test_changed', 'n_files', 'additions', 'deletions',
        'n_py', 'n_cpp', 'n_test_files']

# ---- 6. time split (drop from train any PR that also appears in test)
cut = runs.run_idx.iloc[int(len(runs) * 0.7)]
test = res[res.run_idx >= cut].reset_index(drop=True)
train = res[(res.run_idx < cut) & (~res.pr_name.isin(set(test.pr_name)))]
for nm, d in (('train', train), ('test', test)):
    print(f'{nm}: {d.run_idx.nunique()} runs, {len(d)} rows, {int(d.outcome.sum())} failures '
          f'in {d[d.outcome == 1].run_idx.nunique()} runs')

w = np.where(train.outcome == 1, 20, 1)
for name, feats in (('model_history', HIST), ('model_code', CODE), ('model_all', HIST + CODE)):
    m = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=200, random_state=0)
    m.fit(train[feats], train.outcome, sample_weight=w)
    test[name] = m.predict_proba(test[feats])[:, 1]
    print('trained', name)

# ---- 7. ranking quality within each failing run
def evaluate(d, col, seed=0):
    rng = np.random.default_rng(seed)
    d = d.sort_values('run_idx', kind='stable')
    sc, y, r = d[col].to_numpy(float), d.outcome.to_numpy(), d.run_idx.to_numpy()
    tie = rng.random(len(d))
    st = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    en = np.r_[st[1:], len(d)]
    out = []
    for s, e in zip(st, en):
        yy = y[s:e]
        m = int(yy.sum())
        if m == 0:
            continue
        order = np.lexsort((tie[s:e], -sc[s:e]))
        pos = np.flatnonzero(yy[order] == 1) + 1
        n = e - s
        row = {'apfd': 1 - pos.sum() / (n * m) + 1 / (2 * n)}
        for p in (5, 10, 20):
            row[f'recall@{p}%'] = (pos <= np.ceil(n * p / 100)).mean()
        out.append(row)
    o = pd.DataFrame(out).mean()
    o['runs'] = len(out)
    return o

COLS = ['random', 'overlap', 'history_combo', 'hist_plus_overlap', 'model_history', 'model_code', 'model_all']
for title, mask in (('ALL test rows', test.outcome >= 0),
                    ('tests that have never failed before', test.prior_fails == 0)):
    d = test[mask]
    print(f'\n=== {title}: {len(d)} rows, {int(d.outcome.sum())} failures ===')
    print(pd.DataFrame({c: evaluate(d, c) for c in COLS}).T.round(3))

# ---- 8. selection policy: what do you catch and save if you run only the top k% of tests?
def policy(d, col, fracs=(0.05, 0.10, 0.20, 0.30), seed=0):
    rng = np.random.default_rng(seed)
    d = d.sort_values('run_idx', kind='stable')
    sc, y, du, r = d[col].to_numpy(float), d.outcome.to_numpy(), d.duration.to_numpy(), d.run_idx.to_numpy()
    tie = rng.random(len(d))
    st = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    en = np.r_[st[1:], len(d)]
    caught = {f: 0 for f in fracs}; hit = dict(caught); sel_t = {f: 0.0 for f in fracs}
    tot_f = fail_runs = 0
    for s, e in zip(st, en):
        yy, dd = y[s:e], du[s:e]
        order = np.lexsort((tie[s:e], -sc[s:e]))
        m = int(yy.sum()); tot_f += m; fail_runs += m > 0
        for f in fracs:
            sel = order[:int(np.ceil((e - s) * f))]
            c = int(yy[sel].sum()); caught[f] += c; hit[f] += c > 0; sel_t[f] += dd[sel].sum()
    return [{'method': col, 'run_top%': int(f * 100),
             'failures_caught%': round(100 * caught[f] / tot_f, 1),
             'failing_runs_hit%': round(100 * hit[f] / fail_runs, 1),
             'time_saved%': round(100 * (1 - sel_t[f] / du.sum()), 1)} for f in fracs]

rows = []
for c in ('random', 'history_combo', 'hist_plus_overlap', 'model_all'):
    rows += policy(test, c)
print('\n=== selection policy on the test period ===')
print(pd.DataFrame(rows).to_string(index=False))
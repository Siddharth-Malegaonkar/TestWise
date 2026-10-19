import io, json, re, sys
from zipfile import ZipFile
import numpy as np
import pandas as pd

STAGE = sys.argv[1] if len(sys.argv) > 1 else 'tvm-cpu'
ROOT = 'long_running_test_suites/'
z = ZipFile('data/raw/long_running_test_suites.zip')

df = pd.read_csv(z.open(ROOT + 'dataset.csv'))
t = df[df.project == 'tvm'].copy()
t['stage_id'] = t.stage_id.astype(str)
print(t.groupby('stage_id').build_date.agg(['min', 'max']).to_string(), '\n')

runs = t[t.stage_id == STAGE].sort_values('build_timestamp', kind='stable').reset_index(drop=True)
runs['run_idx'] = runs.index
runs['run_key'] = runs.pr_name + '_build' + runs.build_id.astype(str)
print(f'{STAGE}: {len(runs)} runs, {(runs.num_fail_class > 0).sum()} with failures')

# 1. test results (test names stored as integer ids to save memory)
ids = {}
frames = []
for r in runs.itertuples():
    name = f'{ROOT}processed_test_result/tvm/{r.run_key}/stage_{r.stage_id}/test_class.csv.zip'
    x = pd.read_csv(ZipFile(io.BytesIO(z.read(name))).open('test_class.csv'),
                    usecols=['testclass', 'outcome'])
    x['tid'] = [ids.setdefault(n, len(ids)) for n in x.testclass]
    x['run_idx'] = r.run_idx
    frames.append(x[['run_idx', 'tid', 'outcome']])
res = pd.concat(frames, ignore_index=True)
res['outcome'] = res.outcome.astype('int32')
names = pd.Series(list(ids))
print(f'rows: {len(res)} | distinct tests: {len(names)} | failures: {int(res.outcome.sum())} '
      f'| rate: {res.outcome.mean():.5f}')
print('duplicate (run, test) rows:', int(res.duplicated(['run_idx', 'tid']).sum()))

# 2. strip the cython./ctypes. prefix to get the test file's module path
parts = names.str.split('.', n=1)
prefix = parts.str[0]
canon = parts.str[1].fillna(names)
print('\ndistinct tests per prefix:\n', prefix.value_counts().head(5).to_string())
pcode, _ = pd.factorize(prefix)
ccode, cuniq = pd.factorize(canon)
cuniq = list(cuniq)
res['pcode'] = pcode[res.tid.values]
res['cid'] = ccode[res.tid.values]
both = res.groupby(['run_idx', 'cid']).pcode.nunique()
print('share of (run, test file) pairs present under both prefixes:', round((both > 1).mean(), 3))
print(res.groupby(prefix.values[res.tid.values]).outcome.agg(['size', 'sum']).to_string())

# 3. changed files per run
by_run = {}
for n in z.namelist():
    if '/shadata/tvm/compare_commits/' in n and n.endswith('.json'):
        by_run.setdefault(n.split('/')[4], []).append(n)

GENERIC = {'py', 'cc', 'h', 'c', 'cpp', 'hpp', 'cu', 'rs', 'ts', 'sh', 'md', 'txt', 'json', 'yml', 'yaml',
           'cmake', 'tvm', 'python', 'src', 'include', 'tests', 'test', 'unittest', 'init', 'cython',
           'ctypes', 'lib', 'utils', 'util', 'common', 'base'}

def toks(s):
    return {w for w in re.split(r'[^a-z0-9]+', s.lower()) if w and w not in GENERIC and not w.isdigit()}

changes = {}
for rk in set(runs.run_key):
    ct, cm, n_files = set(), set(), 0
    for f in by_run.get(rk, []):
        for x in json.loads(z.read(f)).get('files', []):
            p = x['filename']
            n_files += 1
            ct |= toks(p)
            if p.endswith('.py'):
                cm.add(p[:-3].replace('/', '.'))
    if n_files:
        changes[rk] = (ct, cm)
print(f'\nruns with code-change data: {len(changes)} of {runs.run_key.nunique()}')

# 4. code-link features: word overlap and "test file itself changed"
tok_c = [toks(u) for u in cuniq]
res = res.sort_values(['run_idx', 'tid']).reset_index(drop=True)
rk_of = runs.run_key.to_numpy()
r_arr = res.run_idx.to_numpy()
c_arr = res.cid.to_numpy()
overlap = np.zeros(len(res), np.float32)
test_changed = np.zeros(len(res), np.int8)
starts = np.flatnonzero(np.r_[True, r_arr[1:] != r_arr[:-1]])
ends = np.r_[starts[1:], len(res)]
for s, e in zip(starts, ends):
    ch = changes.get(rk_of[r_arr[s]])
    if ch is None:
        continue
    ct, cm = ch
    seg = c_arr[s:e]
    overlap[s:e] = [len(tok_c[c] & ct) / max(len(tok_c[c]), 1) for c in seg]
    test_changed[s:e] = [cuniq[c] in cm for c in seg]
res['overlap'] = overlap
res['test_changed'] = test_changed

# 5. history features (only earlier appearances of the same test)
res = res.sort_values(['tid', 'run_idx']).reset_index(drop=True)
g = res.groupby('tid', sort=False)
idx = g.cumcount()
prior_fails = g.outcome.cumsum() - res.outcome
res['prior_fails'] = prior_fails
res['fail_rate'] = (prior_fails / idx.where(idx > 0)).fillna(0)
for w in (5, 20):
    res[f'f{w}'] = prior_fails - prior_fails.groupby(res.tid).shift(w).fillna(0)
res['history_combo'] = res.f5 * 10 + res.f20 + res.fail_rate
res['hist_plus_overlap'] = res.history_combo + 0.5 * res.overlap
res['random'] = 0.0

fl = res[res.outcome == 1]
print(f'\nmean overlap | failing rows: {fl.overlap.mean():.3f} | all rows: {res.overlap.mean():.3f}')
print(f'share with overlap>0 | failing: {(fl.overlap > 0).mean():.3f} | all: {(res.overlap > 0).mean():.3f}')
print(f'test file itself changed | failing: {fl.test_changed.mean():.4f} | all: {res.test_changed.mean():.4f}')
top = fl.groupby('cid').size().sort_values(ascending=False)
print(f'distinct failing tests: {top.size} | top 10 hold {top.head(10).sum() / len(fl):.0%} of failures')

# 6. rank tests within each failing run, using each rule
def evaluate(d, col, seed=0):
    rng = np.random.default_rng(seed)
    d = d.sort_values('run_idx', kind='stable')
    sc = d[col].to_numpy(dtype=float)
    y = d.outcome.to_numpy()
    r = d.run_idx.to_numpy()
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
    res_ = pd.DataFrame(out).mean()
    res_['runs'] = len(out)
    return res_

COLS = ['random', 'overlap', 'test_changed', 'history_combo', 'hist_plus_overlap']
for title, mask in (('ALL rows', res.outcome >= 0),
                    ('tests that have never failed before', res.prior_fails == 0)):
    d = res[mask]
    print(f'\n=== {title}: {len(d)} rows, {int(d.outcome.sum())} failures ===')
    print(pd.DataFrame({c: evaluate(d, c) for c in COLS}).T.round(3))
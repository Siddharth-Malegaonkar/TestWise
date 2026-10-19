import io, json, os, sys
from zipfile import ZipFile
import numpy as np
import pandas as pd

PROJ = sys.argv[1] if len(sys.argv) > 1 else 'activemq'
ZIP_PATH = 'data/raw/long_running_test_suites.zip'
ROOT = 'long_running_test_suites/'
CC_PATH = f'data/processed/code_changes_{PROJ}.csv'
OUT_PATH = f'data/processed/features_{PROJ}.csv.gz'
z = ZipFile(ZIP_PATH)

# 1. runs, oldest first
runs = pd.read_csv(z.open(ROOT + 'dataset.csv'))
runs = runs[runs.project == PROJ].copy()
runs['stage_id'] = runs.stage_id.astype(str)
runs = runs.sort_values('build_timestamp', kind='stable').reset_index(drop=True)
runs['run_idx'] = runs.index
runs['run_key'] = runs.pr_name + '_build' + runs.build_id.astype(str)

# 2. test results, one row per (run, class)
frames = []
for r in runs.itertuples():
    name = f'{ROOT}processed_test_result/{PROJ}/{r.run_key}/stage_{r.stage_id}/test_class.csv.zip'
    inner = ZipFile(io.BytesIO(z.read(name)))
    t = pd.read_csv(inner.open('test_class.csv'), usecols=['testclass', 'duration', 'outcome'])
    t['run_idx'] = r.run_idx
    frames.append(t)
res = pd.concat(frames, ignore_index=True)
n0 = len(res)
res = res.groupby(['run_idx', 'testclass'], as_index=False).agg(
    outcome=('outcome', 'max'), duration=('duration', 'sum'))
print(f'test-result rows: {n0} | duplicate (run, class) rows merged: {n0 - len(res)}')

# 3. code changes per run (cached to csv)
def build_code_changes():
    by_run = {}
    for n in z.namelist():
        if f'/shadata/{PROJ}/compare_commits/' in n and n.endswith('.json'):
            by_run.setdefault(n.split('/')[4], []).append(n)
    rows = []
    for rk, fs in by_run.items():
        names, add, dele = [], 0, 0
        for f in fs:
            for x in json.loads(z.read(f)).get('files', []):
                names.append(x['filename']); add += x['additions']; dele += x['deletions']
        rows.append({'run_key': rk, 'n_pages': len(fs), 'n_files': len(names),
                     'additions': add, 'deletions': dele, 'files': ';'.join(names)})
    return pd.DataFrame(rows)

if os.path.exists(CC_PATH):
    cc = pd.read_csv(CC_PATH)
else:
    cc = build_code_changes()
    cc.to_csv(CC_PATH, index=False)
cc['files'] = cc['files'].fillna('')

def to_fqcn(path):
    if not path.endswith('.java'):
        return None
    for root in ('src/main/java/', 'src/test/java/'):
        if root in path:
            return path.split(root, 1)[1][:-5].replace('/', '.')
    return None

def core(name):
    for s in ('Tests', 'Test'):
        if name.endswith(s) and len(name) > len(s):
            name = name[:-len(s)]
            break
    if name.startswith('Test') and len(name) > 4:
        name = name[4:]
    return name

info = {}
for r in cc.itertuples():
    files = [f for f in r.files.split(';') if f]
    fq = {to_fqcn(f) for f in files} - {None}
    pk = {q.rsplit('.', 1)[0] for q in fq if '.' in q}
    info[r.run_key] = dict(
        n_files=r.n_files, additions=r.additions, deletions=r.deletions,
        n_java=len(fq), n_test_files=sum('src/test/' in f for f in files),
        n_pom=sum(f.endswith('pom.xml') for f in files),
        fqcn=fq, pkgs=pk, pkg_parts=[p.split('.') for p in pk],
        names={core(q.rsplit('.', 1)[-1]) for q in fq})

# 4. class-vs-change overlap features
def common_depth(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

res = res.sort_values(['run_idx', 'testclass']).reset_index(drop=True)
rk = runs.set_index('run_idx')['run_key']
cls_changed = np.zeros(len(res), bool)
same_pkg = np.zeros(len(res), bool)
depth = np.zeros(len(res), int)
name_match = np.zeros(len(res), bool)
for run_idx, grp in res.groupby('run_idx', sort=True):
    ci = info.get(rk[run_idx])
    if ci is None:
        continue
    ix = grp.index.values
    cl = grp.testclass.values
    pk = [c.rsplit('.', 1)[0] if '.' in c else '' for c in cl]
    cls_changed[ix] = [c in ci['fqcn'] for c in cl]
    same_pkg[ix] = [p in ci['pkgs'] for p in pk]
    name_match[ix] = [core(c.rsplit('.', 1)[-1]) in ci['names'] for c in cl]
    cache = {}
    for p in set(pk):
        pp = p.split('.')
        cache[p] = max((common_depth(pp, q) for q in ci['pkg_parts']), default=0)
    depth[ix] = [cache[p] for p in pk]
res['test_class_changed'] = cls_changed.astype(int)
res['same_pkg'] = same_pkg.astype(int)
res['pkg_overlap_depth'] = depth
res['name_match'] = name_match.astype(int)

# 5. history features (only from earlier appearances of the same class)
res = res.sort_values(['testclass', 'run_idx']).reset_index(drop=True)
g = res.groupby('testclass', sort=False)
idx = g.cumcount()
res['prior_runs'] = idx
res['prior_fails'] = g['outcome'].cumsum() - res['outcome']
res['fail_rate'] = (res.prior_fails / res.prior_runs).where(res.prior_runs > 0, 0.0)
prev = g['outcome'].shift(1)
res['last_outcome'] = prev.fillna(0).astype(int)
for w in (5, 20):
    res[f'fails_last{w}'] = (prev.groupby(res['testclass']).rolling(w, min_periods=1).sum()
                             .reset_index(level=0, drop=True).fillna(0))
fail_pos = idx.where(res.outcome == 1)
last_fail = fail_pos.groupby(res['testclass']).shift(1).groupby(res['testclass']).ffill()
res['runs_since_fail'] = (idx - last_fail).fillna(999).astype(int)
res['last_duration'] = g['duration'].shift(1).fillna(0)
res['mean_duration'] = ((g['duration'].cumsum() - res['duration'])
                        / res.prior_runs.where(res.prior_runs > 0)).fillna(0)

# 6. run-level change-size features and metadata
rl = pd.DataFrame.from_dict(
    {k: {f: v[f] for f in ('n_files', 'additions', 'deletions', 'n_java', 'n_test_files', 'n_pom')}
     for k, v in info.items()}, orient='index').rename_axis('run_key').reset_index()
runs = runs.merge(rl, on='run_key', how='left')
RUN_FEATS = ['n_files', 'additions', 'deletions', 'n_java', 'n_test_files', 'n_pom']
res = res.merge(runs[['run_idx', 'pr_name', 'build_id', 'stage_id', 'build_date', 'build_result'] + RUN_FEATS],
                on='run_idx', how='left')
print('rows with no code-change data:', int(res.n_files.isna().sum()))
res[RUN_FEATS] = res[RUN_FEATS].fillna(0)

META = ['run_idx', 'pr_name', 'build_id', 'stage_id', 'build_date', 'build_result', 'testclass', 'outcome']
FEATS = ['last_outcome', 'prior_runs', 'prior_fails', 'fail_rate', 'fails_last5', 'fails_last20',
         'runs_since_fail', 'last_duration', 'mean_duration', 'test_class_changed', 'same_pkg',
         'pkg_overlap_depth', 'name_match'] + RUN_FEATS
out = res[META + FEATS].sort_values(['run_idx', 'testclass']).reset_index(drop=True)
out.to_csv(OUT_PATH, index=False)

# 7. summary: does each signal separate failures from passes?
print('\nsaved', OUT_PATH, out.shape)
print('failures:', int(out.outcome.sum()), '| rate:', round(out.outcome.mean(), 5))
print('runs with a failure:', int(out.groupby('run_idx').outcome.max().sum()), 'of', out.run_idx.nunique())
for c in ['last_outcome', 'test_class_changed', 'same_pkg', 'name_match']:
    print('\n', out.groupby(c).outcome.agg(['size', 'sum', 'mean']).round(5))
print('\n', out.groupby(out.fails_last5 > 0).outcome.agg(['size', 'sum', 'mean']).round(5))
print('\n', out.groupby(pd.cut(out.pkg_overlap_depth, [-1, 0, 2, 4, 20]), observed=True)
      .outcome.agg(['size', 'sum', 'mean']).round(5))
print('\n', out.groupby(out.build_date.astype(str).str[:4]).outcome.agg(['size', 'sum']))
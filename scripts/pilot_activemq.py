import io, json
from zipfile import ZipFile
import pandas as pd

z = ZipFile('data/raw/long_running_test_suites.zip')
ROOT = 'long_running_test_suites/'
PROJ = 'activemq'
KEY = ['pr_name', 'build_id', 'stage_id']

df = pd.read_csv(z.open(ROOT + 'dataset.csv'))
df = df[df.project == PROJ].copy()
df['stage_id'] = df.stage_id.astype(str)
df = df.sort_values('build_timestamp').reset_index(drop=True)
df['run_idx'] = df.index
df['run_key'] = df.pr_name + '_build' + df.build_id.astype(str)

# ---- load every test result for the project
frames = []
for r in df.itertuples():
    name = f"{ROOT}processed_test_result/{PROJ}/{r.run_key}/stage_{r.stage_id}/test_class.csv.zip"
    inner = ZipFile(io.BytesIO(z.read(name)))
    t = pd.read_csv(inner.open('test_class.csv'))
    t['pr_name'], t['build_id'], t['stage_id'] = r.pr_name, r.build_id, r.stage_id
    frames.append(t)
res = pd.concat(frames, ignore_index=True)
print('test-result rows:', len(res), '| failure rate:', round(res.outcome.mean(), 5))

# ---- check 1: what do the dataset.csv counts mean?
res['changed'] = (res.outcome != res.last_outcome).astype(int)
agg = res.groupby(KEY).agg(n=('testclass', 'size'), n_fail=('outcome', 'sum'),
                           n_changed=('changed', 'sum')).reset_index()
chk = df.merge(agg, on=KEY)
tot = chk.num_pass_class + chk.num_fail_class + chk.num_trans_class
print('\n[1] share of runs where:')
print('  n_rows == pass+fail+trans   :', round((chk.n == tot).mean(), 3))
print('  n_rows == pass+fail         :', round((chk.n == chk.num_pass_class + chk.num_fail_class).mean(), 3))
print('  sum(outcome) == num_fail    :', round((chk.n_fail == chk.num_fail_class).mean(), 3))
print('  outcome!=last == num_trans  :', round((chk.n_changed == chk.num_trans_class).mean(), 3))
print(chk[['n', 'n_fail', 'n_changed', 'num_pass_class', 'num_fail_class', 'num_trans_class']].head(8))

# ---- check 2: is last_outcome the class's result in its previous run?
res = res.merge(df[KEY + ['run_idx']], on=KEY)
res = res.sort_values(['testclass', 'run_idx'])
res['prev_outcome'] = res.groupby('testclass')['outcome'].shift(1)
sub = res.dropna(subset=['prev_outcome'])
print('\n[2] last_outcome == outcome in previous run of same class:',
      round((sub.last_outcome == sub.prev_outcome).mean(), 4), 'on', len(sub), 'rows')

# ---- check 3: code changes per run
cc = {}
for n in z.namelist():
    if f'/shadata/{PROJ}/compare_commits/' in n and n.endswith('.json'):
        cc.setdefault(n.split('/')[4], []).append(n)

rows = []
for rk, files in cc.items():
    names, adds, dels = [], 0, 0
    for f in files:
        for x in json.loads(z.read(f)).get('files', []):
            names.append(x['filename']); adds += x['additions']; dels += x['deletions']
    rows.append({'run_key': rk, 'n_pages': len(files), 'n_files': len(names),
                 'additions': adds, 'deletions': dels, 'files': ';'.join(names)})
cdf = pd.DataFrame(rows)
cdf.to_csv(f'data/processed/code_changes_{PROJ}.csv', index=False)
print('\n[3] runs with code-change data:', round(df.run_key.isin(cdf.run_key).mean(), 3))
print(cdf[['n_pages', 'n_files', 'additions', 'deletions']].describe().round(1))
print('runs with 0 changed files:', (cdf.n_files == 0).sum())

# ---- check 4: run sizes
med = chk.n.median()
chk['small'] = chk.n < 0.5 * med
print('\n[4] median classes per run:', med, '| runs below half of that:', int(chk.small.sum()))
print(chk.groupby('small')['n_fail'].agg(['count', 'mean', 'sum']))
print(chk.groupby('build_result')['n'].describe().round(0))
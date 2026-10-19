import io, json, re
from zipfile import ZipFile
import pandas as pd

z = ZipFile('data/raw/long_running_test_suites.zip')
ROOT = 'long_running_test_suites/'
names = z.namelist()

# 1. Uncompressed size per project, to plan extraction (MB)
rows = []
for i in z.infolist():
    if i.is_dir():
        continue
    p = i.filename.split('/')
    if len(p) > 3 and p[1] in ('processed_test_result', 'shadata'):
        rows.append((p[2], p[1], i.file_size))
sz = pd.DataFrame(rows, columns=['project', 'kind', 'bytes'])
print((sz.groupby(['project', 'kind'])['bytes'].sum().unstack() / 1e6).round(0))

# 2. Do result files match dataset.csv runs?
rx = re.compile(r'processed_test_result/([^/]+)/(.+)_build(\d+)/stage_(.+)/test_class\.csv\.zip$')
keys = set()
for n in names:
    m = rx.search(n)
    if m:
        keys.add((m[1], m[2], int(m[3]), m[4]))
df = pd.read_csv(z.open(ROOT + 'dataset.csv'))
dk = set(zip(df.project, df.pr_name, df.build_id, df.stage_id.astype(str)))
print('result files:', len(keys), '| dataset runs:', len(dk), '| both:', len(keys & dk),
      '| files only:', len(keys - dk), '| dataset only:', len(dk - keys))

# 3. What do the outcome codes mean? (first 50 activemq runs)
pt = [n for n in names if '/processed_test_result/activemq/' in n and n.endswith('test_class.csv.zip')]
frames = []
for n in pt[:50]:
    inner = ZipFile(io.BytesIO(z.read(n)))
    frames.append(pd.read_csv(inner.open('test_class.csv')))
a = pd.concat(frames, ignore_index=True)
print(a['outcome'].value_counts(dropna=False))
print(pd.crosstab(a['last_outcome'], a['outcome'], dropna=False))
print(a.isna().sum())

# 4. What is in the code-change data?
print(sorted({n.split('/')[3] for n in names if '/shadata/activemq/' in n}))
cj = [n for n in names if '/shadata/activemq/compare_commits/' in n and n.endswith('.json')]
j = json.loads(z.read(cj[0]))
print(list(j.keys()))
fl = j.get('files', [])
print(len(fl), 'changed files in this compare')
if fl:
    print({k: v for k, v in fl[0].items() if k != 'patch'})

# 5. Do NOT_BUILT / ABORTED runs have any tests?
df['n_classes'] = df.num_pass_class + df.num_fail_class + df.num_trans_class
print(df.groupby('build_result')['n_classes'].describe().round(0))
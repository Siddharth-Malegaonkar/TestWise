import io, json, collections
from zipfile import ZipFile
import pandas as pd

z = ZipFile('data/raw/long_running_test_suites.zip')
ROOT = 'long_running_test_suites/'
df = pd.read_csv(z.open(ROOT + 'dataset.csv'))
t = df[df.project == 'tvm'].copy()
t['stage_id'] = t.stage_id.astype(str)

print(t.groupby('stage_id').agg(
    runs=('build_id', 'size'),
    failing=('num_fail_class', lambda s: (s > 0).sum()),
    median_classes=('num_pass_class', 'median'),
    median_hours=('test_suite_duration_s', lambda s: round(s.median() / 3600, 2))))
print(t.build_result.value_counts(), '\n')

for label, sub in (('failing run', t[t.num_fail_class > 0]), ('passing run', t[t.num_fail_class == 0])):
    r = sub.iloc[len(sub) // 2]
    name = f'{ROOT}processed_test_result/tvm/{r.pr_name}_build{r.build_id}/stage_{r.stage_id}/test_class.csv.zip'
    x = pd.read_csv(ZipFile(io.BytesIO(z.read(name)))
                    .open('test_class.csv'))
    print(f'--- {label}: {r.pr_name} build{r.build_id} stage {r.stage_id} | {len(x)} test rows')
    print(x.sample(8, random_state=0).to_string())
    print(x[x.outcome == 1].head(5).to_string())

# what kinds of files change in TVM pull requests?
cj = [n for n in z.namelist() if '/shadata/tvm/compare_commits/' in n and n.endswith('.json')]
ext = collections.Counter()
examples = []
for n in cj[::max(1, len(cj) // 300)]:
    for f in json.loads(z.read(n)).get('files', []):
        e = f['filename'].rsplit('.', 1)[-1] if '.' in f['filename'] else '(none)'
        ext[e] += 1
        if len(examples) < 12:
            examples.append(f['filename'])
print('\nchanged-file extensions (sample):', ext.most_common(10))
print(examples)
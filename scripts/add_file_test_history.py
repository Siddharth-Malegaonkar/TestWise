"""Build a sparse, prior-only changed-file-to-test failure history sidecar."""
import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'processed' / 'lrts_v1'
SOURCE = DATA_DIR / 'lrts_test_modeling_v1.csv.gz'
CHANGES = DATA_DIR / 'lrts_code_changes_v1.csv.gz'
OUTPUT = DATA_DIR / 'lrts_file_test_links_v1.csv.gz'
RUN_KEY = ['project', 'stage_id', 'run_idx']
OUT_FIELDS = RUN_KEY + ['test_id', 'file_test_link_failure_count', 'file_test_link_max_failures']


def canonical_file(value):
    path = str(value).split('::', 1)[0].replace('\\', '/').strip().lower()
    return re.sub(r'\.(py|java|scala|kt|js|ts|cc|cpp|h|hpp)$', '', path).strip('/')


def load_changed_files():
    changes = pd.read_csv(CHANGES, usecols=['project', 'build_key', 'filename'],
                          dtype={'project': 'string', 'build_key': 'string', 'filename': 'string'})
    changes['path_key'] = changes.filename.map(canonical_file)
    changes = changes[changes.path_key.ne('')].drop_duplicates(['project', 'build_key', 'path_key'])
    return {(str(project), str(build)): tuple(group.path_key.astype(str))
            for (project, build), group in changes.groupby(['project', 'build_key'], sort=False)}


def process_run(run, changed_files, failures_by_file):
    project = str(run.project.iloc[0])
    stage = str(run.stage_id.iloc[0])
    run_idx = int(run.run_idx.iloc[0])
    files = changed_files.get((project, str(run.build_key.iloc[0])), ())
    candidates = set(run.test_id.to_numpy(copy=False))
    link_counts = {}
    for path in files:
        prior = failures_by_file.get((project, stage, path), {})
        for test_id in candidates.intersection(prior):
            count = prior[test_id]
            aggregate = link_counts.get(test_id)
            if aggregate is None:
                link_counts[test_id] = [count, count]
            else:
                aggregate[0] += count
                aggregate[1] = max(aggregate[1], count)

    result = [
        {'project': project, 'stage_id': stage, 'run_idx': run_idx, 'test_id': test_id,
         'file_test_link_failure_count': values[0], 'file_test_link_max_failures': values[1]}
        for test_id, values in link_counts.items()
    ]
    failed_ids = run.loc[run.outcome.eq(1), 'test_id'].unique()
    for test_id in failed_ids:
        for path in files:
            failures_by_file[(project, stage, path)][test_id] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chunksize', type=int, default=100_000)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error('--chunksize must be positive')
    if not SOURCE.is_file() or not CHANGES.is_file():
        raise FileNotFoundError('The finalized modeling and changed-file tables are required')
    if OUTPUT.exists() and not args.overwrite:
        raise FileExistsError(f'{OUTPUT} already exists; pass --overwrite to replace it')
    tmp = OUTPUT.with_suffix(OUTPUT.suffix + '.tmp')
    changed_files = load_changed_files()
    failures_by_file = defaultdict(lambda: defaultdict(int))
    pending = None
    header_written = False
    dtypes = {'project': 'string', 'stage_id': 'string', 'test_id': 'string', 'build_key': 'string'}
    usecols = ['project', 'stage_id', 'run_idx', 'build_key', 'test_id', 'outcome']
    try:
        with gzip.open(tmp, 'wt', newline='', encoding='utf-8', compresslevel=1) as sink:
            for chunk_no, chunk in enumerate(pd.read_csv(SOURCE, usecols=usecols, chunksize=args.chunksize,
                                                         dtype=dtypes), start=1):
                if pending is not None:
                    chunk = pd.concat([pending, chunk], ignore_index=True)
                groups = list(chunk.groupby(RUN_KEY, sort=False, observed=True))
                records = []
                for _, group in groups[:-1]:
                    records.extend(process_run(group, changed_files, failures_by_file))
                if records:
                    pd.DataFrame.from_records(records, columns=OUT_FIELDS).to_csv(
                        sink, index=False, header=not header_written)
                    header_written = True
                pending = groups[-1][1].copy() if groups else None
                if chunk_no % 25 == 0:
                    print(f'Processed {chunk_no * args.chunksize:,} modeling rows; '
                          f'{len(failures_by_file):,} project/stage/file histories')
            if pending is not None:
                records = process_run(pending, changed_files, failures_by_file)
                if records:
                    pd.DataFrame.from_records(records, columns=OUT_FIELDS).to_csv(
                        sink, index=False, header=not header_written)
                    header_written = True
            if not header_written:
                pd.DataFrame(columns=OUT_FIELDS).to_csv(sink, index=False)
        tmp.replace(OUTPUT)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    metadata = {
        'source': str(SOURCE.relative_to(ROOT)),
        'output': str(OUTPUT.relative_to(ROOT)),
        'format': 'sparse; only test/run rows with a nonzero prior file/test failure link are stored',
        'feature_columns': ['file_test_link_failure_count', 'file_test_link_max_failures'],
        'definition': 'For each test and each file changed in the current run, count prior failures of that test in runs that changed that file; emit the sum and maximum across current changed files.',
        'update_order': 'Features are computed before the current run outcomes update prior file/test counts.',
        'scope': 'The full source table is retained; the evaluator excludes TVM before split planning and fitting.',
    }
    OUTPUT.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Saved sparse sidecar {OUTPUT}')


if __name__ == '__main__':
    main()

import argparse
import csv
import gzip
import io
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / 'data' / 'raw' / 'long_running_test_suites.zip'
DATASET_MEMBER = 'long_running_test_suites/dataset.csv'
TEST_FIELDS = [
    'project', 'stage_id', 'run_idx', 'run_key', 'build_key', 'pr_name', 'build_id',
    'build_timestamp', 'test_id', 'outcome', 'duration', 'source_row_count',
    'change_data_available', 'changed_file_count', 'additions', 'deletions',
    'changed_test_file_count', 'path_token_overlap_count', 'path_token_overlap_ratio',
    'test_file_changed', 'path_prefix_overlap_depth', 'prior_runs', 'prior_fails',
    'fail_rate', 'fails_last5', 'fails_last20', 'last_outcome', 'runs_since_fail',
    'last_duration', 'mean_duration', 'build_result',
]
RUN_FIELDS = [
    'project', 'stage_id', 'run_idx', 'run_key', 'build_key', 'pr_name', 'build_id',
    'build_timestamp', 'build_date', 'build_result', 'metadata_test_count',
    'raw_test_rows', 'normalized_test_count', 'normalized_failure_count',
    'test_rows_match_metadata', 'compare_data_available', 'compare_pages',
    'changed_file_count', 'additions', 'deletions', 'changed_test_file_count',
]
CHANGE_FIELDS = [
    'project', 'build_key', 'pr_name', 'build_id', 'filename', 'status', 'additions', 'deletions',
]
MODEL_FEATURES = [
    'change_data_available', 'changed_file_count', 'additions', 'deletions',
    'changed_test_file_count', 'path_token_overlap_count', 'path_token_overlap_ratio',
    'test_file_changed', 'path_prefix_overlap_depth', 'prior_runs', 'prior_fails',
    'fail_rate', 'fails_last5', 'fails_last20', 'last_outcome', 'runs_since_fail',
    'last_duration', 'mean_duration',
]
POST_RUN_FIELDS = [
    'duration', 'build_result', 'build_duration', 'test_suite_duration_s',
    'num_pass_class', 'num_fail_class', 'num_trans_class',
]
STOP_TOKENS = {
    'test', 'tests', 'testing', 'testcase', 'testcases', 'testclass', 'unittest',
    'spec', 'specs', 'src', 'source', 'main', 'java', 'python', 'py', 'scala',
    'kt', 'kotlin', 'js', 'javascript', 'ts', 'typescript', 'cpp', 'cc', 'h',
    'hpp', 'org', 'com', 'net', 'apache', 'lib', 'libs', 'package', 'packages',
}


def text_id(value):
    if pd.isna(value):
        return ''
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def canonical_test_id(project, raw_id):
    test_id = str(raw_id).strip()
    if project == 'tvm':
        for prefix in ('ctypes.', 'cython.'):
            if test_id.startswith(prefix):
                return test_id[len(prefix):]
    return test_id


def tokens(value):
    separated = re.sub(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])', ' ', str(value))
    return {word for word in re.findall(r'[a-z0-9]+', separated.lower())
            if word not in STOP_TOKENS and not word.isdigit()}


def normalized_path(value):
    path = str(value).split('::', 1)[0].replace('\\', '/').strip().lower()
    return re.sub(r'\.(py|java|scala|kt|js|ts|cc|cpp|h|hpp)$', '', path).strip('/')


def path_parts(value, test_id=False):
    value = str(value).split('::', 1)[0].replace('\\', '/')
    if test_id and '/' not in value and '.' in value:
        value = value.replace('.', '/')
    parts = [part.lower() for part in value.split('/') if part]
    if parts:
        parts[-1] = re.sub(r'\.(py|java|scala|kt|js|ts|cc|cpp|h|hpp)$', '', parts[-1])
    return parts


def common_prefix_depth(left, right):
    depth = 0
    for left_part, right_part in zip(left, right):
        if left_part != right_part:
            break
        depth += 1
    return depth


def build_changes_index(archive):
    members_by_build = defaultdict(list)
    for member in archive.namelist():
        parts = member.split('/')
        if (len(parts) > 5 and parts[1] == 'shadata' and parts[3] == 'compare_commits'
                and member.endswith('.json')):
            members_by_build[(parts[2], parts[4])].append(member)
    return members_by_build


def read_build_changes(archive, members):
    by_filename = {}
    for member in members:
        payload = json.loads(archive.read(member))
        for item in payload.get('files', []):
            filename = str(item.get('filename', '')).replace('\\', '/')
            if filename:
                by_filename[filename] = {
                    'filename': filename,
                    'status': str(item.get('status', 'unknown')),
                    'additions': int(item.get('additions') or 0),
                    'deletions': int(item.get('deletions') or 0),
                }
    return [by_filename[name] for name in sorted(by_filename)]


def test_file_match(test_id, changed_paths):
    stem = normalized_path(test_id)
    dotted_stem = normalized_path(test_id.replace('.', '/'))
    candidates = {stem, dotted_stem}
    for changed_path in changed_paths:
        changed_stem = normalized_path(changed_path)
        if any(changed_stem.endswith(candidate) for candidate in candidates if candidate):
            return 1
    return 0


def flush(writer, buffer):
    if buffer:
        writer.writerows(buffer)
        buffer.clear()


def main():
    parser = argparse.ArgumentParser(description='Build a language-agnostic LRTS test-selection corpus.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'processed' / 'lrts_v1')
    parser.add_argument('--limit-runs', type=int, help='Only process the first N run-stage rows for a smoke test.')
    parser.add_argument('--overwrite', action='store_true', help='Replace outputs in the selected output directory.')
    args = parser.parse_args()
    if args.limit_runs is not None and args.limit_runs < 1:
        parser.error('--limit-runs must be positive')
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f'Source archive not found: {ZIP_PATH}')

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_sample' if args.limit_runs is not None else ''
    test_path = output_dir / f'lrts_test_modeling_v1{suffix}.csv.gz'
    run_path = output_dir / f'lrts_run_audit_v1{suffix}.csv.gz'
    change_path = output_dir / f'lrts_code_changes_v1{suffix}.csv.gz'
    manifest_path = output_dir / f'lrts_manifest_v1{suffix}.json'
    output_paths = [test_path, run_path, change_path, manifest_path]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError('Refusing to overwrite existing files; use --overwrite: '
                              + ', '.join(map(str, existing)))

    suffix_tmp = '.tmp'
    temp_paths = [path.with_suffix(path.suffix + suffix_tmp) for path in output_paths]
    stats = {
        'dataset_version': 'lrts_v1',
        'is_complete_dataset': args.limit_runs is None,
        'source_archive': str(ZIP_PATH.relative_to(ROOT)),
        'model_features': MODEL_FEATURES,
        'post_run_fields_forbidden_as_model_features': POST_RUN_FIELDS,
        'history_key': ['project', 'stage_id', 'test_id'],
        'time_order': 'build_timestamp, stable within project and stage',
        'normalization': {
            'unit': 'one row per observed (project, stage_id, run_idx, test_id)',
            'duplicate_test_aggregation': {'outcome': 'max', 'duration': 'sum', 'source_row_count': 'count'},
            'TVM_backend_prefixes_removed': ['ctypes.', 'cython.'],
            'TVM_pseudo_test_dropped': 'pytest',
            'stage_names_preserved': True,
            'missing_code_changes': 'retained and flagged; numeric change features set to zero',
            'duration': 'evaluation-only; never a model feature',
        },
        'source_run_stage_rows': 0,
        'test_rows': 0,
        'failure_rows': 0,
        'row_failure_rate': 0.0,
        'runs_with_result_files': 0,
        'raw_test_counts_matching_metadata': 0,
        'runs_with_compare_data': 0,
        'test_rows_excluded_missing_id': 0,
        'test_rows_excluded_missing_outcome': 0,
        'TVM_pseudo_test_rows_excluded': 0,
        'rows_with_invalid_duration': 0,
        'project_summary': {},
        'outputs': {},
    }

    with ZipFile(ZIP_PATH) as archive:
        runs = pd.read_csv(archive.open(DATASET_MEMBER))
        run_identity = ['project', 'pr_name', 'build_id', 'stage_id']
        if runs.duplicated(run_identity).any():
            raise ValueError('Source dataset has duplicate (project, PR, build, stage) keys')
        runs['stage_id'] = runs['stage_id'].map(text_id)
        runs['build_id'] = runs['build_id'].map(text_id)
        runs['pr_name'] = runs['pr_name'].map(text_id)
        runs = runs.sort_values(['project', 'stage_id', 'build_timestamp'], kind='stable').reset_index(drop=True)
        runs['run_idx'] = runs.groupby(['project', 'stage_id'], sort=False).cumcount()
        runs['build_key'] = runs['project'].astype(str) + '|' + runs['pr_name'] + '|' + runs['build_id']
        runs['run_key'] = runs['build_key'] + '|' + runs['stage_id']
        if args.limit_runs is not None:
            runs = runs.head(args.limit_runs).copy()
        stats['source_run_stage_rows'] = len(runs)

        changes_index = build_changes_index(archive)
        changes_cache = {}
        emitted_build_changes = set()
        history = {}
        project_summary = {}
        test_buffer, run_buffer, change_buffer = [], [], []
        try:
            with gzip.open(temp_paths[0], 'wt', newline='', encoding='utf-8', compresslevel=1) as test_file, \
                    gzip.open(temp_paths[1], 'wt', newline='', encoding='utf-8', compresslevel=1) as run_file, \
                    gzip.open(temp_paths[2], 'wt', newline='', encoding='utf-8', compresslevel=1) as change_file:
                test_writer = csv.DictWriter(test_file, fieldnames=TEST_FIELDS, extrasaction='ignore')
                run_writer = csv.DictWriter(run_file, fieldnames=RUN_FIELDS, extrasaction='ignore')
                change_writer = csv.DictWriter(change_file, fieldnames=CHANGE_FIELDS, extrasaction='ignore')
                test_writer.writeheader()
                run_writer.writeheader()
                change_writer.writeheader()

                for position, run in enumerate(runs.itertuples(index=False), start=1):
                    project = str(run.project)
                    build_member_key = (project, f'{run.pr_name}_build{run.build_id}')
                    result_member = (f'long_running_test_suites/processed_test_result/{project}/'
                                     f'{run.pr_name}_build{run.build_id}/stage_{run.stage_id}/test_class.csv.zip')
                    try:
                        with ZipFile(io.BytesIO(archive.read(result_member))) as result_zip:
                            csv_member = next(name for name in result_zip.namelist()
                                              if name.rsplit('/', 1)[-1] == 'test_class.csv')
                            raw = pd.read_csv(result_zip.open(csv_member),
                                              usecols=['testclass', 'duration', 'outcome'])
                    except (KeyError, BadZipFile, StopIteration) as error:
                        raise ValueError(f'Missing or invalid test-result file: {result_member}') from error

                    stats['runs_with_result_files'] += 1
                    pstats = project_summary.setdefault(project, {
                        'run_stage_rows': 0, 'raw_test_rows': 0, 'test_rows': 0,
                        'failures': 0, 'build_keys': set(), 'compare_build_keys': set(),
                    })
                    pstats['run_stage_rows'] += 1
                    pstats['raw_test_rows'] += len(raw)
                    pstats['build_keys'].add(build_member_key[1])
                    raw_rows = len(raw)
                    raw['testclass'] = raw['testclass'].astype('string')
                    raw['outcome'] = pd.to_numeric(raw['outcome'], errors='coerce')
                    raw['duration'] = pd.to_numeric(raw['duration'], errors='coerce')
                    label_values = set(raw['outcome'].dropna().unique())
                    if not label_values.issubset({0, 1}):
                        raise ValueError(f'Expected binary outcome values in {result_member}, got {label_values}')

                    missing_id = raw['testclass'].isna() | raw['testclass'].str.strip().eq('')
                    missing_label = raw['outcome'].isna()
                    stats['test_rows_excluded_missing_id'] += int(missing_id.sum())
                    stats['test_rows_excluded_missing_outcome'] += int((~missing_id & missing_label).sum())
                    usable = raw.loc[~missing_id & ~missing_label].copy()
                    usable['test_id'] = usable['testclass'].map(lambda value: canonical_test_id(project, value))
                    if project == 'tvm':
                        pseudo_test = usable['test_id'].str.lower().eq('pytest')
                        stats['TVM_pseudo_test_rows_excluded'] += int(pseudo_test.sum())
                        usable = usable.loc[~pseudo_test].copy()

                    invalid_duration = (usable['duration'].isna() | ~np.isfinite(usable['duration'])
                                        | usable['duration'].lt(0))
                    stats['rows_with_invalid_duration'] += int(invalid_duration.sum())
                    usable.loc[invalid_duration, 'duration'] = np.nan
                    tests = (usable.groupby('test_id', as_index=False, sort=True)
                             .agg(outcome=('outcome', 'max'),
                                  duration=('duration', lambda values: values.sum(min_count=1)),
                                  source_row_count=('testclass', 'size')))

                    compare_members = changes_index.get(build_member_key, [])
                    if build_member_key not in changes_cache:
                        changes_cache[build_member_key] = (read_build_changes(archive, compare_members)
                                                            if compare_members else None)
                    build_changes = changes_cache[build_member_key]
                    change_available = build_changes is not None
                    changed_paths = [item['filename'] for item in (build_changes or [])]
                    if change_available:
                        pstats['compare_build_keys'].add(build_member_key[1])
                        if build_member_key not in emitted_build_changes:
                            for item in build_changes:
                                change_buffer.append({
                                    'project': project, 'build_key': run.build_key,
                                    'pr_name': run.pr_name, 'build_id': run.build_id, **item,
                                })
                            emitted_build_changes.add(build_member_key)

                    change_tokens = set().union(*(tokens(path) for path in changed_paths)) if changed_paths else set()
                    change_parents = [path_parts(path)[:-1] for path in changed_paths]
                    additions = sum(item['additions'] for item in (build_changes or []))
                    deletions = sum(item['deletions'] for item in (build_changes or []))
                    changed_test_count = sum(
                        1 for path in changed_paths
                        if any(part in {'test', 'tests', 'testing', 'spec', 'specs'}
                               for part in path.lower().replace('\\', '/').split('/'))
                    )

                    for test in tests.itertuples(index=False):
                        test_id = str(test.test_id)
                        test_tokens = tokens(test_id)
                        overlap_count = len(test_tokens & change_tokens)
                        test_parent = path_parts(test_id, test_id=True)[:-1]
                        prefix_depth = max((common_prefix_depth(test_parent, parent)
                                            for parent in change_parents), default=0)
                        state_key = (project, str(run.stage_id), test_id)
                        state = history.get(state_key)
                        if state is None:
                            state = {
                                'seen': 0, 'failures': 0, 'recent': deque(maxlen=20),
                                'last_fail_seen': None, 'last_duration': None,
                                'duration_sum': 0.0, 'duration_count': 0,
                            }
                            history[state_key] = state
                        duration = float(test.duration) if pd.notna(test.duration) else None
                        test_buffer.append({
                            'project': project, 'stage_id': run.stage_id, 'run_idx': run.run_idx,
                            'run_key': run.run_key, 'build_key': run.build_key,
                            'pr_name': run.pr_name, 'build_id': run.build_id,
                            'build_timestamp': run.build_timestamp, 'test_id': test_id,
                            'outcome': int(test.outcome), 'duration': duration,
                            'source_row_count': int(test.source_row_count),
                            'change_data_available': int(change_available),
                            'changed_file_count': len(changed_paths), 'additions': additions,
                            'deletions': deletions, 'changed_test_file_count': changed_test_count,
                            'path_token_overlap_count': overlap_count,
                            'path_token_overlap_ratio': overlap_count / max(len(test_tokens), 1),
                            'test_file_changed': test_file_match(test_id, changed_paths),
                            'path_prefix_overlap_depth': prefix_depth,
                            'prior_runs': state['seen'], 'prior_fails': state['failures'],
                            'fail_rate': state['failures'] / state['seen'] if state['seen'] else 0.0,
                            'fails_last5': sum(list(state['recent'])[-5:]),
                            'fails_last20': sum(state['recent']),
                            'last_outcome': state['recent'][-1] if state['recent'] else 0,
                            'runs_since_fail': (state['seen'] - state['last_fail_seen']
                                                if state['last_fail_seen'] is not None else 999),
                            'last_duration': state['last_duration'] if state['last_duration'] is not None else 0.0,
                            'mean_duration': (state['duration_sum'] / state['duration_count']
                                              if state['duration_count'] else 0.0),
                            'build_result': run.build_result,
                        })
                        outcome = int(test.outcome)
                        if outcome:
                            state['last_fail_seen'] = state['seen']
                        state['seen'] += 1
                        state['failures'] += outcome
                        state['recent'].append(outcome)
                        if duration is not None and np.isfinite(duration) and duration >= 0:
                            state['last_duration'] = duration
                            state['duration_sum'] += duration
                            state['duration_count'] += 1

                    metadata_count = int(run.num_pass_class) + int(run.num_fail_class)
                    raw_failure_count = int(raw['outcome'].fillna(0).sum())
                    rows_match = raw_rows == metadata_count
                    run_buffer.append({
                        'project': project, 'stage_id': run.stage_id, 'run_idx': run.run_idx,
                        'run_key': run.run_key, 'build_key': run.build_key, 'pr_name': run.pr_name,
                        'build_id': run.build_id, 'build_timestamp': run.build_timestamp,
                        'build_date': run.build_date, 'build_result': run.build_result,
                        'metadata_test_count': metadata_count, 'raw_test_rows': raw_rows,
                        'normalized_test_count': len(tests),
                        'normalized_failure_count': int(tests['outcome'].sum()),
                        'test_rows_match_metadata': int(rows_match),
                        'compare_data_available': int(change_available),
                        'compare_pages': len(compare_members), 'changed_file_count': len(changed_paths),
                        'additions': additions, 'deletions': deletions,
                        'changed_test_file_count': changed_test_count,
                    })
                    stats['test_rows'] += len(tests)
                    stats['failure_rows'] += int(tests['outcome'].sum())
                    stats['runs_with_compare_data'] += int(change_available)
                    stats['raw_test_counts_matching_metadata'] += int(rows_match)
                    pstats['test_rows'] += len(tests)
                    pstats['failures'] += int(tests['outcome'].sum())

                    if len(test_buffer) >= 50000:
                        flush(test_writer, test_buffer)
                    if len(run_buffer) >= 5000:
                        flush(run_writer, run_buffer)
                    if len(change_buffer) >= 50000:
                        flush(change_writer, change_buffer)
                    if position % 5000 == 0:
                        flush(test_writer, test_buffer)
                        flush(run_writer, run_buffer)
                        flush(change_writer, change_buffer)
                        print(f'Processed {position:,}/{len(runs):,} run-stage records; '
                              f'{stats["test_rows"]:,} normalized test rows')

                flush(test_writer, test_buffer)
                flush(run_writer, run_buffer)
                flush(change_writer, change_buffer)
        except Exception:
            for path in temp_paths:
                if path.exists():
                    path.unlink()
            raise

    for project, pstats in project_summary.items():
        pstats['unique_builds'] = len(pstats.pop('build_keys'))
        pstats['builds_with_compare_data'] = len(pstats.pop('compare_build_keys'))
        pstats['compare_build_coverage'] = (
            pstats['builds_with_compare_data'] / pstats['unique_builds'] if pstats['unique_builds'] else 0.0)
    stats['project_summary'] = project_summary
    stats['row_failure_rate'] = stats['failure_rows'] / stats['test_rows'] if stats['test_rows'] else 0.0
    stats['run_metadata_count_match_rate'] = (
        stats['raw_test_counts_matching_metadata'] / stats['source_run_stage_rows']
        if stats['source_run_stage_rows'] else 0.0)
    stats['outputs'] = {
        'test_modeling_table': str(test_path.relative_to(ROOT)),
        'run_audit_table': str(run_path.relative_to(ROOT)),
        'code_change_table': str(change_path.relative_to(ROOT)),
        'manifest': str(manifest_path.relative_to(ROOT)),
    }
    temp_paths[3].write_text(json.dumps(stats, indent=2), encoding='utf-8')
    for temp_path, final_path in zip(temp_paths, output_paths):
        temp_path.replace(final_path)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
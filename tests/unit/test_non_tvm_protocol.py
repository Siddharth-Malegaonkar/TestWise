import pandas as pd

from scripts.evaluate_model import filter_non_tvm, load_split_plan


def test_tvm_is_removed_before_split_planning():
    audit = pd.DataFrame({
        "project": ["activemq", "TVM", "activemq", "TVM"],
        "stage_id": ["s", "CPU", "s", "CPU"],
        "run_idx": [0, 0, 1, 1],
        "pr_name": ["old", "tvm-old", "new", "tvm-new"],
        "build_timestamp": ["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"],
        "test_rows_match_metadata": [1, 1, 1, 1],
    })
    filtered = filter_non_tvm(audit)
    assert filtered.project.tolist() == ["activemq", "activemq"]
    cutoffs, _, split_plan = load_split_plan(filtered)
    assert set(cutoffs) == {("activemq", "s")}
    assert set(split_plan.project) == {"activemq"}


def test_time_split_purges_prs_seen_in_test():
    audit = pd.DataFrame({
        "project": ["p"] * 4,
        "stage_id": ["s"] * 4,
        "run_idx": [0, 1, 2, 3],
        "pr_name": ["repeat", "older", "repeat", "latest"],
        "build_timestamp": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        "test_rows_match_metadata": [1] * 4,
    })
    _, _, plan = load_split_plan(audit)
    row = plan.iloc[0]
    assert row.test_runs == 2
    assert row.purged_train_run_stages == 1
    assert row.train_runs_after_purge == 1


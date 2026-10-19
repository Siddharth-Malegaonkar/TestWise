# Non-TVM contextual bandit offline replay

This is a chronological offline replay, not online deployment or evidence of real-world improvement. TVM was excluded before replay.

Each action selects the top 10% of candidates within one project/stage run. The supervised model probability is the base score; an epsilon-greedy linear contextual correction (epsilon=0.1) updates after each run using feedback only from selected tests.

Reward for selected test i is `failure_i - 0.1 * min(observed_duration_i / max(prior_mean_duration_i, 1s), 5) / 5`. The test outcome and duration are revealed only after selection. The scaler is fit using the first run's candidate contexts per project/stage and then frozen.

## Pooled held-out results

| Method | Failures caught (%) | Precision (%) | Failing stages hit (%) | Duration savings proxy (%) |
|---|---:|---:|---:|---:|
| random | 9.96 | 0.29 | 33.52 | 89.96 |
| history_combo | 83.82 | 2.43 | 97.18 | 43.30 |
| model | 84.05 | 2.44 | 97.86 | 41.45 |
| bandit | 73.41 | 2.13 | 93.92 | 63.38 |

The replay selected 2,806,243 of 27,975,576 held-out candidates (10.03%). Model training excluded project `tvm`; model artifact version is `testwise-lrts-non-tvm-v1`.

The replay simulates feedback from historical outcomes and durations. Real CI integration, delayed/partial feedback behavior, exploration safeguards, and online monitoring have not been validated.

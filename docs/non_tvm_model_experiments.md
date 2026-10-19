# Non-TVM model experiments

## Scope and protocol

This comparison uses the existing non-TVM chronological 70/30 split within project/stage, with training PRs purged when the PR appears in any held-out stage. TVM was excluded before split planning and from all fit/evaluation rows. The fit sample uses seed 0, all positive training rows, and the same deterministic 0.5% negative sample with inverse-probability weights as the production evaluation.

The experiment compares Random Forest (48 trees, depth 18, leaf size 10) and HistGradientBoosting (100 iterations, 31 maximum leaves, learning rate 0.08, L2 1.0) for history-only, code-only, and combined features. Metrics rank tests within each held-out run-stage. APFD and recall are averaged over failing run-stages. Confidence intervals resample failing run-stages within each project (2,000 percentile bootstrap replicates); they describe run-to-run variability, not model-training variability.

The code/change slice requires available diff metadata and at least one changed file. `quiet` means `fails_last20 == 0`; `never failed` means `prior_fails == 0`. Never-failed implies quiet in this dataset, so their intersection has the same rows as the never-failed slice. These are ranking evaluations within the eligible tests of each run-stage.

## Aggregate comparison

Values below are failing-run-stage-weighted means across the nine projects. They summarize the project-level estimates; bootstrap intervals are in the CSV artifact.

| Slice | Model | APFD | Recall at 10% |
|---|---|---:|---:|
| All held-out failures | RF history | 0.9603 | 0.8812 |
| All held-out failures | RF code | 0.5857 | 0.2056 |
| All held-out failures | RF both | 0.9680 | 0.9034 |
| All held-out failures | HGB history | 0.8990 | 0.8338 |
| All held-out failures | HGB code | 0.6145 | 0.2325 |
| All held-out failures | HGB both | 0.9624 | 0.9190 |
| Quiet, change-bearing | RF both | 0.9303 | 0.7675 |
| Quiet, change-bearing | HGB both | 0.9442 | 0.8436 |
| Never-failed, change-bearing | RF both | 0.8929 | 0.7423 |
| Never-failed, change-bearing | HGB both | 0.8858 | 0.7550 |

The all-failure slice contains 14,893 failing run-stages; the quiet, change-bearing slice contains 8,146, and the never-failed, change-bearing slice contains 777. The broader change-bearing slice contains 14,888, excluding five failing run-stages without usable changed-file data. The combined models outperform their corresponding code-only variants by a wide margin. RF has higher aggregate APFD on the full slice; HGB has higher recall at 10%. In the never-failed slice, RF has slightly higher APFD while HGB catches slightly more failures at 10%.

## Per-project APFD

Each cell is the project mean with its 95% run-stage bootstrap interval. The number of failing held-out run-stages appears in the second column.

| Project | Failing run-stages | RF both | HGB both |
|---|---:|---:|---:|
| activemq | 52 | 0.936 (0.901-0.966) | 0.957 (0.932-0.979) |
| hadoop | 231 | 0.936 (0.922-0.949) | 0.938 (0.925-0.951) |
| hbase | 137 | 0.855 (0.824-0.884) | 0.874 (0.843-0.902) |
| hive | 361 | 0.931 (0.919-0.941) | 0.939 (0.930-0.948) |
| jackrabbit-oak | 239 | 0.955 (0.942-0.966) | 0.943 (0.926-0.958) |
| james | 703 | 0.947 (0.939-0.954) | 0.946 (0.938-0.954) |
| kafka | 13,057 | 0.973 (0.972-0.973) | 0.966 (0.965-0.967) |
| karaf | 59 | 0.929 (0.904-0.952) | 0.922 (0.878-0.955) |
| log4j | 54 | 0.955 (0.924-0.977) | 0.962 (0.930-0.983) |

## Change-awareness feature

The dataset builder now computes `file_test_link_failure_count` and `file_test_link_max_failures`: prior failures for a test among earlier runs that changed each file in the current run, aggregated as sum and maximum over current changed files. Updates occur only after scoring a run, preserving prior-only semantics. The current processed modeling table and the experiment above predate this addition. Rebuild the processed corpus with `scripts/build_dataset.py --overwrite` and rerun the experiment before using this signal in an artifact or reporting its effect.

The change-bearing and history-constrained slices do not establish that a code change caused a failure. They measure ranking on failures with diff metadata, and on tests with no prior failures or no failures in their last 20 appearances. The quiet slice still contains older failure history; only the never-failed slice has no prior test failures. The LRTS labels remain observed pass/fail outcomes.

## Artifacts

- Per-project metrics and 95% intervals: [non_tvm_model_experiments.csv](../artifacts/evaluation/non_tvm_model_experiments.csv)
- Reproducible runner: [run_non_tvm_model_experiments.py](../scripts/run_non_tvm_model_experiments.py)
- Prior-only file/test history implementation: [build_dataset.py](../scripts/build_dataset.py)

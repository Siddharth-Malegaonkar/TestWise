# TestWise model protocol

- **Target:** `outcome`, the observed pass/fail result for one test in one build/stage. Scores are failure-risk scores, not change relevance.
- **Included projects:** activemq, hadoop, hbase, hive, jackrabbit-oak, james, kafka, karaf, log4j.
- **Excluded project:** tvm. The non-TVM evaluator filters the run audit before creating split cutoffs and filters modeling rows against those cutoffs, so no TVM run enters fit or evaluation.
- **Split:** chronological 70/30 within project/stage by `run_idx`; purge a training PR if it occurs in any held-out stage for that project.
- **Training:** deterministic 0.5% sample of negative training rows plus all positive training rows; sampled negatives receive inverse-probability sample weight. Histories in the processed feature table are prior-only. Post-run outcome and duration are not model features.
- **File-to-test history feature:** the dataset builder now supports prior failures of a test on each changed file in the current run (summed count and maximum single-file count). It is updated after scoring each run. The current model artifact predates this feature; rebuild the processed corpus and rerun the evaluation before using or reporting it.
- **Model:** RandomForestClassifier with 48 trees, depth capped at 18, and minimum leaf size 10, selected here to permit reproducible execution in the restricted Windows environment. Held-out comparisons include random, history-combination, path-overlap, risk score, and risk per prior mean-second rankings.
- **Selection metrics:** APFD; recall among failures within each failing run-stage at 5/10/20%; total failures caught; failing run-stages hit; and duration-weighted selected-cost proxy. The latter is not a wall-clock early-detection simulation.
- **Reproducibility:** seed 0, configuration in `config.yaml`, full feature order and estimator settings in the model payload and evaluation manifest.

No metrics are reported in this protocol until the non-TVM evaluation completes successfully. For a new repository, history features require actual prior CI results. Code-change features may be derived from the diff and test paths. If prior outcomes are absent, the predictor requires an explicit product-level fallback policy; it does not fabricate history.

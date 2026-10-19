# TestWise non-TVM evaluation

## Scope and objective

The model scores the probability of an observed per-test `outcome` failure. It does not estimate code-change relevance and it does not execute tests.

Included projects: activemq, hadoop, hbase, hive, jackrabbit-oak, james, kafka, karaf, log4j. The source audit has 10 projects before filtering and 9 after filtering; TVM is excluded before split construction, training, and evaluation.

The non-TVM source population contains 87,792,866 test rows and 220,827 failures (0.252% failure rate).

Training used 39,203,536 eligible non-TVM historical rows; the sampled fit contained 286,242 rows (90,711 failures) with inverse-probability weights for sampled negatives.

## Evaluation protocol

Chronological 70/30 split within each project and stage by run index. Training PRs that also appear in any held-out stage for that project are purged. All nine included projects have complete test-count reconciliation in the source run audit. The final estimator is a single-worker RandomForestClassifier (48 trees, depth capped at 18, minimum leaf size 10); the existing HistGradientBoosting implementation could not fit under this Windows sandbox because scikit-learn thread-pool creation was denied.

Ranking results are aggregated from held-out runs. Precision/F1 and selection percentage are derived from measured failure catches and the exact test counts selected at each per-run budget. Full-suite execution is a reference baseline: 100% of tests selected and 100% of recorded failures observed by definition; it has no test-time saving.

Row-wise average precision/PR-AUC, fixed-threshold classification metrics, and probability calibration were not computed; the executed evaluation focuses on within-run ranking and budgeted failure detection.

## Pooled held-out results

| Method | APFD | Recall @ 5% (mean/run) | Failures caught @ 5% | Precision @ 5% | F1 @ 5% | Failures caught @ 10% | Failures caught @ 20% | Time saved proxy @ 10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 0.50 | 0.05 | 4.92% | 0.28% | 0.54% | 10.11% | 20.09% | 89.95% |
| history_combo | 0.96 | 0.83 | 72.07% | 4.17% | 7.89% | 83.82% | 94.00% | 43.30% |
| path_overlap | 0.60 | 0.14 | 13.04% | 0.76% | 1.43% | 20.34% | 33.79% | 88.31% |
| model | 0.97 | 0.84 | 72.74% | 4.21% | 7.96% | 84.05% | 94.96% | 41.45% |
| model_per_prior_second | 0.82 | 0.34 | 28.75% | 1.67% | 3.15% | 41.87% | 61.52% | 96.97% |

## Interpretation and limitations

The model APFD is higher the strongest listed non-model baseline (history_combo: 0.959; model: 0.968). At 5% selection, the model catches 72.74% of failures versus 72.07% for that baseline. The model's 10% duration-savings proxy (41.45%) is lower than history_combo (43.30%). The APFD gap is small and no uncertainty interval was computed.
The separate 10% chronological bandit replay caught 73.41% of failures and had 63.38% duration savings proxy, versus 84.05% and 41.45% for the supervised model. It trades failure catch for lower selected duration and is not selected as the default policy.
The duration-based measure is a selected-duration savings proxy, not an execution-order simulator. Partial execution can miss failures; the full-suite baseline remains the complete test run.
The fitted model is validated only on these nine LRTS projects. Historical features require real prior CI results. New repositories need a test identifier mapping, current change features, and a policy for absent history; the prediction module rejects missing feature values rather than inventing them.
The contextual-bandit result is an offline replay only. No online learning or real-world improvement is claimed.

## Artifacts

- Model: `artifacts/models/testwise_lrts_non_tvm.joblib`
- Model metadata: `artifacts/models/testwise_lrts_non_tvm.metadata.json`
- Result table: `artifacts/evaluation/lrts_non_tvm_results.csv`
- Split plan: `artifacts/evaluation/non_tvm_time_split_plan.csv`
- Evaluation manifest: `artifacts/evaluation/non_tvm_evaluation_manifest.json`
- Bandit replay: `artifacts/evaluation/non_tvm_bandit_replay.csv` and `docs/bandit_replay.md`
- Prediction API: `ml/prediction.py`

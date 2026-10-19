# TestWise

TestWise ranks candidate tests by estimated probability of a failure, using the LRTS per-test `outcome` label. The model does not predict test relevance to changed code and does not execute tests.

## Model scope

TVM is excluded before split planning, sampling, fitting, and held-out scoring. The included projects are ActiveMQ, Hadoop, HBase, Hive, Jackrabbit Oak, James, Kafka, Karaf, and Log4j. The supported evaluation is chronological per project and stage (70/30), with training PRs purged when the PR also appears in the held-out period.

Train and evaluate using the repository environment:

```powershell
.venv\Scripts\python.exe scripts\evaluate_model.py --overwrite
```

The run writes the deployable model to `artifacts/models/` and the evaluation CSV, split plan, count diagnostics, and manifest to `artifacts/evaluation/`. It reads the existing compressed processed table in place; it does not unpack or copy the raw archive.

Run the additional non-TVM experiments (history/code/both ablations, Random Forest versus HistGradientBoosting, per-project bootstrap intervals, and quiet/never-failed change-bearing slices) with:

```powershell
.venv\Scripts\python.exe scripts\run_non_tvm_model_experiments.py
```

Results and methodology are in `docs/non_tvm_model_experiments.md`. To add the prior-only file-to-test failure counts to the existing processed table without rebuilding the raw corpus, run:

```powershell
.venv\Scripts\python.exe scripts\add_file_test_history.py
```

The model evaluator automatically uses the enriched table when it exists. Rebuild from the raw archive with `scripts\build_dataset.py --overwrite` only when regenerating the entire processed corpus.

After supervised evaluation completes, run the chronological, non-TVM offline bandit replay with:

```powershell
.venv\Scripts\python.exe scripts\replay_bandit_evaluation.py
```

The replay's outcomes are recorded separately in `artifacts/evaluation/non_tvm_bandit_replay.csv` and `docs/bandit_replay.md`.

## Prediction

```python
import pandas as pd
from ml.prediction import TestWisePredictor

predictor = TestWisePredictor()
# candidate_frame must include `test_id` and every configured model feature,
# populated from the current diff and actual historical CI results.
ranked = predictor.predict(candidate_frame, top_k=20)
```

The predictor rejects missing model features. A time budget requires an observed `estimated_duration` column and selects tests in risk-score order until the next test would exceed the budget. Test execution and outcome reporting remain separate integration work.

## Current limits

This model has only been trained and evaluated on the nine listed LRTS projects. New repositories need compatible test identifiers, code-change features, and real historical CI outcomes for history features. No FastAPI, Streamlit, or GitHub Actions end-to-end connection is claimed here.

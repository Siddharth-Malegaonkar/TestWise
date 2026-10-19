"""Load and use a trained TestWise failure-risk model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "models" / "testwise_lrts_non_tvm.joblib"
ID_COLUMNS = ("test_id", "testclass", "test_name")


class TestWisePredictor:
    __test__ = False

    def __init__(self, artifact_path: str | Path = DEFAULT_ARTIFACT):
        payload = joblib.load(artifact_path)
        self.model = payload["model"]
        self.features = tuple(payload["features"])
        self.version = payload["version"]
        self.target = payload["target"]

    def predict(self, candidates: pd.DataFrame, *, top_k: int | None = None,
                budget_seconds: float | None = None) -> pd.DataFrame:
        """Score candidates and mark a deterministic top-K or duration-budget subset.

        For a time budget, candidates must include a positive ``estimated_duration``.
        Tests are added in descending risk score until the next test would exceed budget.
        """
        if top_k is not None and budget_seconds is not None:
            raise ValueError("Choose either top_k or budget_seconds, not both")
        missing = [name for name in self.features if name not in candidates.columns]
        if missing:
            raise ValueError(f"Missing required model features: {missing}")
        if candidates.loc[:, self.features].isna().any().any():
            raise ValueError("Model features contain missing values; provide observed history/features explicitly")
        ids = next((name for name in ID_COLUMNS if name in candidates.columns), None)
        if ids is None:
            raise ValueError(f"Candidates must include one test identifier column: {ID_COLUMNS}")
        x = candidates.loc[:, self.features].to_numpy(dtype=np.float32)
        if candidates.empty:
            return pd.DataFrame(columns=["test_id", "score", "rank", "selected", "model_version"])
        scores = self.model.predict_proba(x)[:, 1]
        result = pd.DataFrame({"test_id": candidates[ids].astype(str).to_numpy(), "score": scores})
        result["_order"] = np.arange(len(result))
        result = result.sort_values(["score", "test_id", "_order"], ascending=[False, True, True], kind="stable")
        result["rank"] = np.arange(1, len(result) + 1)
        result["selected"] = False
        if budget_seconds is not None:
            if budget_seconds < 0:
                raise ValueError("budget_seconds must be non-negative")
            if "estimated_duration" not in candidates.columns:
                raise ValueError("budget_seconds selection requires estimated_duration")
            durations = pd.to_numeric(candidates["estimated_duration"], errors="coerce")
            if durations.isna().any() or (durations < 0).any():
                raise ValueError("estimated_duration must contain non-negative observed estimates")
            duration_by_id_order = durations.to_numpy()
            chosen_duration = duration_by_id_order[result["_order"].to_numpy()]
            selected, spent = [], 0.0
            for position, cost in enumerate(chosen_duration):
                if spent + cost <= budget_seconds:
                    selected.append(position)
                    spent += cost
            if selected:
                result.iloc[selected, result.columns.get_loc("selected")] = True
        elif top_k is not None:
            if top_k < 0:
                raise ValueError("top_k must be non-negative")
            result.loc[result.index[:top_k], "selected"] = True
        result["model_version"] = self.version
        result = result.drop(columns=["_order"]).reset_index(drop=True)
        return result[["test_id", "score", "rank", "selected", "model_version"]]


import joblib
import numpy as np
import pandas as pd
import pytest

from ml.prediction import DEFAULT_ARTIFACT, TestWisePredictor


class FakeModel:
    def predict_proba(self, x):
        score = x[:, 0] / 3
        return np.column_stack([1 - score, score])


@pytest.fixture
def predictor(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump({"model": FakeModel(), "features": ["f"], "version": "test", "target": "outcome"}, path)
    return TestWisePredictor(path)


def test_scores_and_top_k_are_valid_and_deterministic(predictor):
    frame = pd.DataFrame({"test_id": ["b", "a", "c"], "f": [1, 1, 2]})
    result = predictor.predict(frame, top_k=2)
    assert result.columns.tolist() == ["test_id", "score", "rank", "selected", "model_version"]
    assert result["rank"].tolist() == [1, 2, 3]
    assert result["test_id"].tolist() == ["c", "a", "b"]
    assert result["selected"].sum() == 2
    assert result["model_version"].eq("test").all()


def test_empty_and_missing_features(predictor):
    empty = predictor.predict(pd.DataFrame({"test_id": [], "f": []}), top_k=2)
    assert empty.empty
    with pytest.raises(ValueError, match="Missing required"):
        predictor.predict(pd.DataFrame({"test_id": ["x"]}))
    with pytest.raises(ValueError, match="missing values"):
        predictor.predict(pd.DataFrame({"test_id": ["x"], "f": [np.nan]}))


def test_duration_budget(predictor):
    frame = pd.DataFrame({"test_id": ["a", "b", "c"], "f": [0, 3, 2], "estimated_duration": [1, 3, 2]})
    result = predictor.predict(frame, budget_seconds=3)
    assert result.loc[result.selected, "test_id"].tolist() == ["b"]


def test_final_artifact_round_trip_and_non_tvm_scope():
    if not DEFAULT_ARTIFACT.is_file():
        pytest.skip("Final trained artifact is created by scripts/evaluate_model.py")
    payload = joblib.load(DEFAULT_ARTIFACT)
    assert "tvm" not in {project.lower() for project in payload["training_projects"]}
    assert payload["excluded_projects"] == ["tvm"]
    predictor = TestWisePredictor(DEFAULT_ARTIFACT)
    candidate = pd.DataFrame([{**{feature: 0.0 for feature in predictor.features}, "test_id": "synthetic-check"}])
    result = predictor.predict(candidate, top_k=1)
    assert result.loc[0, "test_id"] == "synthetic-check"
    assert result.loc[0, "rank"] == 1
    assert bool(result.loc[0, "selected"])


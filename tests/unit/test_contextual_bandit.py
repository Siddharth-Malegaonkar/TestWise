import numpy as np
import pytest

from ml.bandit.contextual_bandit import CostAwareContextualBandit


def test_bandit_selects_and_updates_only_selected_feedback():
    policy = CostAwareContextualBandit(2, epsilon=0, random_state=0)
    X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
    probs = np.array([0.1, 0.8, 0.4])
    selected = policy.select(X, probs, 1)
    assert selected.tolist() == [1]
    before = policy.weights.copy()
    policy.update(X[selected], probs[selected], [1], [3.0], [3.0])
    assert not np.array_equal(policy.weights, before)


def test_bandit_empty_selection_and_validation():
    policy = CostAwareContextualBandit(1)
    assert policy.select(np.empty((0, 1)), [], 2).size == 0
    with pytest.raises(ValueError, match="non-negative"):
        policy.update([[1]], [0.5], [1], [-1], [1])


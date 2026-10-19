"""Linear epsilon-greedy correction over the supervised failure-risk model."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


class CostAwareContextualBandit:
    """Select a top-k batch and update from observed selected-test rewards.

    Reward is failure detection (1 for failure, 0 for pass) minus a bounded
    execution-cost penalty: ``cost_weight * min(duration / prior_mean, 5) / 5``.
    The model probability remains the base score; a linear online correction
    learns from feedback. The scaler is fitted on the first decision's context
    only and then frozen, so later contexts share one feature coordinate system.
    """

    def __init__(self, n_features: int, *, epsilon: float = 0.1,
                 learning_rate: float = 0.02, l2_penalty: float = 1e-3,
                 cost_weight: float = 0.1, random_state: int = 0):
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        if cost_weight < 0:
            raise ValueError("cost_weight must be non-negative")
        self.n_features = n_features
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.cost_weight = cost_weight
        self.rng = np.random.default_rng(random_state)
        self.scaler = StandardScaler()
        self.weights = np.zeros(n_features, dtype=float)
        self._scaled = False

    def _context(self, X, *, initialize=False):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"Expected a 2D context with {self.n_features} features")
        if not np.isfinite(X).all():
            raise ValueError("Context contains non-finite values")
        if not self._scaled:
            if not initialize:
                raise RuntimeError("Call select before update to initialize context scaling")
            self.scaler.fit(X)
            self._scaled = True
        # Clamp distribution shift against the frozen first-run scaler so one
        # large code-change value cannot destabilize the online correction.
        return np.clip(self.scaler.transform(X), -10.0, 10.0)

    def select(self, X, model_probabilities, k: int) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if len(X) == 0 or k <= 0:
            return np.array([], dtype=int)
        if len(model_probabilities) != len(X):
            raise ValueError("One model probability is required per candidate")
        if not np.isfinite(model_probabilities).all():
            raise ValueError("Model probabilities must be finite")
        k = min(int(k), len(X))
        scaled = self._context(X, initialize=True)
        score = np.asarray(model_probabilities, dtype=float) + scaled @ self.weights
        exploit_count = min(k, max(1, int(np.floor((1 - self.epsilon) * k)))) if self.epsilon < 1 else 0
        exploit = np.lexsort((np.arange(len(X)), -score))[:exploit_count]
        available = np.setdiff1d(np.arange(len(X)), exploit, assume_unique=True)
        explore_count = k - exploit_count
        explore = self.rng.choice(available, size=explore_count, replace=False) if explore_count else np.array([], dtype=int)
        return np.concatenate([exploit, explore]).astype(int)

    def update(self, X, model_probabilities, outcomes, durations, prior_mean_durations):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        p = np.asarray(model_probabilities, dtype=float).reshape(-1)
        y = np.asarray(outcomes, dtype=float).reshape(-1)
        duration = np.asarray(durations, dtype=float).reshape(-1)
        prior = np.asarray(prior_mean_durations, dtype=float).reshape(-1)
        if not (len(X) == len(p) == len(y) == len(duration) == len(prior)):
            raise ValueError("Selected contexts, probabilities, outcomes, and durations must align")
        if not len(X):
            return
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Observed outcomes must be binary")
        if (duration < 0).any() or not np.isfinite(duration).all():
            raise ValueError("Observed durations must be finite and non-negative")
        scaled = self._context(X)
        expected_duration = np.maximum(prior, 1.0)
        relative_cost = np.minimum(duration / expected_duration, 5.0) / 5.0
        reward = y - self.cost_weight * relative_cost
        prediction = p + np.clip(scaled @ self.weights, -2.0, 2.0)
        residual = prediction - reward
        gradient = scaled.T @ residual / len(X) + self.l2_penalty * self.weights
        self.weights -= self.learning_rate * gradient
        self.weights = np.clip(self.weights, -1.0, 1.0)


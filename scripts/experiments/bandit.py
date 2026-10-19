import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class EpsilonGreedyBandit:
    """Linear epsilon-greedy bandit using the model logit plus a learned correction."""

    def __init__(self, n_features, epsilon=0.1, learning_rate=0.01, l2_penalty=1e-3,
                 failure_weight=20.0, random_state=None):
        # The model probability contributes a fixed logit term; the learned correction starts at zero.
        self.n_features = n_features
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.l2_penalty = l2_penalty
        self.failure_weight = failure_weight
        self.rng = np.random.default_rng(random_state)
        self.weights = np.zeros(self.n_features, dtype=float)
        self.scaler = StandardScaler()
        self._fitted = False

    def _logit(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))

    def _prepare_features(self, X):
        # Standardize only the feature rows used by the learned correction.
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features:
            raise ValueError(f'Expected {self.n_features} feature columns, got {X.shape[1]}')

        if not self._fitted:
            self.scaler.fit(X)
            self._fitted = True
        return self.scaler.transform(X)

    def select(self, X, k, model_probabilities):
        """Pick the best tests by a score = model logit + learned correction."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_tests = len(X)
        if k <= 0:
            return np.array([], dtype=int)
        if n_tests == 0:
            return np.array([], dtype=int)
        if k > n_tests:
            k = n_tests

        X_std = self._prepare_features(X)
        base_scores = self._logit(np.clip(model_probabilities, 1e-6, 1 - 1e-6))
        scores = base_scores + X_std @ self.weights

        n_exploit = max(1, int(np.floor((1.0 - self.epsilon) * k)))
        if n_exploit > n_tests:
            n_exploit = n_tests

        exploit_idx = np.argsort(scores)[-n_exploit:]
        chosen = np.array([], dtype=int)
        chosen = np.concatenate([chosen, exploit_idx])

        remaining_slots = k - len(chosen)
        if remaining_slots > 0:
            available = np.setdiff1d(np.arange(n_tests), exploit_idx, assume_unique=True)
            if len(available) == 0:
                random_idx = self.rng.choice(n_tests, size=remaining_slots, replace=True)
            else:
                random_idx = self.rng.choice(available, size=min(remaining_slots, len(available)), replace=False)
            chosen = np.concatenate([chosen, random_idx])

        return chosen.astype(int)

    def update(self, X_selected, rewards, model_probabilities):
        """Online logistic regression update with an L2 penalty and rare-failure weighting."""
        X_selected = np.asarray(X_selected, dtype=float)
        if X_selected.ndim == 1:
            X_selected = X_selected.reshape(1, -1)
        rewards = np.asarray(rewards, dtype=float).reshape(-1)
        if len(X_selected) != len(rewards):
            raise ValueError('X_selected and rewards must have the same number of rows')

        X_std = self._prepare_features(X_selected)
        model_probabilities = np.asarray(model_probabilities, dtype=float).reshape(-1)
        if len(model_probabilities) != len(X_selected):
            raise ValueError('model_probabilities must have one value per selected row')

        # The model contribution stays fixed; only the correction weights are learned.
        logit_base = self._logit(np.clip(model_probabilities, 1e-6, 1 - 1e-6))
        logits = logit_base + X_std @ self.weights
        probs = 1.0 / (1.0 + np.exp(-logits))

        # Failures are rare, so treat them as much more important than passes.
        sample_weights = np.where(rewards == 1, self.failure_weight, 1.0)
        residual = probs - rewards
        grad = (X_std.T @ (sample_weights * residual)) / sample_weights.sum()
        grad += self.l2_penalty * self.weights
        self.weights -= self.learning_rate * grad


if __name__ == '__main__':
    # Simple example: load the saved model, build a score vector, and rank a small batch.
    # The fixed model logit is combined with a learned correction that starts at zero.
    import joblib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    model_path = root / 'models' / 'model.joblib'
    feature_path = root / 'data' / 'processed' / 'features_activemq.csv.gz'

    model = joblib.load(model_path)
    df = pd.read_csv(feature_path)
    X = df[[c for c in model.feature_names_in_ if c in df.columns]].to_numpy()
    probs = model.predict_proba(X)[:, 1]

    bandit = EpsilonGreedyBandit(n_features=X.shape[1], epsilon=0.1, learning_rate=0.01, random_state=0)
    selected = bandit.select(X, k=min(10, len(X)), model_probabilities=probs)
    print('Selected test indices:', selected)

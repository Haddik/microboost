"""Gradient boosting on top of the CART regressor in :mod:`microboost.tree`.

The core loop follows Friedman's stagewise algorithm:

1. Start from a constant model ``F_0 = argmin_c sum L(y_i, c)``.
2. For each stage ``m``:
     a. compute pseudo-residuals ``r = -dL/dF`` at the current model,
     b. fit a regression tree ``h_m`` to those residuals,
     c. update ``F_m = F_{m-1} + learning_rate * h_m``.

An optional row subsample (``subsample < 1.0``) turns this into stochastic
gradient boosting, which usually improves generalisation.
"""

from __future__ import annotations

import numpy as np

from .losses import LogisticLoss, Loss, SquaredError, sigmoid
from .tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    """Least-squares (or custom loss) gradient boosting for regression."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        loss: Loss | None = None,
        random_state: int | None = None,
    ) -> None:
        if not 0.0 < subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.loss = loss or SquaredError()
        self.random_state = random_state

        self.init_: float | None = None
        self.trees_: list[DecisionTreeRegressor] = []
        self.train_loss_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)

        self.init_ = self.loss.init_estimate(y)
        f = np.full(len(y), self.init_, dtype=float)
        self.trees_ = []
        self.train_loss_ = []

        n_samples = len(y)
        sample_size = max(1, int(round(self.subsample * n_samples)))

        for _ in range(self.n_estimators):
            residuals = self.loss.negative_gradient(y, f)

            if self.subsample < 1.0:
                idx = rng.choice(n_samples, size=sample_size, replace=False)
            else:
                idx = np.arange(n_samples)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X[idx], residuals[idx])

            # Update the *whole* training set, even under subsampling, so the
            # monitored loss reflects the deployed model.
            f += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.train_loss_.append(self.loss(y, f))

        return self

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        f = np.full(len(X), self.init_, dtype=float)
        for tree in self.trees_:
            f += self.learning_rate * tree.predict(X)
        return f

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.init_ is None:
            raise RuntimeError("call fit() before predict()")
        return self._raw_predict(X)

    def staged_predict(self, X: np.ndarray):
        """Yield predictions after each boosting stage.

        Handy for plotting test error against the number of trees and for
        picking an early-stopping point.
        """
        if self.init_ is None:
            raise RuntimeError("call fit() before staged_predict()")
        X = np.asarray(X, dtype=float)
        f = np.full(len(X), self.init_, dtype=float)
        for tree in self.trees_:
            f += self.learning_rate * tree.predict(X)
            yield f.copy()


class GradientBoostingClassifier(GradientBoostingRegressor):
    """Binary gradient boosting with a logistic loss.

    The base class does all the work; we only fix the loss to cross-entropy
    and add probability / label outputs on top of the raw log-odds score.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            loss=LogisticLoss(),
            random_state=random_state,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba_positive = sigmoid(self._raw_predict(X))
        return np.column_stack([1.0 - proba_positive, proba_positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (sigmoid(self._raw_predict(X)) >= 0.5).astype(int)

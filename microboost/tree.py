"""Regression decision tree used as the base learner for gradient boosting.

This is a compact CART implementation. Splits are chosen greedily to maximise
the reduction in sum of squared errors (equivalently, weighted variance
reduction). Only numeric features are supported, which is enough for the
gradient boosting experiments in this project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _Node:
    """A single node of the tree.

    Internal nodes use ``feature`` and ``threshold``; leaves use ``value``.
    """

    feature: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    value: float | None = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeRegressor:
    """Greedy CART regressor.

    Parameters
    ----------
    max_depth:
        Maximum depth of the tree. ``None`` grows until the other stopping
        criteria kick in.
    min_samples_split:
        A node with fewer samples than this becomes a leaf.
    min_samples_leaf:
        A split is only accepted if both children keep at least this many
        samples. Acts as a simple regulariser.
    """

    def __init__(
        self,
        max_depth: int | None = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._root: _Node | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        self._root = self._build(X, y, depth=0)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples = len(y)

        # Stopping conditions -> make a leaf holding the mean target.
        if (
            n_samples < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.all(y == y[0])
        ):
            return _Node(value=float(np.mean(y)))

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return _Node(value=float(np.mean(y)))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)
        return _Node(feature=feature, threshold=threshold, left=left, right=right)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[int | None, float | None]:
        """Find the split that minimises the total child SSE.

        We scan every feature and every candidate threshold (midpoints between
        consecutive unique values). Prefix sums let us evaluate each threshold
        in O(1), so a feature costs O(n log n) dominated by the sort.
        """

        n_samples, n_features = X.shape
        parent_sse = _sse(y)
        best_gain = 0.0
        best_feature: int | None = None
        best_threshold: float | None = None

        for feature in range(n_features):
            column = X[:, feature]
            order = np.argsort(column, kind="mergesort")
            x_sorted = column[order]
            y_sorted = y[order]

            # Running sums from the left, used to compute child SSE cheaply.
            cumsum = np.cumsum(y_sorted)
            cumsum_sq = np.cumsum(y_sorted ** 2)
            total_sum = cumsum[-1]
            total_sq = cumsum_sq[-1]

            for i in range(self.min_samples_leaf, n_samples - self.min_samples_leaf + 1):
                # Don't split between identical feature values.
                if x_sorted[i] == x_sorted[i - 1]:
                    continue

                n_left = i
                n_right = n_samples - i
                left_sum = cumsum[i - 1]
                left_sq = cumsum_sq[i - 1]
                right_sum = total_sum - left_sum
                right_sq = total_sq - left_sq

                left_sse = left_sq - left_sum ** 2 / n_left
                right_sse = right_sq - right_sum ** 2 / n_right
                gain = parent_sse - (left_sse + right_sse)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float((x_sorted[i] + x_sorted[i - 1]) / 2.0)

        return best_feature, best_threshold

    # -------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._root is None:
            raise RuntimeError("call fit() before predict()")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self._root) for row in X])

    def _predict_row(self, row: np.ndarray, node: _Node) -> float:
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


def _sse(y: np.ndarray) -> float:
    """Sum of squared errors around the mean."""
    if len(y) == 0:
        return 0.0
    return float(np.sum((y - y.mean()) ** 2))

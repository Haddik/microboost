"""Loss functions for gradient boosting.

Each loss exposes two things the booster needs:

* ``init_estimate(y)`` -- the constant prediction that minimises the loss,
  used as the model's starting point (F_0).
* ``negative_gradient(y, f)`` -- the pseudo-residuals that each new tree is
  fit against, where ``f`` is the current raw model output.

Working in terms of the negative gradient is what makes the algorithm
"gradient" boosting: every tree performs one approximate step of gradient
descent in function space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


class Loss(ABC):
    @abstractmethod
    def init_estimate(self, y: np.ndarray) -> float:
        ...

    @abstractmethod
    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def __call__(self, y: np.ndarray, f: np.ndarray) -> float:
        """Mean loss, mostly for logging/monitoring."""
        ...


class SquaredError(Loss):
    """L(y, f) = 1/2 (y - f)^2.

    The negative gradient is simply the residual, which recovers the classic
    least-squares boosting of Friedman (2001).
    """

    def init_estimate(self, y: np.ndarray) -> float:
        return float(np.mean(y))

    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        return y - f

    def __call__(self, y: np.ndarray, f: np.ndarray) -> float:
        return float(0.5 * np.mean((y - f) ** 2))


class LogisticLoss(Loss):
    """Binary cross-entropy on raw scores (log-odds).

    ``f`` is the log-odds; the probability is ``sigmoid(f)``. The negative
    gradient works out to ``y - p``, the difference between the label and the
    predicted probability.
    """

    def init_estimate(self, y: np.ndarray) -> float:
        # Log-odds of the base rate, clipped to stay finite for degenerate y.
        p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    def negative_gradient(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        return y - sigmoid(f)

    def __call__(self, y: np.ndarray, f: np.ndarray) -> float:
        p = np.clip(sigmoid(f), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

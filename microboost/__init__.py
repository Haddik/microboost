"""MicroBoost -- a compact, readable gradient boosting implementation.

The public API mirrors the small subset of scikit-learn that the experiments
use, so models are essentially drop-in for ``fit`` / ``predict``.
"""

from __future__ import annotations

from .boosting import GradientBoostingClassifier, GradientBoostingRegressor
from .losses import LogisticLoss, SquaredError
from .tree import DecisionTreeRegressor

__all__ = [
    "GradientBoostingRegressor",
    "GradientBoostingClassifier",
    "DecisionTreeRegressor",
    "SquaredError",
    "LogisticLoss",
]

__version__ = "0.1.0"

"""Benchmark the MicroBoost classifier on the Breast Cancer dataset.

Usage:
    python experiments/run_classification.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier as SkGBC
from sklearn.model_selection import train_test_split

from microboost import GradientBoostingClassifier
from microboost.metrics import accuracy

PARAMS = dict(n_estimators=150, learning_rate=0.1, max_depth=3)


def log_loss(y: np.ndarray, proba: np.ndarray) -> float:
    p = np.clip(proba, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    ours = GradientBoostingClassifier(**PARAMS).fit(X_train, y_train)
    ref = SkGBC(**PARAMS, random_state=0).fit(X_train, y_train)

    p_ours = ours.predict_proba(X_test)[:, 1]
    p_ref = ref.predict_proba(X_test)[:, 1]

    header = f"{'model':<16}{'accuracy':>10}{'log-loss':>10}"
    print(header)
    print("-" * len(header))
    print(f"{'MicroBoost':<16}{accuracy(y_test, ours.predict(X_test)):>10.3f}"
          f"{log_loss(y_test, p_ours):>10.3f}")
    print(f"{'scikit-learn':<16}{accuracy(y_test, ref.predict(X_test)):>10.3f}"
          f"{log_loss(y_test, p_ref):>10.3f}")


if __name__ == "__main__":
    main()

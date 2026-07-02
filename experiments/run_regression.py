"""Benchmark MicroBoost against scikit-learn on a regression dataset.

Reproduces the headline number in the report: on the Diabetes data our
from-scratch booster lands within a small margin of the reference
implementation, which is the sanity check we care about. The dataset ships
with scikit-learn, so the experiment runs fully offline.

Usage:
    python experiments/run_regression.py
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor as SkGBR
from sklearn.model_selection import train_test_split

from microboost import GradientBoostingRegressor
from microboost.metrics import mae, r2_score, rmse

PARAMS = dict(n_estimators=200, learning_rate=0.05, max_depth=2)


def main() -> None:
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    print(f"train={len(X_train)}  test={len(X_test)}  features={X.shape[1]}\n")

    t0 = time.perf_counter()
    ours = GradientBoostingRegressor(**PARAMS).fit(X_train, y_train)
    t_ours = time.perf_counter() - t0
    pred_ours = ours.predict(X_test)

    t0 = time.perf_counter()
    ref = SkGBR(**PARAMS, random_state=0).fit(X_train, y_train)
    t_ref = time.perf_counter() - t0
    pred_ref = ref.predict(X_test)

    header = f"{'model':<16}{'RMSE':>8}{'MAE':>8}{'R2':>8}{'fit (s)':>10}"
    print(header)
    print("-" * len(header))
    for name, pred, secs in [
        ("MicroBoost", pred_ours, t_ours),
        ("scikit-learn", pred_ref, t_ref),
    ]:
        print(
            f"{name:<16}{rmse(y_test, pred):>8.3f}{mae(y_test, pred):>8.3f}"
            f"{r2_score(y_test, pred):>8.3f}{secs:>10.2f}"
        )

    gap = rmse(y_test, pred_ours) - rmse(y_test, pred_ref)
    print(f"\nRMSE gap vs scikit-learn: {gap:+.3f}")


if __name__ == "__main__":
    main()

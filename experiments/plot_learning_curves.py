"""Plot train/test error against the number of boosting stages.

Demonstrates two textbook effects:

* the learning rate trades off convergence speed vs final accuracy;
* test error flattens (and eventually creeps up) long after train error keeps
  falling -- the signature of overfitting that motivates early stopping.

Saves ``docs/learning_curves.png``.

Usage:
    python experiments/plot_learning_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from microboost import GradientBoostingRegressor
from microboost.metrics import rmse

OUT = Path(__file__).resolve().parents[1] / "docs" / "learning_curves.png"


def test_curve(model, X_test, y_test):
    return [rmse(y_test, pred) for pred in model.staged_predict(X_test)]


def main() -> None:
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    plt.figure(figsize=(8, 5))
    for lr in (0.02, 0.05, 0.2):
        model = GradientBoostingRegressor(
            n_estimators=300, learning_rate=lr, max_depth=2
        ).fit(X_train, y_train)
        curve = test_curve(model, X_test, y_test)
        plt.plot(range(1, len(curve) + 1), curve, label=f"learning_rate={lr}")

    plt.xlabel("number of trees")
    plt.ylabel("test RMSE")
    plt.title("MicroBoost on Diabetes: effect of learning rate")
    plt.legend()
    plt.grid(alpha=0.3)
    OUT.parent.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=120)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

import numpy as np

from microboost import GradientBoostingClassifier, GradientBoostingRegressor
from microboost.metrics import accuracy, rmse


def _make_regression(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 3))
    y = X[:, 0] ** 2 + np.sin(2 * X[:, 1]) + 0.5 * X[:, 2]
    y += rng.normal(scale=0.1, size=n)
    return X, y


def test_regression_beats_constant_baseline():
    X, y = _make_regression()
    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    baseline = rmse(y, np.full_like(y, y.mean()))
    assert rmse(y, model.predict(X)) < 0.5 * baseline


def test_training_loss_decreases_monotonically():
    X, y = _make_regression()
    model = GradientBoostingRegressor(n_estimators=40, learning_rate=0.1).fit(X, y)
    losses = model.train_loss_
    # Squared-error boosting is guaranteed to be non-increasing on train loss.
    assert all(b <= a + 1e-9 for a, b in zip(losses, losses[1:]))


def test_staged_predict_matches_final():
    X, y = _make_regression()
    model = GradientBoostingRegressor(n_estimators=25).fit(X, y)
    *_, last = model.staged_predict(X)
    np.testing.assert_allclose(last, model.predict(X))


def test_subsample_runs_and_learns():
    X, y = _make_regression()
    model = GradientBoostingRegressor(
        n_estimators=60, subsample=0.7, random_state=42
    ).fit(X, y)
    assert rmse(y, model.predict(X)) < rmse(y, np.full_like(y, y.mean()))


def test_classifier_separates_two_gaussians():
    rng = np.random.default_rng(7)
    n = 200
    X = np.vstack(
        [rng.normal(-1.5, 1.0, size=(n, 2)), rng.normal(1.5, 1.0, size=(n, 2))]
    )
    y = np.concatenate([np.zeros(n), np.ones(n)])
    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, max_depth=2)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (2 * n, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert accuracy(y, clf.predict(X)) > 0.9

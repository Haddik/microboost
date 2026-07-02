import numpy as np

from microboost.losses import LogisticLoss, SquaredError, sigmoid


def test_sigmoid_is_stable_for_large_inputs():
    z = np.array([-1000.0, 0.0, 1000.0])
    out = sigmoid(z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-12)


def test_squared_error_init_is_mean():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    assert SquaredError().init_estimate(y) == y.mean()


def test_squared_error_gradient_is_residual():
    y = np.array([1.0, 2.0, 3.0])
    f = np.array([0.5, 2.5, 2.0])
    np.testing.assert_allclose(SquaredError().negative_gradient(y, f), y - f)


def test_logistic_init_is_base_log_odds():
    y = np.array([1.0, 1.0, 1.0, 0.0])  # base rate 0.75
    expected = np.log(0.75 / 0.25)
    assert abs(LogisticLoss().init_estimate(y) - expected) < 1e-9


def test_logistic_gradient_is_label_minus_prob():
    y = np.array([1.0, 0.0])
    f = np.array([0.0, 0.0])  # p = 0.5
    np.testing.assert_allclose(LogisticLoss().negative_gradient(y, f), [0.5, -0.5])


def test_logistic_loss_minimised_at_true_log_odds():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.3).astype(float)
    loss = LogisticLoss()
    best = loss.init_estimate(y)
    f_best = np.full_like(y, best)
    f_off = np.full_like(y, best + 0.5)
    assert loss(y, f_best) < loss(y, f_off)

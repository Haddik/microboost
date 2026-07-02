import numpy as np

from microboost.tree import DecisionTreeRegressor, _sse


def test_perfect_fit_on_step_function():
    # A depth-1 tree should split a clean step function exactly.
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 5.0, 5.0])
    tree = DecisionTreeRegressor(max_depth=1)
    tree.fit(X, y)
    np.testing.assert_allclose(tree.predict(X), y)


def test_constant_target_returns_leaf():
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.full(20, 4.2)
    tree = DecisionTreeRegressor(max_depth=5).fit(X, y)
    np.testing.assert_allclose(tree.predict(X), 4.2)


def test_depth_limits_capacity():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 1))
    y = np.sin(X[:, 0] * 3)
    shallow = DecisionTreeRegressor(max_depth=1).fit(X, y)
    deep = DecisionTreeRegressor(max_depth=6).fit(X, y)
    shallow_err = np.mean((shallow.predict(X) - y) ** 2)
    deep_err = np.mean((deep.predict(X) - y) ** 2)
    assert deep_err < shallow_err


def test_min_samples_leaf_respected():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 2))
    y = rng.normal(size=50)
    tree = DecisionTreeRegressor(max_depth=None, min_samples_leaf=10).fit(X, y)

    leaf_sizes: list[int] = []

    def walk(node, mask):
        if node.is_leaf:
            leaf_sizes.append(int(mask.sum()))
            return
        left = mask & (X[:, node.feature] <= node.threshold)
        right = mask & (X[:, node.feature] > node.threshold)
        walk(node.left, left)
        walk(node.right, right)

    walk(tree._root, np.ones(len(X), dtype=bool))
    assert min(leaf_sizes) >= 10


def test_sse_zero_for_constant():
    assert _sse(np.array([3.0, 3.0, 3.0])) == 0.0

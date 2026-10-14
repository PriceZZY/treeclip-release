import numpy as np
from treeclip.train.linear_probe import linear_probe_macro_f1


def test_linear_probe_on_separable_data():
    """Three well-separated clusters → near-perfect F1."""
    rng = np.random.default_rng(0)
    X_train = np.concatenate([
        rng.normal(loc=[0, 0], scale=0.1, size=(20, 2)),
        rng.normal(loc=[5, 5], scale=0.1, size=(20, 2)),
        rng.normal(loc=[-5, 5], scale=0.1, size=(20, 2)),
    ]).astype("float32")
    y_train = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    X_val, y_val = X_train.copy(), y_train.copy()
    f1 = linear_probe_macro_f1(X_train, y_train, X_val, y_val)
    assert f1 > 0.9


def test_linear_probe_returns_zero_for_single_class():
    X = np.random.RandomState(0).rand(20, 5)
    y = np.array(["a"] * 20)
    f1 = linear_probe_macro_f1(X, y, X, y)
    assert f1 == 0.0


def test_linear_probe_drops_non_finite():
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0], [6.0, 7.0]])
    y = np.array(["a", "a", "b", "b"])
    f1 = linear_probe_macro_f1(X, y, X, y)
    # Two rows dropped due to NaN; remaining is 3 finite rows → should still produce a number in [0,1]
    assert 0.0 <= f1 <= 1.0

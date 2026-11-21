import numpy as np
import pytest

from mlp import MLP, SGD, Adam, train, check_gradients
from mlp.network import softmax


@pytest.fixture
def small_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 3))
    y = rng.integers(0, 3, 8)
    return X, y


@pytest.mark.parametrize("activation", ["relu", "tanh"])
@pytest.mark.parametrize("l2", [0.0, 0.01])
def test_backward_matches_numerical_gradient(small_data, activation, l2):
    X, y = small_data
    model = MLP([3, 5, 4, 3], activation=activation, l2=l2, seed=2)
    assert check_gradients(model, X, y) < 1e-6


def test_softmax_rows_sum_to_one_and_handle_large_values():
    p = softmax(np.array([[1000.0, 1000.0], [1.0, 2.0]]))
    assert np.allclose(p.sum(axis=1), 1)
    assert np.allclose(p[0], [0.5, 0.5])


@pytest.mark.parametrize("opt", [SGD(lr=0.1), Adam(lr=0.01)])
def test_learns_xor(opt):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    model = MLP([2, 8, 2], activation="tanh", seed=0)
    train(model, X, y, opt, epochs=500, batch_size=4)
    assert (model.predict(X) == y).all()

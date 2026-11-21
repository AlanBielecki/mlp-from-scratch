import numpy as np


def numerical_gradient(f, param, eps=1e-5):
    """Central difference estimate of df/dparam, one element at a time."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = param[i]
        param[i] = old + eps
        plus = f()
        param[i] = old - eps
        minus = f()
        param[i] = old
        grad[i] = (plus - minus) / (2 * eps)
    return grad


def relative_error(a, b):
    return np.abs(a - b).max() / max(1e-8, np.abs(a).max() + np.abs(b).max())


def check_gradients(model, X, y):
    """Compare backward() with numerical gradients for every parameter.

    Returns the largest relative error. Below ~1e-6 means the backward pass
    is right (in float64).
    """
    model.forward(X)
    dW, db = model.backward(y)
    worst = 0.0
    for param, analytic in zip(model.params, dW + db):
        numeric = numerical_gradient(lambda: model.loss(X, y), param)
        worst = max(worst, relative_error(analytic, numeric))
    return worst

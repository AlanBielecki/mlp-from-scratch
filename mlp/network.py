"""A small fully connected network written with NumPy only.

Forward pass, backward pass and the update step are all written out by hand,
so every gradient can be checked against a numerical estimate.
"""

from __future__ import annotations

import numpy as np


# ---------- activations ----------

def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1 - np.tanh(z) ** 2


ACTIVATIONS = {"relu": (relu, relu_grad), "tanh": (tanh, tanh_grad)}


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # avoids overflow in exp
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    """Mean negative log-likelihood. y holds class indices."""
    n = y.shape[0]
    return -np.log(probs[np.arange(n), y] + 1e-12).mean()


# ---------- network ----------

class MLP:
    def __init__(self, sizes: list[int], activation: str = "relu", l2: float = 0.0, seed: int = 0):
        """sizes = [n_inputs, hidden_1, ..., n_classes]"""
        rng = np.random.default_rng(seed)
        self.act, self.act_grad = ACTIVATIONS[activation]
        self.l2 = l2
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # He initialisation for relu, Xavier for tanh
            scale = np.sqrt(2 / n_in) if activation == "relu" else np.sqrt(1 / n_in)
            self.W.append(rng.normal(0, scale, (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    @property
    def params(self):
        return self.W + self.b

    def forward(self, X):
        """Returns class probabilities and keeps what backward() needs."""
        self._a = [X]   # layer inputs
        self._z = []    # pre-activations
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            self._z.append(z)
            a = softmax(z) if i == len(self.W) - 1 else self.act(z)
            self._a.append(a)
        return a

    def loss(self, X, y):
        data_loss = cross_entropy(self.forward(X), y)
        reg = 0.5 * self.l2 * sum((W ** 2).sum() for W in self.W)
        return data_loss + reg

    def backward(self, y):
        """Gradients of loss() w.r.t. W and b, after a forward() call."""
        n = y.shape[0]
        probs = self._a[-1]

        # softmax + cross-entropy together have a simple gradient: p - one_hot(y)
        delta = probs.copy()
        delta[np.arange(n), y] -= 1
        delta /= n

        dW, db = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            dW[i] = self._a[i].T @ delta + self.l2 * self.W[i]
            db[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * self.act_grad(self._z[i - 1])
        return dW, db

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


# ---------- optimisers ----------

class SGD:
    def __init__(self, lr=0.1, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.momentum
            v -= self.lr * g
            p += v


class Adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g ** 2
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train(model, X, y, optimizer, epochs=200, batch_size=32, seed=0, X_val=None, y_val=None):
    rng = np.random.default_rng(seed)
    history = {"loss": [], "val_acc": []}
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            model.forward(X[idx])
            dW, db = model.backward(y[idx])
            optimizer.step(model.params, dW + db)
        history["loss"].append(model.loss(X, y))
        if X_val is not None:
            history["val_acc"].append((model.predict(X_val) == y_val).mean())
    return history

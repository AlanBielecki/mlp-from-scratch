"""Train the network on two toy datasets and compare optimisers.

    python demo.py

Saves the decision boundary plot to images/decision_boundaries.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split

from mlp import MLP, SGD, Adam, train

Path("images").mkdir(exist_ok=True)

datasets = {
    "moons": make_moons(n_samples=600, noise=0.25, random_state=0),
    "circles": make_circles(n_samples=600, noise=0.1, factor=0.5, random_state=0),
}
optimisers = {
    "SGD + momentum": lambda: SGD(lr=0.05, momentum=0.9),
    "Adam": lambda: Adam(lr=0.01),
}

fig, axes = plt.subplots(len(datasets), len(optimisers), figsize=(10, 9))

print(f"{'dataset':<10}{'optimiser':<17}{'train acc':>10}{'test acc':>10}")
for r, (name, (X, y)) in enumerate(datasets.items()):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    for c, (opt_name, make_opt) in enumerate(optimisers.items()):
        model = MLP([2, 16, 16, 2], activation="relu", l2=1e-4, seed=0)
        train(model, X_tr, y_tr, make_opt(), epochs=150, batch_size=32)
        tr = (model.predict(X_tr) == y_tr).mean()
        te = (model.predict(X_te) == y_te).mean()
        print(f"{name:<10}{opt_name:<17}{tr:>10.3f}{te:>10.3f}")

        # decision boundary
        xx, yy = np.meshgrid(
            np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
            np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200),
        )
        probs = model.forward(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        ax = axes[r, c]
        ax.imshow(probs, extent=(xx.min(), xx.max(), yy.min(), yy.max()), origin="lower", cmap="RdBu", alpha=0.6, aspect="auto")
        ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="RdBu", edgecolors="k", s=18)
        ax.set_title(f"{name}, {opt_name}: test acc {te:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])

fig.tight_layout()
fig.savefig("images/decision_boundaries.png", dpi=110)
print("saved images/decision_boundaries.png")

# mlp-from-scratch

A multilayer perceptron written in NumPy only: forward pass, backpropagation, softmax with cross-entropy, L2 regularisation, SGD with momentum and Adam. No PyTorch, no autograd.

I wrote it while studying deep learning, to make sure I understand what a framework does when it calls `loss.backward()`. The most useful part turned out to be the gradient check: every gradient from the backward pass is compared with a numerical estimate, and the tests fail if they differ.

## Structure

```
mlp/
  network.py     MLP class, activations, softmax / cross-entropy, SGD, Adam, training loop
  gradcheck.py   numerical gradients (central differences) and the comparison
tests/
  test_mlp.py    gradient check for relu/tanh with and without L2, softmax stability, XOR
demo.py          trains on two toy datasets and plots decision boundaries
```

## Backward pass in short

For the output layer, softmax and cross-entropy combined give a simple gradient:

```
dL/dz_out = (p - one_hot(y)) / n
```

Then for each layer, going backwards:

```
dW  = a_prev.T @ delta + l2 * W
db  = sum(delta, axis=0)
delta_prev = (delta @ W.T) * activation'(z_prev)
```

`check_gradients()` perturbs every single weight by ±1e-5, recomputes the loss and compares. In float64 the relative error comes out around 1e-10. A bug in the backward pass (wrong transpose, missing L2 term) shows up as an error around 1e-1 or worse.

## Usage

```bash
pip install -r requirements.txt
pytest
python demo.py
```

```python
from mlp import MLP, Adam, train

model = MLP([2, 16, 16, 2], activation="relu", l2=1e-4)
train(model, X_train, y_train, Adam(lr=0.01), epochs=150, batch_size=32)
accuracy = (model.predict(X_test) == y_test).mean()
```

## Results

Network `2 → 16 → 16 → 2`, ReLU, L2 = 1e-4, 150 epochs, 70/30 train/test split, 600 points:

| dataset | optimiser | train acc | test acc |
|---|---|---|---|
| moons (noise 0.25) | SGD + momentum | 0.957 | 0.917 |
| moons (noise 0.25) | Adam | 0.964 | 0.911 |
| circles (noise 0.1) | SGD + momentum | 1.000 | 0.983 |
| circles (noise 0.1) | Adam | 1.000 | 0.978 |

`demo.py` also saves the decision boundaries to `images/decision_boundaries.png`. On "moons" the noise makes the classes overlap, so ~92% is close to what any model can get without overfitting.

## What I'd add next

- dropout and batch normalisation, with their own gradient checks
- a comparison with the same network in PyTorch on MNIST, to check that results match

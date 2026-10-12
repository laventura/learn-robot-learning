"""Tiny neural networks in NumPy: an MLP with hand-written backprop, Adam,
and mini-batching. Starts from the ADM Companion's adm/neural.py.

Shapes: a batch X has one sample per row, (n, d_in). A layer is a pair
(W, b) with W of shape (d_in, d_out), so a layer computes X @ W + b.
Hidden layers apply the activation; the output layer is linear, so a
policy's output is either a continuous action (regression) or logits
over discrete actions (turned into probabilities by softmax).

Training is split into pieces, so the same code serves supervised
learning (behavior cloning) and reinforcement learning:
- a loss maps the network output to (loss, dLoss/dOutput);
- backward() turns dLoss/dOutput into parameter gradients;
- Adam applies the gradients.
fit() wires the three together for a fixed dataset. A policy-gradient
loop calls forward(), computes its own dLoss/dOutput, then backward()
and Adam.step().
"""

from collections.abc import Callable, Iterator

import numpy as np

Array = np.ndarray
Layers = list[tuple[Array, Array]]                      # [(W, b), ...]
Loss = Callable[[Array, Array], tuple[float, Array]]    # (Yhat, Y) -> (loss, dYhat)

# Each activation is (phi, dphi): the function and its derivative.
ACTIVATIONS = {
    "relu": (lambda z: np.maximum(z, 0.0), lambda z: (z > 0) * 1.0),
    "leaky_relu": (lambda z: np.where(z > 0, z, 0.1 * z),
                   lambda z: np.where(z > 0, 1.0, 0.1)),
    "tanh": (np.tanh, lambda z: 1.0 - np.tanh(z) ** 2),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)),
                lambda z: np.exp(-z) / (1 + np.exp(-z)) ** 2),
    "identity": (lambda z: z, lambda z: np.ones_like(z)),
}


def softmax(z, axis=-1) -> Array:
    """exp(z_i) / sum_j exp(z_j). Subtracting the max first changes
    nothing mathematically but avoids overflow in exp."""
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def init_mlp(sizes, rng) -> Layers:
    """Layers for sizes = [d_in, h1, ..., d_out]. Weights ~ N(0, 1/d_in),
    so each unit's input starts with variance about 1."""
    return [(rng.normal(0, 1 / np.sqrt(m), (m, n)), np.zeros(n))
            for m, n in zip(sizes[:-1], sizes[1:])]


def forward(params: Layers, X, act="tanh"):
    """Returns the output and a cache of (layer input, pre-activation)
    per layer, which the backward pass needs."""
    phi = ACTIVATIONS[act][0]
    A, cache = X, []
    for k, (W, b) in enumerate(params):
        Z = A @ W + b                   # (n, d_in) @ (d_in, d_out)
        cache.append((A, Z))
        A = Z if k == len(params) - 1 else phi(Z)   # linear output layer
    return A, cache


def backward(params: Layers, cache, dY, act="tanh") -> Layers:
    """Chain rule from the output back: dY is dLoss/dOutput, (n, d_out).
    Returns [(dW, db)] matching params."""
    dphi = ACTIVATIONS[act][1]
    grads = [None] * len(params)
    dA = dY
    for k in reversed(range(len(params))):
        A_prev, Z = cache[k]
        # Through the activation (skipped at the linear output layer)
        dZ = dA if k == len(params) - 1 else dA * dphi(Z)
        W = params[k][0]
        # Z = A_prev @ W + b: each weight's gradient sums over the batch
        grads[k] = (A_prev.T @ dZ, dZ.sum(axis=0))
        dA = dZ @ W.T                   # hand the gradient to layer k - 1
    return grads


def predict(params: Layers, x, act="tanh") -> Array:
    """The network's output for one observation (d_in,) or a batch
    (n, d_in). A policy acts on one observation at a time, so a 1-D
    input gives a 1-D output (d_out,)."""
    x = np.asarray(x, dtype=float)
    out, _ = forward(params, np.atleast_2d(x), act)
    return out[0] if x.ndim == 1 else out


def mse(Yhat, Y) -> tuple[float, Array]:
    """Squared error summed over output dimensions, averaged over the
    batch, and its gradient with respect to Yhat. Y has Yhat's shape."""
    err = Yhat - Y                       # (n, d_out)
    return (err ** 2).sum() / len(Y), 2 * err / len(Y)


def cross_entropy(logits, labels) -> tuple[float, Array]:
    """Mean of -log softmax(logits)[label], and its gradient with respect
    to the logits. labels holds class indices, shape (n,).
    The gradient is (probabilities - one-hot) / n: raise the logit of the
    correct class, lower the others in proportion to their probability."""
    n = len(labels)
    p = softmax(logits)                  # (n, n_classes)
    rows = np.arange(n)
    loss = -np.log(p[rows, labels] + 1e-12).mean()
    dlogits = p.copy()
    dlogits[rows, labels] -= 1.0         # subtract the one-hot
    return loss, dlogits / n


LOSSES: dict[str, Loss] = {"mse": mse, "cross_entropy": cross_entropy}


def loss_and_grads(params: Layers, X, Y, loss="mse", act="tanh", beta=0.0):
    """The loss on (X, Y) plus beta * (sum of squared weights), and its
    gradient for every layer. Biases are not penalized."""
    Yhat, cache = forward(params, X, act)
    value, dY = LOSSES[loss](Yhat, Y)
    value += beta * sum((W ** 2).sum() for W, _ in params)
    grads = backward(params, cache, dY, act)
    grads = [(dW + 2 * beta * W, db) for (dW, db), (W, _) in zip(grads, params)]
    return value, grads


class Adam:
    """Adam: each step moves each parameter by alpha times its running
    mean gradient over the root of its running mean squared gradient.
    The two running means start at zero, so early steps divide them by
    (1 - b**t) to undo that bias."""

    def __init__(self, params: Layers, alpha=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.alpha, self.b1, self.b2, self.eps = alpha, b1, b2, eps
        # One running mean per array, in params order: W1, b1, W2, b2, ...
        self.m = [np.zeros_like(p) for layer in params for p in layer]
        self.v = [np.zeros_like(p) for layer in params for p in layer]
        self.t = 0

    def step(self, params: Layers, grads: Layers) -> Layers:
        """New parameters after one descent step; params is not modified."""
        self.t += 1
        flat = [p for layer in params for p in layer]
        flat_grads = [g for layer in grads for g in layer]
        new = []
        for i, (p, g) in enumerate(zip(flat, flat_grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            new.append(p - self.alpha * mhat / (np.sqrt(vhat) + self.eps))
        return list(zip(new[::2], new[1::2]))           # back to [(W, b), ...]


def minibatches(n, size, rng) -> Iterator[Array]:
    """Index arrays that split range(n) into shuffled batches of `size`
    (the last may be smaller). size=None gives one batch of everything."""
    order = rng.permutation(n)
    size = size or n
    for start in range(0, n, size):
        yield order[start:start + size]


def fit(params: Layers, X, Y, loss="mse", act="tanh", alpha=0.01, epochs=2000,
        beta=0.0, batch=None, rng=None):
    """Train on a fixed dataset with Adam. Returns the trained params and
    the full-data loss after each epoch."""
    rng = rng or np.random.default_rng(0)
    opt = Adam(params, alpha)
    losses = []
    for _ in range(epochs):
        for idx in minibatches(len(X), batch, rng):
            _, grads = loss_and_grads(params, X[idx], Y[idx], loss, act, beta)
            params = opt.step(params, grads)
        losses.append(loss_and_grads(params, X, Y, loss, act, beta)[0])
    return params, np.array(losses)

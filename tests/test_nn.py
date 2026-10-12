"""Tests for rlc.nn."""

import numpy as np
import pytest

from rlc import nn as N


def numeric_grads_match(params, X, Y, loss, act):
    _, grads = N.loss_and_grads(params, X, Y, loss, act, beta=0.01)
    h = 1e-6
    for k, (W, b) in enumerate(params):
        for P, G in ((W, grads[k][0]), (b, grads[k][1])):
            for idx in np.ndindex(P.shape):
                old = P[idx]
                P[idx] = old + h
                up = N.loss_and_grads(params, X, Y, loss, act, beta=0.01)[0]
                P[idx] = old - h
                down = N.loss_and_grads(params, X, Y, loss, act, beta=0.01)[0]
                P[idx] = old
                assert np.isclose((up - down) / (2 * h), G[idx], atol=1e-5)


@pytest.mark.parametrize("act", ["relu", "leaky_relu", "tanh", "sigmoid", "identity"])
def test_mse_backprop_matches_finite_differences(act):
    rng = np.random.default_rng(1)
    params = N.init_mlp([3, 5, 4, 2], rng)
    numeric_grads_match(params, rng.normal(size=(7, 3)), rng.normal(size=(7, 2)), "mse", act)


def test_cross_entropy_backprop_matches_finite_differences():
    rng = np.random.default_rng(3)
    params = N.init_mlp([3, 5, 4], rng)
    numeric_grads_match(params, rng.normal(size=(7, 3)), rng.integers(0, 4, 7), "cross_entropy", "tanh")


def test_softmax():
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
    s = N.softmax(z)
    assert np.allclose(s.sum(axis=1), 1) and np.allclose(s[1], 1 / 3)
    assert np.allclose(N.softmax(z[:1] + 5), s[:1])


def test_regression_fits_to_noise_level():
    rng = np.random.default_rng(0)
    f = lambda x: np.sin(2 * x) * np.exp(-x ** 2 / 4)
    X = rng.uniform(-3, 3, (60, 1))
    Y = f(X) + rng.normal(0, 0.05, X.shape)
    params, losses = N.fit(N.init_mlp([1, 16, 16, 1], rng), X, Y, "mse", "tanh",
                           alpha=0.01, epochs=1500)
    grid = np.linspace(-3, 3, 200)[:, None]
    assert losses[-1] < 0.004 and losses[-1] < losses[0] / 20
    assert np.mean((N.predict(params, grid) - f(grid)) ** 2) < 0.005


def test_classifier_separates_two_blobs_with_minibatches():
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(-2, 0.5, (40, 2)), rng.normal(2, 0.5, (40, 2))])
    labels = np.repeat([0, 1], 40)
    params, losses = N.fit(N.init_mlp([2, 8, 2], rng), X, labels, "cross_entropy",
                           "relu", alpha=0.02, epochs=100, batch=16, rng=rng)
    assert (N.predict(params, X, "relu").argmax(axis=1) == labels).all()
    assert losses[-1] < 0.05 < losses[0]


def test_predict_one_observation_gives_one_action():
    params = N.init_mlp([4, 6, 2], np.random.default_rng(5))
    obs = np.ones(4)
    assert N.predict(params, obs).shape == (2,)
    assert np.allclose(N.predict(params, obs), N.predict(params, obs[None])[0])


def test_adam_leaves_params_unchanged_and_descends():
    rng = np.random.default_rng(6)
    params = N.init_mlp([2, 3, 1], rng)
    before = [W.copy() for W, _ in params]
    X, Y = rng.normal(size=(10, 2)), rng.normal(size=(10, 1))
    opt = N.Adam(params, alpha=0.01)
    loss0, grads = N.loss_and_grads(params, X, Y)
    new = opt.step(params, grads)
    assert all(np.array_equal(W, W0) for (W, _), W0 in zip(params, before))
    assert N.loss_and_grads(new, X, Y)[0] < loss0


def test_minibatches_partition_the_data():
    batches = list(N.minibatches(10, 4, np.random.default_rng(7)))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(np.concatenate(batches)) == list(range(10))
    assert len(list(N.minibatches(10, None, np.random.default_rng(7)))) == 1


def test_linear_layers_collapse():
    rng = np.random.default_rng(2)
    params = N.init_mlp([2, 6, 3], rng)
    X = rng.normal(size=(5, 2))
    (W1, b1), (W2, b2) = params
    assert np.allclose(N.forward(params, X, "identity")[0], X @ (W1 @ W2) + (b1 @ W2 + b2))

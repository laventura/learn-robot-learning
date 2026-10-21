import numpy as np
import pytest

from rlc.toys import w07_sequence as w


# --- Toy 1: attention ----------------------------------------------------------------

def test_cue_batch_targets_the_later_goal():
    X, y, pos = w.cue_batch(500, 12, np.random.default_rng(0))
    assert X.shape == (500, 12, 3) and (pos[:, 0] < pos[:, 1]).all() and (pos[:, 1] <= 10).all()
    assert (X[:, :, 0].sum(1) == 2).all() and (X[:, -1] == [0, 0, 1]).all()
    assert np.allclose(y, X[np.arange(500), pos[:, 1], 1])


@pytest.mark.parametrize("relative", [False, True])
def test_attention_gradients_match_finite_differences(relative):
    rng = np.random.default_rng(1)
    params = w.init_attention(rng)
    params[1] = (params[1][0], np.array([0.3]))
    X, y, _ = w.cue_batch(8, 6, rng)
    _, grads = w.attention_grads(params, X, y, relative)
    for (pa, pb), (ga, gb) in zip(params, grads):
        for arr, g in ((pa, ga), (pb, gb)):
            for idx in np.ndindex(arr.shape):
                old = arr[idx]
                arr[idx] = old + 1e-6
                up = w.attention_grads(params, X, y, relative)[0]
                arr[idx] = old - 1e-6
                down = w.attention_grads(params, X, y, relative)[0]
                arr[idx] = old
                assert abs((up - down) / 2e-6 - g[idx]) < 1e-6


@pytest.fixture(scope="module")
def trained():
    return {rel: w.train_attention(rel, 0)[0] for rel in (False, True)}


def test_without_position_the_head_can_only_average(trained):
    # The best order-blind answer is the mean of the two goals: MSE 1/2.
    mse = w.attention_mse(trained[False], w.L_TRAIN, False, np.random.default_rng(2))
    assert abs(mse - 0.5) < 0.03


def test_relative_bias_retrieves_the_later_goal_within_trained_distance(trained):
    params = trained[True]
    assert w.attention_mse(params, w.L_TRAIN, True, np.random.default_rng(3)) < 0.002
    X, y, pos = w.cue_batch(20_000, 128, np.random.default_rng(4))
    err = (w.attend(params, X, True)[0] - y) ** 2
    back = 127 - pos[:, 1]
    assert err[back <= 14].mean() < 0.002          # any length, if the goal is recent enough
    assert err[back >= 17].mean() > 1.0            # further back than training ever showed


# --- Toy 2: tokens -------------------------------------------------------------------

def test_bpe_on_text_and_integers():
    merges, corpus = w.bpe_train(["abab", "abab", "abc"], 5)
    # (a, b) first; then (ab, ab), seen twice; then no pair repeats.
    assert merges == [("a", "b"), (("a", "b"), ("a", "b"))]
    assert [len(s) for s in corpus] == [1, 1, 2]
    assert "".join(w.unpack(corpus[0][0])) == "abab"
    enc = w.bpe_encode([0, 0, 0, 0, 5], [(0, 0)])
    assert enc == [(0, 0), (0, 0), 5] and sum((w.unpack(t) for t in enc), []) == [0, 0, 0, 0, 5]


@pytest.fixture(scope="module")
def actions():
    return w.signal_params(300, np.random.default_rng(0))


def test_dct_round_trip_is_close(actions):
    A = w.sample_actions(actions, 50)
    b = w.quantile_bounds(A)
    rec = w.dct_reconstruct(w.dct_coefficients(A, b, 1e6), b, 1e6)
    inside = (A >= b[0]) & (A <= b[1])               # clipping is the only loss left
    assert np.abs(rec - A)[inside].max() < 1e-4


def test_bins_get_predictable_and_dct_tokens_stay_few(actions):
    copy, nonzero = [], []
    for f in w.FREQS:
        A = w.sample_actions(actions, f)
        b = w.quantile_bounds(A)
        tok = w.bin_tokens(A, b)
        assert tok.min() >= 0 and tok.max() == w.N_BINS - 1
        copy.append(w.copy_rate(tok))
        nonzero.append((w.dct_coefficients(A, b, 10) != 0).sum((1, 2)).mean())
    assert all(a < b for a, b in zip(copy, copy[1:]))       # copying gets easier with rate
    assert nonzero[-1] < 0.2 * w.FREQS[-1] * w.D_ACT         # most coefficients round to 0


# --- Toy 3: chunks -------------------------------------------------------------------

def test_noise_free_runs_reach_both_goals():
    for k, mode in ((1, "chunk"), (25, "chunk"), (25, "ensemble")):
        x = w.run_chunks(k, mode, np.random.default_rng(0), noise=0.0)
        assert abs(x[w.T_MOVE] - w.GOAL_A) < 0.02 and abs(x[-1] - w.GOAL_B) < 0.02


def test_chunk_trade_offs():
    def stats(k, mode):
        runs = [w.run_chunks(k, mode, np.random.default_rng(s)) for s in range(100)]
        return np.mean([w.roughness(x) for x in runs]), np.mean([w.settle_error(x) for x in runs])
    rough1, err1 = stats(1, "chunk")
    rough_c, err_c = stats(25, "chunk")
    rough_e, err_e = stats(25, "ensemble")
    assert rough_c < 0.6 * rough1 and rough_e < 0.6 * rough1       # chunks are smoother
    assert err_c > 1.8 * err1 and err_e < 1.3 * err1                # the ensemble keeps accuracy
    assert w.mean_half_time(25, "chunk") > 1.5 * w.mean_half_time(1, "chunk")
    assert w.mean_half_time(25, "ensemble") > 1.5 * w.mean_half_time(1, "chunk")

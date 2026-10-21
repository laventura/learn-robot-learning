"""Week 7 toys: attention as memory, action tokens, and action chunks.

Toy 1, memory by attention. A robot sees a stream of tokens; twice, at
random moments, a token announces a goal, and the later announcement
replaces the earlier one. At the end the robot must act on the latest
goal. A reactive policy sees only the last token and cannot know it. A
single attention head (written out in NumPy with its gradients) can look
back at every token. Without positional information it cannot tell which
announcement came later; with a relative-position bias on its scores it
can, but only as far back as training required.

Toy 2, action tokens. Smooth action chunks sampled at 2 to 50 Hz are
turned into tokens two ways: one bin per dimension per time step (RT-2,
OpenVLA), and a discrete cosine transform, rounded, then compressed with
byte-pair encoding (FAST). At high rates neighbouring bin tokens are
nearly always equal, so "copy the last token" predicts most of them.

Toy 3, action chunks. A robot moves to a goal it estimates from noisy
observations; partway through, the goal moves. Single-step control,
open-loop chunks, and chunks with ACT's temporal ensemble trade jitter,
error and reaction delay differently.
"""

from collections import Counter

import numpy as np
from scipy.fft import dct, idct

from rlc.nn import Adam, softmax

Array = np.ndarray

# --- Toy 1: two goal announcements, one attention head ----------------------

L_TRAIN, D_K = 16, 2            # training sequence length; query/key width
FEATURES = ("is goal", "value", "constant")


def cue_batch(n, L, rng) -> tuple[Array, Array, Array]:
    """n sequences of L tokens, each [is goal, value, 1]; shape (n, L, 3).

    Two goal tokens sit at distinct random positions among the first L - 1;
    the other tokens are sensor readings with random values. The last
    token is the query, [0, 0, 1]: "what is my goal now?". Returns the
    tokens, the target (the later goal's value) and the two positions.
    """
    X = np.zeros((n, L, 3))
    X[:, :, 2] = 1.0
    X[:, :-1, 1] = rng.standard_normal((n, L - 1))
    # Two distinct positions per row: the first two of a random permutation.
    pos = np.sort(rng.random((n, L - 1)).argsort(axis=1)[:, :2], axis=1)
    rows = np.arange(n)
    X[rows, pos[:, 0], 0] = X[rows, pos[:, 1], 0] = 1.0
    return X, X[rows, pos[:, 1], 1], pos


def init_attention(rng) -> list[tuple[Array, Array]]:
    """[(W_q, W_k), (w_v, lam)], in pairs so rlc.nn.Adam can step them."""
    W_q = 0.5 * rng.standard_normal((3, D_K))
    W_k = 0.5 * rng.standard_normal((3, D_K))
    return [(W_q, W_k), (0.1 * rng.standard_normal(3), np.zeros(1))]


def attend(params, X, relative=True) -> tuple[Array, tuple]:
    """One attention head, read out at the last token: (n,) and a cache.

    score_i = q . k_i / sqrt(d) - lam * (how many steps back token i is).
    The last token is the query and every other token is in its past, so
    the causal mask removes nothing here. relative=False drops the bias,
    and the head then sees its inputs as an unordered set.
    """
    (W_q, W_k), (w_v, lam) = params
    L = X.shape[1]
    q = X[:, -1] @ W_q              # (n, d): what the query asks for
    K = X @ W_k                     # (n, L, d): what each token offers
    back = np.arange(L)[::-1]       # (L,): L-1, ..., 1, 0 steps back
    s = np.einsum("nld,nd->nl", K, q) / np.sqrt(D_K)
    if relative:
        s = s - lam[0] * back       # the same bias at every length
    a = softmax(s, axis=1)          # (n, L): weights, each row sums to 1
    v = X @ w_v                     # (n, L): what each token contributes
    return (a * v).sum(axis=1), (q, K, a, v, back)


def attention_grads(params, X, y, relative=True) -> tuple[float, list]:
    """Mean squared error and its gradient for every parameter, by hand."""
    yhat, (q, K, a, v, back) = attend(params, X, relative)
    dy = 2 * (yhat - y) / len(y)                  # (n,)
    da, dv = dy[:, None] * v, dy[:, None] * a     # yhat = sum_i a_i v_i
    # Softmax backward: ds_i = a_i (da_i - sum_j a_j da_j).
    ds = a * (da - (a * da).sum(axis=1, keepdims=True))
    dw_v = np.einsum("nl,nlf->f", dv, X)
    dlam = np.array([-(ds * back).sum()]) if relative else np.zeros(1)
    dK = ds[:, :, None] * q[:, None, :] / np.sqrt(D_K)     # (n, L, d)
    dq = np.einsum("nl,nld->nd", ds, K) / np.sqrt(D_K)     # (n, d)
    dW_k = np.einsum("nlf,nld->fd", X, dK)
    dW_q = X[:, -1].T @ dq
    loss = float(np.mean((yhat - y) ** 2))
    return loss, [(dW_q, dW_k), (dw_v, dlam)]


def train_attention(relative, seed, steps=3000, n=256, alpha=0.03):
    """Adam on fresh batches of length L_TRAIN; returns params and losses."""
    rng = np.random.default_rng(seed)
    params = init_attention(rng)
    opt, losses = Adam(params, alpha), []
    for _ in range(steps):
        X, y, _ = cue_batch(n, L_TRAIN, rng)
        loss, grads = attention_grads(params, X, y, relative)
        params = opt.step(params, grads)
        losses.append(loss)
    return params, np.array(losses)


def attention_mse(params, L, relative, rng, n=20_000) -> float:
    """Test error on fresh sequences of length L."""
    X, y, _ = cue_batch(n, L, rng)
    return float(np.mean((attend(params, X, relative)[0] - y) ** 2))


# --- Toy 2: from smooth actions to tokens -----------------------------------

D_ACT, N_BINS = 2, 256
FREQS = (2, 5, 10, 20, 50)      # control rates in Hz; each chunk lasts 1 s
N_WAVES, MAX_HZ = 3, 1.0        # each dimension: 3 sinusoids up to 1 Hz


def signal_params(n, rng, dims=D_ACT) -> tuple[Array, Array, Array]:
    """Amplitudes, frequencies (Hz) and phases, each (n, dims, N_WAVES).

    Drawing them once and sampling at several rates gives the same motions
    at every control frequency.
    """
    shape = (n, dims, N_WAVES)
    return (rng.uniform(0.2, 1.0, shape), rng.uniform(0.1, MAX_HZ, shape),
            rng.uniform(0, 2 * np.pi, shape))


def sample_actions(params, f) -> Array:
    """The chunks sampled at f Hz over one second: (n, f, dims)."""
    amp, freq, phase = params
    t = np.arange(f)[None, :, None, None] / f            # (1, f, 1, 1)
    # Broadcast to (n, f, dims, waves), then sum the waves.
    waves = amp[:, None] * np.sin(2 * np.pi * freq[:, None] * t
                                  + phase[:, None])
    return waves.sum(axis=-1)


def quantile_bounds(A, lo=1.0, hi=99.0) -> Array:
    """Per-dimension 1st and 99th percentiles over all samples: (2, dims)."""
    return np.percentile(A.reshape(-1, A.shape[-1]), [lo, hi], axis=0)


def normalize(A, bounds) -> Array:
    """Map [low, high] to [-1, 1] per dimension, clipping the rest."""
    low, high = bounds
    return np.clip(2 * (A - low) / (high - low) - 1, -1, 1)


def bin_tokens(A, bounds, bins=N_BINS) -> Array:
    """One integer in 0..bins-1 per dimension per step, same shape as A."""
    u = normalize(A, bounds)
    return np.minimum(((u + 1) / 2 * bins).astype(int), bins - 1)


def bin_centres(tokens, bounds, bins=N_BINS) -> Array:
    """The action each token stands for: its bin's centre."""
    low, high = bounds
    return low + (tokens + 0.5) / bins * (high - low)


def copy_rate(tokens) -> float:
    """Share of tokens equal to the previous step's, same dimension."""
    return float(np.mean(tokens[:, 1:] == tokens[:, :-1]))


def copy_explained(tokens) -> float:
    """Share of the tokens' variance that "repeat the last token" predicts.

    1 - Var(token_t - token_{t-1}) / Var(token_t): near 1 means the next
    token carries little that the last one did not already say.
    """
    t = tokens.astype(float)
    return float(1 - np.var(t[:, 1:] - t[:, :-1]) / np.var(t))


def dct_coefficients(A, bounds, scale) -> Array:
    """Normalize, DCT along time, scale and round: integers (n, f, dims).

    A smooth signal puts nearly all its energy in the first few
    frequencies, so most rounded coefficients are zero.
    """
    c = dct(normalize(A, bounds), axis=1, norm="ortho")
    return np.round(scale * c).astype(int)


def dct_reconstruct(C, bounds, scale) -> Array:
    """Inverse of dct_coefficients, in original units (up to rounding)."""
    low, high = bounds
    u = idct(C / scale, axis=1, norm="ortho")
    return (u + 1) / 2 * (high - low) + low


def flatten_low_first(C) -> list[list[int]]:
    """Each chunk's coefficients: frequency 0 for every dimension, then 1."""
    # (f, dims) row-major; plain ints, since a NumPy int compared with a
    # merged token (a tuple) would broadcast instead of returning False.
    return [c.reshape(-1).tolist() for c in C]


# Byte-pair encoding on sequences of hashable symbols (characters or ints).
# A merged token is the pair itself, so it can be unpacked into symbols.

def bpe_merge(seq, pair, new) -> list:
    """Replace each non-overlapping occurrence of pair, left to right."""
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def bpe_train(corpus, k) -> tuple[list, list]:
    """Up to k merges of the most frequent adjacent pair.

    Returns the merges and the merged corpus. Ties go to the pair counted
    first (Counter keeps insertion order). Stops early when no pair occurs
    twice.
    """
    corpus, merges = [list(s) for s in corpus], []
    for _ in range(k):
        counts = Counter(p for s in corpus for p in zip(s, s[1:]))
        if not counts:
            break
        pair, c = counts.most_common(1)[0]
        if c < 2:
            break
        corpus = [bpe_merge(s, pair, pair) for s in corpus]
        merges.append(pair)
    return merges, corpus


def bpe_encode(seq, merges) -> list:
    """Apply learned merges, in the order they were learned."""
    seq = list(seq)
    for pair in merges:
        seq = bpe_merge(seq, pair, pair)
    return seq


def unpack(token) -> list:
    """The symbols a (possibly merged) token stands for."""
    if isinstance(token, tuple):
        return unpack(token[0]) + unpack(token[1])
    return [token]


# --- Toy 3: chunks, temporal ensembling and a goal that moves ---------------

HZ, T_RUN, T_MOVE = 50, 180, 80     # 50 Hz, a 3.6 s run, goal moves at 1.6 s
GOAL_A, GOAL_B = 1.0, 0.4           # the goal before and after it moves
RATE, OBS_NOISE = 0.06, 0.08        # gap closed per planned step; camera noise


def plan_chunk(x, goal_seen, k) -> Array:
    """k planned positions closing RATE of the remaining gap per step."""
    return goal_seen + (x - goal_seen) * (1 - RATE) ** np.arange(1, k + 1)


def run_chunks(k, mode, rng, m=0.0, noise=OBS_NOISE, t_move=T_MOVE,
               keep=False):
    """Positions x_0..x_T under one execution scheme.

    The robot tracks each action exactly. mode "chunk": query every k
    steps and execute the whole chunk open loop (k = 1 is ordinary
    single-step control). mode "ensemble": query every step and execute
    the weighted average of every chunk that covers this step, weight
    exp(-m i) with i = 0 the oldest prediction, as in ACT. keep=True also
    returns every query's (time, planned chunk).
    """
    x = np.zeros(T_RUN + 1)
    preds = [[] for _ in range(T_RUN + k)]   # step t: predictions, oldest 1st
    plan, plans = None, []
    for t in range(T_RUN):
        goal = GOAL_A if t < t_move else GOAL_B
        if mode == "ensemble" or t % k == 0:
            # Each query sees the goal through a new noisy observation.
            seen = goal + noise * rng.standard_normal()
            plan = plan_chunk(x[t], seen, k)
            plans.append((t, plan))
            for j in range(k):
                preds[t + j].append(plan[j])
        if mode == "ensemble":
            p = np.array(preds[t])
            w = np.exp(-m * np.arange(len(p)))
            x[t + 1] = (w * p).sum() / w.sum()
        else:
            x[t + 1] = plan[t % k]
    return (x, plans) if keep else x


def roughness(x) -> float:
    """RMS second difference of the path: how jerky the motion is."""
    return float(np.sqrt(np.mean(np.diff(x, 2) ** 2)))


def settle_error(x) -> float:
    """RMS distance from the goal over each segment's last 25 steps."""
    a, b = x[T_MOVE - 25:T_MOVE], x[T_RUN - 24:]
    return float(np.sqrt(np.mean(np.r_[a - GOAL_A, b - GOAL_B] ** 2)))


MOVE_TIMES = range(60, 110)     # when the goal may move, for averaging


def half_time(k, mode, m=0.0, t_move=T_MOVE) -> int:
    """Steps after the goal moves until a noise-free run is halfway."""
    x = run_chunks(k, mode, np.random.default_rng(0), m, noise=0.0,
                   t_move=t_move)
    half = (x[t_move] + GOAL_B) / 2              # the new goal is below
    return int(np.argmax(x[t_move:] <= half))


def mean_half_time(k, mode, m=0.0) -> float:
    """half_time averaged over MOVE_TIMES (where in a chunk the move falls)."""
    return float(np.mean([half_time(k, mode, m, t) for t in MOVE_TIMES]))

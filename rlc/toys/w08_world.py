"""Week 8 toys: learned world models, planning in imagination, and JEPA.

Toy 1, a learned simulator. A puck slides on a table with drag and a hill
in the middle. Its true dynamics are hidden from the agent, which only has
transitions recorded while the puck was shoved around at random. A small
MLP fitted to them is the world model. The toy measures three things the
lecture names: errors compound over imagined rollouts; planning with the
cross-entropy method inside the model works when the robot replans from
real observations; and a planner allowed to leave the data finds plans the
model scores well but the real table does not.

Toy 2, predicting embeddings instead of pixels (JEPA). An encoder maps an
observation to an embedding and a predictor maps it, with the action, to
the next embedding. Trained on that loss alone, the encoder collapses: it
maps everything to the same point and the loss goes to zero. Three fixes
from the lecture are compared.
"""

import numpy as np

from rlc import nn

Array = np.ndarray

# --- Toy 1: the puck table --------------------------------------------------

DT, GAIN, DRAG = 0.1, 2.0, 0.6          # step (s), push strength, quadratic drag
HILL_AT, HILL_H, HILL_W = np.array([0.0, 0.0]), 2.5, 0.45
START, GOAL = np.array([-1.5, 0.0]), np.array([1.5, 0.0])
BOX = 2.0                               # data cover positions in [-BOX, BOX]^2
A_MAX = 1.0                             # pushes in the data stay in [-1, 1]^2
SAT = 1.0                               # the motor saturates: stronger pushes do nothing more


def hill_force(p) -> Array:
    """-grad of a Gaussian hill: pushes the puck away from HILL_AT. (..., 2)"""
    d = p - HILL_AT
    g = np.exp(-0.5 * (d**2).sum(-1, keepdims=True) / HILL_W**2)
    return HILL_H * d / HILL_W**2 * g


def true_step(s, a) -> Array:
    """The real table: s = (x, y, vx, vy), a = push (ax, ay). Works on (..., 4)."""
    p, v = s[..., :2], s[..., 2:]
    speed = np.linalg.norm(v, axis=-1, keepdims=True)
    push = GAIN * np.clip(a, -SAT, SAT)
    v = v + DT * (push - DRAG * speed * v + hill_force(p))
    return np.concatenate([p + DT * v, v], axis=-1)


def random_transitions(n, rng, a_max=A_MAX) -> tuple[Array, Array, Array]:
    """n transitions from random states, under random pushes: s, a, s'."""
    s = np.concatenate([rng.uniform(-BOX, BOX, (n, 2)),
                        rng.uniform(-1.5, 1.5, (n, 2))], axis=1)
    a = rng.uniform(-a_max, a_max, (n, 2))
    return s, a, true_step(s, a)


# The world model predicts the change of state from (state, push).
# Inputs and outputs are scaled to unit size, which the MLP prefers.
IN_SCALE = np.array([BOX, BOX, 1.5, 1.5, A_MAX, A_MAX])


def fit_world_model(n, seed, hidden=64, epochs=400) -> tuple[list, Array]:
    """Fit an MLP to n random transitions. Returns params and output scale."""
    rng = np.random.default_rng(seed)
    s, a, s2 = random_transitions(n, rng)
    X = np.concatenate([s, a], axis=1) / IN_SCALE
    D = s2 - s
    out_scale = D.std(axis=0)
    params = nn.init_mlp([6, hidden, hidden, 4], rng)
    params, _ = nn.fit(params, X, D / out_scale, alpha=0.003, epochs=epochs,
                       batch=256, rng=rng)
    return params, out_scale


def model_step(model, s, a) -> Array:
    """The world model's guess of the next state. Works on (n, 4) and (n, 2)."""
    params, out_scale = model
    X = np.concatenate([s, a], axis=-1) / IN_SCALE
    return s + nn.predict(params, X) * out_scale


def rollout(step, s0, actions) -> Array:
    """States s_0..s_H under an action sequence (n, H, 2) from s0 (n, 4)."""
    states = [s0]
    for t in range(actions.shape[1]):
        states.append(step(states[-1], actions[:, t]))
    return np.stack(states, axis=1)                       # (n, H + 1, 4)


def cost(states) -> Array:
    """Distance to the goal at every step, plus the final speed: (n,)."""
    dist = np.linalg.norm(states[:, 1:, :2] - GOAL, axis=-1)
    return dist.sum(axis=1) + 2 * np.linalg.norm(states[:, -1, 2:], axis=-1)


def cem_plan(step, s0, rng, H=15, n=200, elite=20, iters=6, a_max=A_MAX,
             init=None) -> tuple[Array, float]:
    """Cross-entropy method over action sequences in a (true or learned) model.

    Sample n sequences of H pushes from a Gaussian, roll each out in `step`,
    keep the `elite` cheapest, refit the Gaussian, repeat. Returns the final
    mean sequence (H, 2) and its cost according to `step`.
    """
    mu = np.zeros((H, 2)) if init is None else init.copy()
    std = np.full((H, 2), a_max / 2)
    for _ in range(iters):
        A = np.clip(mu + std * rng.standard_normal((n, H, 2)), -a_max, a_max)
        c = cost(rollout(step, np.repeat(s0[None], n, 0), A))
        best = A[np.argsort(c)[:elite]]
        mu, std = best.mean(axis=0), best.std(axis=0) + 0.05
    return mu, float(cost(rollout(step, s0[None], mu[None]))[0])


START_STATE = np.r_[START, 0.0, 0.0]


def run_mpc(plan_step, rng, T=60, H=15, a_max=A_MAX, look=True, start=None):
    """Model predictive control: plan H steps, execute the first push, repeat.

    look=True: every plan starts from the real puck's state. look=False:
    the agent never looks; it replans from the state its own model
    predicted, so the whole run happens in imagination, and the pushes are
    then replayed on the real table. Returns (real path, imagined path),
    each (T + 1, 4); with look=True the two coincide.
    """
    s0 = START_STATE if start is None else start
    real, imag = [s0.copy()], [s0.copy()]
    prev = None
    for _ in range(T):
        here = real[-1] if look else imag[-1]
        # Warm start from the last plan, shifted by one step.
        init = None if prev is None else np.r_[prev[1:], prev[-1:]]
        prev, _ = cem_plan(plan_step, here, rng, H=H, a_max=a_max, init=init)
        a = prev[0]
        real.append(true_step(real[-1], a))
        imag.append(real[-1] if look else plan_step(imag[-1][None], a[None])[0])
    return np.array(real), np.array(imag)


# --- Toy 2: predicting the next embedding (JEPA) ----------------------------

OBS_DIM, EMB_DIM = 16, 4
JEPA_VARIANTS = ("plain", "stop-gradient", "EMA target", "frozen encoder",
                 "covariance regularizer")


def jepa_data(n, rng) -> tuple[Array, Array, Array, Array]:
    """Observations of the puck: a fixed random 16-d linear view of its
    state plus noise. Returns o, a, o' (the next observation) and s."""
    s, a, s2 = random_transitions(n, rng)
    M = np.random.default_rng(99).standard_normal((4, OBS_DIM)) / 2
    noise = lambda: 0.05 * rng.standard_normal((n, OBS_DIM))
    return s @ M + noise(), a, s2 @ M + noise(), s


def jepa_train(variant, seed, steps=3000, n=512, lr=0.01, tau=0.99,
               lam=1.0) -> dict:
    """Train a linear encoder W and predictor (P_z, P_a) on next-embedding error.

    z = o W, prediction = z P_z + a P_a, target z' = o' W_target.
    plain: the target uses W itself and gradients flow into both sides.
    stop-gradient: same W, but the target is treated as a constant.
    EMA target: W_target is a slow moving average of W (no gradient).
    frozen encoder: W stays at its random start; only the predictor learns.
    covariance regularizer: plain, plus lam * ||Cov(z) - I||^2.
    Returns the history of the loss and of the embeddings' spread.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((OBS_DIM, EMB_DIM)) / np.sqrt(OBS_DIM)
    P_z, P_a = np.eye(EMB_DIM), np.zeros((2, EMB_DIM))
    W_tgt = W.copy()
    hist = {"loss": [], "spread": [], "rank": []}
    for _ in range(steps):
        o, a, o2, _ = jepa_data(n, rng)
        z = o @ W                                      # (n, k)
        target = o2 @ (W if variant in ("plain", "stop-gradient",
                                        "covariance regularizer",
                                        "frozen encoder") else W_tgt)
        r = z @ P_z + a @ P_a - target                 # (n, k) residual
        loss = float((r**2).sum(1).mean())
        dr = 2 * r / n
        dP_z, dP_a = z.T @ dr, a.T @ dr
        dz = dr @ P_z.T                                # through the prediction
        dW = o.T @ dz
        if variant in ("plain", "covariance regularizer"):
            dW -= o2.T @ dr                            # through the target too
        if variant == "covariance regularizer":
            zc = z - z.mean(0)
            C = zc.T @ zc / (n - 1)                    # (k, k)
            loss += lam * float(((C - np.eye(EMB_DIM)) ** 2).sum())
            dW += o.T @ (4 * lam * zc @ (C - np.eye(EMB_DIM)) / (n - 1))
        if variant == "frozen encoder":
            dW = 0 * dW
        W, P_z, P_a = W - lr * dW, P_z - lr * dP_z, P_a - lr * dP_a
        W_tgt = tau * W_tgt + (1 - tau) * W            # used by "EMA target"
        hist["loss"].append(loss)
        hist["spread"].append(float(z.std(0).mean()))
        hist["rank"].append(effective_rank(z))
    return {"W": W, **{k: np.array(v) for k, v in hist.items()}}


def effective_rank(z) -> float:
    """How many directions the embeddings really use: (sum l)^2 / sum l^2
    over the eigenvalues l of their covariance; EMB_DIM if all are equal,
    1 if everything lies on one line."""
    lam = np.clip(np.linalg.eigvalsh(np.cov(z.T)), 0, None)
    return float(lam.sum() ** 2 / (lam**2).sum())


def probe_r2(W, seed=7, n=4000) -> float:
    """How much of the puck's position a linear readout of z recovers (R^2)."""
    o, _, _, s = jepa_data(n, np.random.default_rng(seed))
    Z = np.c_[o @ W, np.ones(n)]
    coef, *_ = np.linalg.lstsq(Z, s[:, :2], rcond=None)
    resid = s[:, :2] - Z @ coef
    return float(1 - resid.var(0).sum() / s[:, :2].var(0).sum())


# --- Toy 3: a video model plus an inverse dynamics model --------------------

def random_start(rng) -> Array:
    """A resting puck somewhere on the left of the table."""
    return np.r_[rng.uniform(-1.8, -1.2), rng.uniform(-1.0, 1.0), 0.0, 0.0]


def expert_demos(n_traj, seed, T=60) -> tuple[Array, Array]:
    """Expert runs (MPC in the true dynamics) from random starts.

    Returns states (n_traj, T + 1, 4) and pushes (n_traj, T, 2). The
    pushes are what an action-labelled data set would contain; a video
    shows only the states.
    """
    rng = np.random.default_rng(seed)
    S, A = [], []
    for _ in range(n_traj):
        path, _ = run_mpc(true_step, rng, T=T, start=random_start(rng))
        S.append(path)
        # Recover the push of each step from the true dynamics' inverse:
        # the velocity change, minus drag and hill, divided by the gain.
        p, v, v2 = path[:-1, :2], path[:-1, 2:], path[1:, 2:]
        speed = np.linalg.norm(v, axis=-1, keepdims=True)
        A.append(((v2 - v) / DT + DRAG * speed * v - hill_force(p)) / GAIN)
    return np.array(S), np.array(A)


def fit_mlp(X, Y, seed, hidden=64, epochs=300) -> tuple:
    """An MLP from X to Y with standardized inputs and outputs."""
    rng = np.random.default_rng(seed)
    mx, sx, my, sy = X.mean(0), X.std(0) + 1e-8, Y.mean(0), Y.std(0) + 1e-8
    params = nn.init_mlp([X.shape[1], hidden, hidden, Y.shape[1]], rng)
    params, _ = nn.fit(params, (X - mx) / sx, (Y - my) / sy, alpha=0.003,
                       epochs=epochs, batch=min(256, len(X)), rng=rng)
    return params, mx, sx, my, sy


def mlp_apply(net, X) -> Array:
    params, mx, sx, my, sy = net
    return nn.predict(params, (X - mx) / sx) * sy + my


def fit_vam(S_video, k, seed) -> tuple:
    """Video model on action-free expert videos; IDM on k random pushes.

    The video model predicts the next state from the current one (what
    the expert will do next, seen as motion). The inverse dynamics model
    learns which push turns s into s' from k labelled random transitions,
    which need no expert at all.
    """
    s, s2 = S_video[:, :-1].reshape(-1, 4), S_video[:, 1:].reshape(-1, 4)
    video = fit_mlp(s, s2 - s, seed)
    rs, ra, rs2 = random_transitions(k, np.random.default_rng(seed + 1))
    idm = fit_mlp(np.c_[rs, rs2 - rs], ra, seed)
    return video, idm


def vam_act(vam, s, oracle_next=None) -> Array:
    """Imagine the next state with the video model, then decode the push.
    oracle_next replaces the imagined state with a given one."""
    video, idm = vam
    nxt = s + mlp_apply(video, s) if oracle_next is None else oracle_next
    return np.clip(mlp_apply(idm, np.c_[s, nxt - s]), -A_MAX, A_MAX)


def fit_bc(S, A, k, seed) -> tuple:
    """Behavior cloning on k action-labelled expert transitions."""
    rng = np.random.default_rng(seed)
    s, a = S[:, :-1].reshape(-1, 4), A.reshape(-1, 2)
    idx = rng.choice(len(s), size=k, replace=False)
    return fit_mlp(s[idx], a[idx], seed)


def evaluate(act, seed, n=20, T=60) -> Array:
    """Final distance to the goal from n random starts, for a policy act(s)."""
    rng = np.random.default_rng(seed)
    s = np.array([random_start(rng) for _ in range(n)])
    for _ in range(T):
        s = true_step(s, act(s))
    return np.linalg.norm(s[:, :2] - GOAL, axis=1)

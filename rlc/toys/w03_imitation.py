"""Week 3 toys: imitation learning.

Toy 1, compounding error. A car drives at constant speed along a curved road
and steers. Like a human driver, the expert keeps the heading aligned with the
road but ignores a small offset from the centre line (a deadband); only beyond
DEADBAND does it steer back. Its demonstrations are recorded on a calm day, so
the car never leaves the deadband and the data never shows a steer-back. At
deployment, a rare gust (probability EPS per step) knocks the heading
sideways and pushes the car past the deadband. Behavioral cloning (BC) has
only ever seen "ignore the offset", so it straightens up parallel to the road,
off it, and stays there. DAgger also asks the expert to label the states the
learner itself visits, gusts included, and so learns the steer-back.

Toy 2, multimodal demonstrations: see the second half of this module.
"""

import numpy as np

from rlc import nn

Array = np.ndarray

# --- Toy 1: the road and the car ------------------------------------------

AMP, WAVELEN = 2.0, 40.0      # road centre line y = AMP sin(2π x / WAVELEN)
SPEED = 0.5                   # distance travelled per step
HALF_WIDTH = 1.0              # off the road when |offset| > HALF_WIDTH
U_MAX = 0.3                   # steering limit, radians per step
K_HEADING, K_OFFSET = 0.15, 0.3  # expert's feedback gains
DEADBAND = 0.3                # the expert ignores offsets smaller than this
JITTER = 0.0015               # small steering noise, always present (radians)
GUST = 0.4                    # a gust turns the heading by ±GUST radians
EPS = 0.002                   # chance of a gust per step at deployment
SCALE = np.array([HALF_WIDTH, 0.5, 0.05])   # typical size of each observation


def road_y(x):
    return AMP * np.sin(2 * np.pi * x / WAVELEN)


def road_angle(x):
    """Direction of the road at x (radians), from the slope of road_y."""
    k = 2 * np.pi / WAVELEN
    return np.arctan(AMP * k * np.cos(k * x))


def wrap(a):
    """Angle in (-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# Everything below works on a batch of n cars at once: a state batch is
# (n, 3) with columns (x, y, heading); an observation batch is (n, 3).

def observe(states) -> Array:
    """What the driver sees: (offset from the centre line, heading error,
    how much the road turns over the next step)."""
    x, y, heading = states.T
    return np.stack([y - road_y(x),
                     wrap(heading - road_angle(x)),
                     road_angle(x + SPEED) - road_angle(x)], axis=1)


def expert(obs) -> Array:
    """Follow the road's turn and straighten the heading; steer back towards
    the centre line only for the part of the offset beyond DEADBAND.
    (n, 3) -> (n,)."""
    offset, heading_err, turn = obs.T
    excess = np.sign(offset) * np.maximum(np.abs(offset) - DEADBAND, 0.0)
    u = turn - K_HEADING * heading_err - K_OFFSET * excess
    return np.clip(u, -U_MAX, U_MAX)


def step(states, u, kick) -> Array:
    """Steer by u (limited to ±U_MAX) plus any gust, then move SPEED along
    the new heading. u and kick are (n,)."""
    x, y, heading = states.T
    heading = heading + np.clip(u, -U_MAX, U_MAX) + kick
    x, y = x + SPEED * np.cos(heading), y + SPEED * np.sin(heading)
    return np.stack([x, y, heading], axis=1)


def start_states(x0) -> Array:
    """Cars on the centre line, pointing along the road, at x0 (n,)."""
    x0 = np.asarray(x0, dtype=float)
    return np.stack([x0, road_y(x0), road_angle(x0)], axis=1)


def rollout(policy, T, eps, rng, x0, jitter=JITTER):
    """Drive len(x0) cars for T steps. Each step, each car gets steering
    noise N(0, jitter²) and, with probability eps, a gust of ±GUST.
    Returns states (n, T+1, 3) and observations (n, T, 3)."""
    s = start_states(x0)
    states, obs = [s], []
    for _ in range(T):
        o = observe(s)
        hit = rng.random(len(s)) < eps
        kick = np.where(hit, GUST * rng.choice([-1.0, 1.0], len(s)), 0.0)
        kick = kick + jitter * rng.normal(size=len(s))
        s = step(s, policy(o), kick)
        obs.append(o)
        states.append(s)
    return np.stack(states, axis=1), np.stack(obs, axis=1)


def off_road(states) -> Array:
    """1.0 where a car is off the road; same leading shape as states."""
    x, y = states[..., 0], states[..., 1]
    return (np.abs(y - road_y(x)) > HALF_WIDTH).astype(float)


# --- Toy 1: learners ------------------------------------------------------

def fit_policy(obs, actions, rng, epochs=300, hidden=32):
    """MLP from scaled observations to steering. Returns a batch policy
    (n, 3) -> (n,)."""
    X = obs / SCALE                         # (m, 3), each feature O(1)
    net = nn.init_mlp([3, hidden, hidden, 1], rng)
    params, _ = nn.fit(net, X, actions[:, None], "mse", "tanh",
                       alpha=0.01, epochs=epochs, rng=rng)
    return lambda o: nn.predict(params, o / SCALE)[:, 0]


def expert_data(n_demos, T, rng):
    """Calm-day demonstrations: expert rollouts with jitter but no gusts, from
    random points along the road. The car stays inside the deadband.
    Returns observations (n_demos*T, 3) and the expert's actions."""
    _, obs = rollout(expert, T, 0.0, rng, rng.uniform(0, WAVELEN, n_demos))
    obs = obs.reshape(-1, 3)
    return obs, expert(obs)


def behavior_cloning(n_demos, T, rng):
    obs, act = expert_data(n_demos, T, rng)
    return fit_policy(obs, act, rng)


def dagger(n_demos, T, rng, eps=EPS, iters=4, stride=4, history=False):
    """Start from BC on the calm-day demos; then repeatedly drive the current
    policy at deployment (gusts included), ask the expert what it would have
    done at each visited state, add those (state, expert action) pairs, and
    refit on everything. Only every `stride`-th visited state is kept:
    neighbouring states are nearly identical, and fewer points fit faster.
    history=True returns the policy after every round, BC first."""
    obs, act = expert_data(n_demos, T, rng)
    policies = [fit_policy(obs, act, rng)]
    for _ in range(iters):
        starts = rng.uniform(0, WAVELEN, n_demos)
        _, new = rollout(policies[-1], T, eps, rng, starts)
        new = new[:, ::stride].reshape(-1, 3)    # (n_demos * T/stride, 3)
        obs = np.vstack([obs, new])
        act = np.concatenate([act, expert(new)])  # the expert relabels
        policies.append(fit_policy(obs, act, rng))
    return policies if history else policies[-1]


def first_departure(off) -> Array:
    """Step index of each car's first off-road step, or -1 if it never left.
    off is (n, T) of 0/1."""
    left = off.any(axis=1)
    return np.where(left, off.argmax(axis=1), -1)


def predicted_cost(h, T) -> Array:
    """Expected cumulative off-road steps J(1..T) if a car on the road leaves
    with probability h per step and never comes back: the car is off at step
    t with probability 1 - (1-h)^t, so J(T) = sum of those."""
    t = np.arange(1, T + 1)
    return np.cumsum(1 - (1 - h) ** t)


def experiment1(seeds=16, n_cars=200, T=400, n_demos=10, T_demo=200,
                iters=4):
    """BC and DAgger trained with each seed, then driven by n_cars cars for
    T steps with gusts. Both policies of a seed face the same gusts (same
    random stream). Returns a dict of arrays; J_* are mean cumulative
    off-road steps per car, (seeds, T). The trained policies come back
    too, as lists under "bc" and "dagger"."""
    out = {"J_bc": [], "J_dagger": [], "rounds_left": [], "nogust_left": [],
           "bc_at_risk": 0, "bc_departures": 0, "bc_returned": 0}
    policies = {"bc": [], "dagger": []}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        bc = behavior_cloning(n_demos, T_demo, rng)
        rounds = dagger(n_demos, T_demo, rng, iters=iters, history=True)
        policies["bc"].append(bc)
        policies["dagger"].append(rounds[-1])
        x0 = np.random.default_rng(10_000 + seed).uniform(0, WAVELEN, n_cars)
        gusts = lambda: np.random.default_rng(20_000 + seed)   # same gusts
        for name, policy in (("bc", bc), ("dagger", rounds[-1])):
            s, _ = rollout(policy, T, EPS, gusts(), x0)
            off = off_road(s[:, 1:])                       # (n_cars, T)
            out[f"J_{name}"].append(off.cumsum(axis=1).mean(axis=0))
            if name == "bc":
                first = first_departure(off)
                # Steps each car spent on the road before leaving (or T):
                out["bc_at_risk"] += np.where(first >= 0, first + 1, T).sum()
                out["bc_departures"] += (first >= 0).sum()
                back = (first >= 0) & (off[:, -1] == 0)
                out["bc_returned"] += back.sum()
        # Fraction of cars that ever leave the road, after each round:
        out["rounds_left"].append(
            [off_road(rollout(p, T, EPS, gusts(), x0)[0]).any(axis=1).mean()
             for p in rounds])
        calm = np.random.default_rng(30_000 + seed)
        s, _ = rollout(bc, T, 0.0, calm, x0)
        out["nogust_left"].append(off_road(s).any(axis=1).mean())
    return {**{k: np.array(v) for k, v in out.items()}, **policies}


def single_gust(policies, x0=12.0, t_gust=10, T=60, kick=GUST):
    """Offsets (len(policies), T+1) of cars started at x0 with no jitter and
    one gust of +kick at step t_gust, one car per policy. Positions too."""
    s = start_states(np.full(len(policies), x0))
    states = [s]
    for t in range(T):
        o = observe(s)
        u = np.array([p(o[i:i + 1])[0] for i, p in enumerate(policies)])
        s = step(s, u, np.full(len(policies), kick if t == t_gust else 0.0))
        states.append(s)
    states = np.stack(states, axis=1)                         # (k, T+1, 3)
    return states[..., 1] - road_y(states[..., 0]), states


# --- Toy 2: two ways around an obstacle -----------------------------------
#
# A robot moves right at constant speed from (0, 0) and chooses its sideways
# velocity. A round obstacle sits on its path. Each demonstrator passes it on
# the left (y > 0) or on the right (y < 0); a fraction p choose left. At the
# start every demo looks the same, so the same observation is paired with
# two different actions.

FWD = 0.2                     # forward distance per step
STEPS2 = 50                   # steps per episode: x goes from 0 to 10
OBSTACLE = np.array([5.0, 0.0])
RADIUS = 1.0
LATERAL_MAX = 0.3             # sideways speed limit per step
DETOUR = 1.8                  # how far sideways the expert passes
BINS = np.linspace(-LATERAL_MAX, LATERAL_MAX, 11)   # discretized actions
SCALE2 = np.array([10.0, 2.0])                       # typical size of (x, y)


def detour_y(x, side):
    """The expert's planned path: a smooth bump of height DETOUR around the
    obstacle, on the given side (+1 left, -1 right)."""
    return side * DETOUR * np.exp(-((x - OBSTACLE[0]) / 2.0) ** 2)


def expert2(obs, side) -> Array:
    """Sideways velocity that heads for the planned path one step ahead.
    obs (n, 2) = (x, y); side (n,)."""
    x, y = obs.T
    return np.clip(detour_y(x + FWD, side) - y, -LATERAL_MAX, LATERAL_MAX)


def rollout2(policy, n=1, side=None, start_noise=0.0, jitter=0.0, rng=None):
    """Positions (n, STEPS2+1, 2) and observations (n, STEPS2, 2). The policy
    maps observations (n, 2) to sideways velocities (n,); with `side` given,
    the expert drives instead. start_noise and jitter (sideways, per step)
    spread the demonstrations so the data covers a band around each path."""
    p = np.zeros((n, 2))
    if start_noise:
        p[:, 1] = start_noise * rng.normal(size=n)
    path, obs = [p], []
    for _ in range(STEPS2):
        o = p.copy()
        u = policy(o) if side is None else expert2(o, side)
        dy = np.clip(u, -LATERAL_MAX, LATERAL_MAX)
        if jitter:
            dy = dy + jitter * rng.normal(size=n)
        p = np.stack([p[:, 0] + FWD, p[:, 1] + dy], axis=1)
        obs.append(o)
        path.append(p)
    return np.stack(path, axis=1), np.stack(obs, axis=1)


def hits(path) -> Array:
    """True for each path that comes within RADIUS of the obstacle centre."""
    return (np.linalg.norm(path - OBSTACLE, axis=-1) < RADIUS).any(axis=-1)


def demos2(n, p_left, rng):
    """n demonstrations, a fraction p_left of them passing on the left.
    Returns paths (n, STEPS2+1, 2), observations (n*STEPS2, 2), actions."""
    side = np.where(np.arange(n) < round(p_left * n), 1.0, -1.0)
    path, obs = rollout2(None, n, side=side, start_noise=0.3, jitter=0.05,
                         rng=rng)
    acts = expert2(obs.reshape(-1, 2), np.repeat(side, STEPS2))
    return path, obs.reshape(-1, 2), acts


def fit_mse(obs, acts, rng, epochs=400):
    """Regression: one sideways velocity per observation, MSE loss. Where
    demos disagree, MSE is minimized by their average."""
    net = nn.init_mlp([2, 32, 32, 1], rng)
    params, _ = nn.fit(net, obs / SCALE2, acts[:, None], "mse", "tanh",
                       alpha=0.01, epochs=epochs, rng=rng)
    return lambda o: nn.predict(params, o / SCALE2)[:, 0]


def to_bin(acts) -> Array:
    """Index of the nearest bin centre, (n,) ints."""
    return np.abs(acts[:, None] - BINS[None, :]).argmin(axis=1)


def fit_binned(obs, acts, rng, epochs=400):
    """Classification over BINS with cross-entropy. The network outputs a
    probability for every bin, so two modes stay two peaks. The policy takes
    the most likely bin."""
    net = nn.init_mlp([2, 32, 32, len(BINS)], rng)
    params, _ = nn.fit(net, obs / SCALE2, to_bin(acts), "cross_entropy",
                       "tanh", alpha=0.01, epochs=epochs, rng=rng)
    # (n, 2) -> (n, len(BINS)): one probability per bin, rows sum to 1
    probs = lambda o: nn.softmax(nn.predict(params, o / SCALE2))
    policy = lambda o: BINS[probs(o).argmax(axis=1)]
    return policy, probs


def experiment2(p_left, seed=0, n_demos=20):
    """Demos, and the paths of the MSE and binned policies, for one p_left."""
    rng = np.random.default_rng(seed)
    paths, obs, acts = demos2(n_demos, p_left, rng)
    mse = fit_mse(obs, acts, rng)
    binned, probs = fit_binned(obs, acts, rng)
    mse_path, _ = rollout2(mse)
    bin_path, _ = rollout2(binned)
    return {"demos": paths, "mse": mse_path[0], "binned": bin_path[0],
            "mse_hits": bool(hits(mse_path)[0]),
            "binned_hits": bool(hits(bin_path)[0]),
            "mse_policy": mse, "probs": probs, "obs": obs, "acts": acts}


def hit_rates(p_left, seeds=20, n_demos=20):
    """Fraction of seeds whose MSE / binned policy hits the obstacle."""
    runs = [experiment2(p_left, seed, n_demos) for seed in range(seeds)]
    return (np.mean([r["mse_hits"] for r in runs]),
            np.mean([r["binned_hits"] for r in runs]))

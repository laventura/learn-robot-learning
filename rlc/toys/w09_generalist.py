"""Week 9 toys: one policy for many robots, evaluating policies in a
simulator, and feeding a generalist mixed data.

The shared task. A robot moves a point on a table from a start on the
left to a goal on the right, around a round post in the middle. The
expert's motion in table coordinates is the same for every robot: head
for the goal, and near the post cancel any motion into it and slide
round its side instead. Robots differ in how their motor commands move
the point: x' = x + DT * J a, with a robot-specific matrix J of shape
(2, d). The expert command is the least-norm a with J a = v.

Toy 1, one policy for many robots. Four robots with very different
amounts of data. A specialist per robot, against one network trained on
all of them: separate output heads per robot (CrossFormer style), or one
padded output that has to tell the robots apart from a cue in its input
(pi0 style). Then a fifth robot is added with a few demos, either on top
of the pooled network or from scratch.

Toy 2, evaluating in simulation (SIMPLER's question). A family of
policies is scored on the "real" robot and in simulators with a wrong
motor gain or a shifted camera. Does the simulator rank the policies
the way the real robot does, before and after calibration?

Toy 3, the data pipeline. Padding against sequence packing for samples
with different numbers of camera tokens, why a packed batch needs a
block-diagonal attention mask, and how a small shuffle buffer fills
batches with frames from the same few trajectories.

Worked example, advantage conditioning (pi*0.6). A corridor whose
operators sometimes step back: the exact value function, the advantage
of each step, and what conditioning on "advantage > 0" buys over plain
cloning.
"""

import numpy as np

from rlc import nn

Array = np.ndarray

# --- The shared task ----------------------------------------------------

DT, T_RUN = 0.1, 60                     # step (s), steps per episode
POST, R_POST = np.array([0.0, 0.0]), 0.35
KEEP = R_POST + 0.12                    # the expert keeps this far away
REACH = R_POST + 0.4                    # the post affects motion within this
TOL = 0.1                               # "at the goal" radius


def expert_velocity(x, g) -> Array:
    """The expert's table-frame velocity at x for goal g, both (n, 2).

    Speed 1 far from the goal, slowing within 0.3 of it. Near the post,
    the part of the motion pointing into the post is removed and turned
    into motion round its side, so a path never touches it."""
    d = g - x
    v = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 0.3)
    r = x - POST
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    out = r / rn                                         # unit, away from post
    side = np.concatenate([-out[:, 1:], out[:, :1]], axis=-1)   # tangent
    into = np.maximum(-(v * out).sum(-1, keepdims=True), 0.0)
    # Weight 1 inside KEEP, falling to 0 at REACH
    w = np.clip((REACH - rn) / (REACH - KEEP), 0.0, 1.0)
    # Slide round the side that is closer to the goal direction
    s = np.where((side * v).sum(-1, keepdims=True) >= 0, 1.0, -1.0)
    return v + w * into * (out + s * side)


def sample_tasks(rng, n) -> tuple[Array, Array]:
    """n (start, goal) pairs, each (n, 2)."""
    x = np.stack([rng.normal(-0.8, 0.1, n), rng.uniform(-0.6, 0.6, n)], 1)
    g = np.stack([rng.normal(0.8, 0.1, n), rng.uniform(-0.6, 0.6, n)], 1)
    return x, g


def run_policy(act, J, x, g, steps=T_RUN, gain=1.0, noise=0.0,
               delay=False, rng=None):
    """Roll out act(x, g) -> commands (n, d) on a robot with matrix J.
    gain scales the motors, noise adds N(0, noise^2) to each command,
    and delay=True executes each command one step late (the robot moves
    on the previous command meanwhile). Returns the path (steps + 1, n,
    2), whether each run touched the post, and the step at which it
    first reached its goal (steps if never)."""
    path, touched = [x], np.zeros(len(x), bool)
    t_goal = np.full(len(x), steps)
    prev = None
    for t in range(steps):
        a = np.clip(act(x, g), -2.0, 2.0)                # motor limits
        if delay:
            a, prev = (np.zeros_like(a) if prev is None else prev), a
        if noise:
            a = a + noise * rng.normal(size=a.shape)
        x = x + DT * gain * a @ J.T
        path.append(x)
        touched |= np.linalg.norm(x - POST, axis=1) < R_POST
        new = (np.linalg.norm(x - g, axis=1) < TOL) & (t_goal == steps)
        t_goal[new] = t + 1
    return np.array(path), touched, t_goal


def success(act, J, x, g, **kw) -> float:
    """Share of runs that reach the goal without touching the post."""
    _, touched, t_goal = run_policy(act, J, x, g, **kw)
    return float(((t_goal < T_RUN) & ~touched).mean())


def fit_masked(params, X, Y, M, steps, rng, alpha=0.003, batch=256):
    """Adam on the squared error over the entries where the mask M is 1.

    X (n, d_in), Y and M (n, d_out). Entries with M = 0 are padding or
    another robot's head: they get no gradient, so whatever the network
    outputs there is never trained. Every method gets the same number of
    gradient steps, whatever the size of its data."""
    opt = nn.Adam(params, alpha)
    for _ in range(steps):
        idx = rng.integers(0, len(X), min(batch, len(X)))
        Yh, cache = nn.forward(params, X[idx])
        m = M[idx]
        dY = 2 * m * (Yh - Y[idx]) / len(idx)       # zero where masked
        params = opt.step(params, nn.backward(params, cache, dY))
    return params


# --- Toy 1: one policy for many robots ----------------------------------

def _wheels(angles_deg, gain) -> Array:
    """J for wheels pushing along the given angles: column i is the
    motion that wheel i alone produces. (2, len(angles))"""
    t = np.deg2rad(angles_deg)
    return gain * np.stack([np.cos(t), np.sin(t)])


# name -> J (2, d). Command sizes 2, 2, 3 and 4.
ROBOTS = {
    "point": np.eye(2),
    "turned": 0.8 * np.array([[0.5, -0.866], [0.866, 0.5]]),  # rotated 60°
    "three wheels": _wheels([0, 120, 240], 0.9),
    "four motors": np.array([[1.0, 0.5, 0.0, 0.3], [0.0, 0.4, 1.0, -0.6]]),
}
NEW_ROBOT = _wheels([20, 110, 205], 1.1)      # the robot added afterwards
D_MAX = 4                                      # largest command size
EPISODES = {"point": 40, "turned": 20, "three wheels": 6, "four motors": 3}
STEPS = 4000                                   # gradient steps per training
HIDDEN = 64


def expert_command(J, x, g) -> Array:
    """The least-norm motor command a with J a = expert velocity. (n, d)"""
    return expert_velocity(x, g) @ np.linalg.pinv(J).T


def demos(J, n_episodes, rng, noise=0.05) -> tuple[Array, Array]:
    """Expert episodes on robot J. Returns observations (n, 4) = current
    position and goal, and the expert's commands (n, d). The executed
    commands carry a little noise, so the demos also show how the expert
    corrects small drifts (the noise is not in the labels)."""
    x, g = sample_tasks(rng, n_episodes)
    obs, act = [], []
    for _ in range(T_RUN):
        a = expert_command(J, x, g)
        obs.append(np.c_[x, g])
        act.append(a)
        x = x + DT * (a + noise * rng.normal(size=a.shape)) @ J.T
    return np.concatenate(obs), np.concatenate(act)


def train_specialist(obs, act, seed) -> list:
    """One network for one robot, on that robot's demos alone."""
    rng = np.random.default_rng(seed)
    params = nn.init_mlp([4, HIDDEN, HIDDEN, act.shape[1]], rng)
    return fit_masked(params, obs, act, np.ones_like(act), STEPS, rng)


def head_slices(robots=ROBOTS) -> dict:
    """Each robot's columns in the multi-head output, e.g. point -> 0:2."""
    ends = np.cumsum([J.shape[1] for J in robots.values()])
    return {k: slice(int(e - J.shape[1]), int(e))
            for (k, J), e in zip(robots.items(), ends)}


def train_heads(data, seed) -> list:
    """One network, one output head per robot (CrossFormer style).

    The last layer is linear, so a block of its output columns is a
    separate linear head reading the same hidden features. Each sample
    is masked to its own robot's block: only that head, and the shared
    layers below it, learn from it. data: name -> (obs, act)."""
    cols = head_slices()
    n_out = sum(J.shape[1] for J in ROBOTS.values())
    obs, Y, M = [], [], []
    for name, (o, a) in data.items():
        y, m = np.zeros((len(o), n_out)), np.zeros((len(o), n_out))
        y[:, cols[name]], m[:, cols[name]] = a, 1.0
        obs.append(o), Y.append(y), M.append(m)
    rng = np.random.default_rng(seed)
    params = nn.init_mlp([4, HIDDEN, HIDDEN, n_out], rng)
    return fit_masked(params, np.concatenate(obs), np.concatenate(Y),
                      np.concatenate(M), STEPS, rng)


def cue(name, n, on=True) -> Array:
    """An embodiment cue: a one-hot code for the robot, standing in for
    what tells robots apart in real data (camera view, joint readings).
    on=False gives zeros: the network sees positions only. (n, 4)"""
    c = np.zeros((n, len(ROBOTS)))
    if on:
        c[:, list(ROBOTS).index(name)] = 1.0
    return c


def train_padded(data, seed, cue_on=True) -> list:
    """One network with one padded output of size D_MAX (pi0 style).

    A robot with d < D_MAX motors fills the first d outputs; the rest
    are padding, masked out of the loss. Nothing in the output says
    which robot is driven, so the network must infer it from its input:
    position, goal and, if cue_on, the embodiment cue."""
    obs, Y, M = [], [], []
    for name, (o, a) in data.items():
        y, m = np.zeros((len(o), D_MAX)), np.zeros((len(o), D_MAX))
        y[:, :a.shape[1]], m[:, :a.shape[1]] = a, 1.0
        obs.append(np.c_[o, cue(name, len(o), cue_on)])
        Y.append(y), M.append(m)
    rng = np.random.default_rng(seed)
    params = nn.init_mlp([4 + len(ROBOTS), HIDDEN, HIDDEN, D_MAX], rng)
    return fit_masked(params, np.concatenate(obs), np.concatenate(Y),
                      np.concatenate(M), STEPS, rng)


def act_specialist(params):
    return lambda x, g: nn.predict(params, np.c_[x, g])


def act_head(params, name):
    cols = head_slices()[name]
    return lambda x, g: nn.predict(params, np.c_[x, g])[:, cols]


def act_padded(params, name, cue_on=True):
    d = ROBOTS[name].shape[1]
    return lambda x, g: nn.predict(
        params, np.c_[x, g, cue(name, len(x), cue_on)])[:, :d]


def test_tasks(seed, n=300) -> tuple[Array, Array]:
    """Held-out (start, goal) pairs, the same for every method."""
    return sample_tasks(np.random.default_rng(10_000 + seed), n)


def pooled_data(seed) -> dict:
    """Each robot's demos for one seed: name -> (obs, act)."""
    rng = np.random.default_rng(seed)
    return {k: demos(J, EPISODES[k], rng) for k, J in ROBOTS.items()}


def compare_pooling(seed) -> dict:
    """Success per method and robot, for one draw of demos and inits.
    Keys: (method, robot). Methods: specialist, heads, padded+cue,
    padded-cue."""
    data = pooled_data(seed)
    x, g = test_tasks(seed)
    heads = train_heads(data, seed)
    pad = {on: train_padded(data, seed, on) for on in (True, False)}
    out = {}
    for k, J in ROBOTS.items():
        spec = train_specialist(*data[k], seed)
        out["specialist", k] = success(act_specialist(spec), J, x, g)
        out["heads", k] = success(act_head(heads, k), J, x, g)
        out["padded+cue", k] = success(act_padded(pad[True], k), J, x, g)
        out["padded-cue", k] = success(act_padded(pad[False], k, False),
                                       J, x, g)
    return out


def features(params, obs) -> Array:
    """The last hidden layer of a network: what every head reads. (n, H)"""
    h = obs
    for W, b in params[:-1]:
        h = np.tanh(h @ W + b)
    return h


def fit_new_head(params, obs, act, ridge=1e-3) -> tuple[Array, Array]:
    """A new linear head on frozen pooled features, by ridge regression:
    solves (H'H + ridge I) w = H'a, with a column of ones for the bias.
    Returns (W (H, d), b (d,))."""
    H = np.c_[features(params, obs), np.ones(len(obs))]
    w = np.linalg.solve(H.T @ H + ridge * np.eye(H.shape[1]), H.T @ act)
    return w[:-1], w[-1]


def act_new_head(params, head):
    W, b = head
    return lambda x, g: features(params, np.c_[x, g]) @ W + b


NEW_COUNTS = (1, 2, 4, 8, 16, 32)


def new_robot_curve(seed, counts=NEW_COUNTS) -> dict:
    """Success on NEW_ROBOT after n demos, for each n in counts:
    "scratch" trains a network on them alone, "pooled" fits a new head
    on the features of the multi-head network trained on pooled_data
    (same seed). Keys: (method, n)."""
    heads = train_heads(pooled_data(seed), seed)
    x, g = test_tasks(seed)
    out = {}
    for n in counts:
        obs, act = demos(NEW_ROBOT, n, np.random.default_rng(
            20_000 + 100 * seed + n))
        scratch = train_specialist(obs, act, seed)
        out["scratch", n] = success(act_specialist(scratch), NEW_ROBOT, x, g)
        head = fit_new_head(heads, obs, act)
        out["pooled", n] = success(act_new_head(heads, head), NEW_ROBOT, x, g)
    return out


# --- Toy 2: evaluating in a simulator -----------------------------------
# The "real" robot is the point robot with motor gain 1 and motor noise.
# A simulator has no noise, but its motor gain or its camera may be off.

REAL_NOISE = 0.3                # std of the real motors' noise per command
REAL_DELAY = False              # real commands act one step late
SPEEDS = (0.4, 1.0, 2.2)        # each policy is run at these speed factors
EVAL_DEMOS = (4, 8, 32)         # ... after cloning this many demos


def policy_family(seed=0) -> dict:
    """Policies of mixed quality and style: a clone of the expert from n
    demos (point robot), commanded at a speed factor s. Fast policies
    finish sooner but cut closer to the post. (n, s) -> act."""
    rng = np.random.default_rng(seed)
    family = {}
    for n in EVAL_DEMOS:
        params = train_specialist(*demos(np.eye(2), n, rng), seed + n)
        for s in SPEEDS:
            family[n, s] = (lambda x, g, p=params, s=s:
                            s * nn.predict(p, np.c_[x, g]))
    return family


def eval_tasks(n=300) -> tuple[Array, Array]:
    """The tasks every policy is scored on, real or simulated."""
    return sample_tasks(np.random.default_rng(40_000), n)


def real_success(family) -> dict:
    """Each policy's success on the real robot: gain 1, motor noise and a
    one-step delay (noise from a fixed seed)."""
    x, g = eval_tasks()
    return {k: success(a, np.eye(2), x, g, noise=REAL_NOISE,
                       delay=REAL_DELAY, rng=np.random.default_rng(41_000))
            for k, a in family.items()}


def sim_success(family, gain=1.0, noise=0.0) -> dict:
    """Each policy's success in a simulator with motor gain `gain` and
    motor noise `noise`, and no delay (noise from a fixed seed)."""
    x, g = eval_tasks()
    return {k: success(a, np.eye(2), x, g, gain=gain, noise=noise,
                       rng=np.random.default_rng(42_000))
            for k, a in family.items()}


def sysid(n, rng) -> tuple[float, float]:
    """System identification: hold each of n random commands on the real
    robot for one step, measure the motion dx, and fit dx = DT k a by
    least squares, k = sum(a . dx) / (DT sum(a . a)). The residuals
    give the motor noise: std(dx / DT - k a). Holding each command for
    one step before measuring hides the delay from this test."""
    a = rng.uniform(-1, 1, (n, 2))
    dx = DT * (a + REAL_NOISE * rng.normal(size=a.shape))   # real, gain 1
    k = (a * dx).sum() / (DT * (a * a).sum())
    return float(k), float((dx / DT - k * a).std())


def mmrv(sim, real) -> float:
    """Mean maximum rank violation (SIMPLER): for each policy, the worst
    real-success gap to a policy the simulator orders the other way;
    averaged over policies. 0 means the simulator never misorders a
    pair; a misordered pair costs its real-success gap."""
    s, r = np.asarray(sim), np.asarray(real)
    flip = (s[:, None] < s[None, :]) != (r[:, None] < r[None, :])
    return float((np.abs(r[:, None] - r[None, :]) * flip).max(1).mean())


def pearson(sim, real) -> float:
    return float(np.corrcoef(sim, real)[0, 1])


# --- Toy 3: the data pipeline -------------------------------------------

def padding_cost(lengths) -> tuple[int, int]:
    """Tokens actually used, and tokens computed when every sample in the
    batch is padded to the longest one."""
    lengths = np.asarray(lengths)
    return int(lengths.sum()), int(len(lengths) * lengths.max())


def pack(lengths) -> tuple[Array, Array, Array]:
    """Sequence packing: samples laid end to end in one sequence.

    Returns cu_seqlens, the prefix sums [0, L1, L1 + L2, ...] that mark
    where each sample starts; seg, the sample index of every token; and
    pos, every token's position within its own sample (it restarts at 0
    at each boundary)."""
    lengths = np.asarray(lengths)
    cu = np.r_[0, np.cumsum(lengths)]
    seg = np.repeat(np.arange(len(lengths)), lengths)
    pos = np.arange(cu[-1]) - cu[seg]
    return cu, seg, pos


def attend(q, k, v, allowed) -> Array:
    """Softmax attention, q k v (n, d), allowed (n, n) boolean: query i
    may look at key j only where allowed[i, j]."""
    s = q @ k.T / np.sqrt(q.shape[1])
    s = np.where(allowed, s, -np.inf)
    w = np.exp(s - s.max(1, keepdims=True))
    return (w / w.sum(1, keepdims=True)) @ v


def packing_check(lengths, d=8, seed=0) -> dict:
    """Attention over a packed sequence, with and without the
    block-diagonal mask, against attention over each sample on its own.
    Returns the largest difference from the per-sample result for each."""
    rng = np.random.default_rng(seed)
    cu, seg, _ = pack(lengths)
    q, k, v = (rng.normal(size=(cu[-1], d)) for _ in range(3))
    alone = np.concatenate([attend(q[a:b], k[a:b], v[a:b],
                                   np.ones((b - a, b - a), bool))
                            for a, b in zip(cu[:-1], cu[1:])])
    block = seg[:, None] == seg[None, :]
    full = np.ones_like(block)
    return {"block-diagonal": float(np.abs(attend(q, k, v, block)
                                           - alone).max()),
            "no mask": float(np.abs(attend(q, k, v, full) - alone).max())}


def shuffle_buffer_batches(n_traj, traj_len, buffer, batch, n_batches,
                           rng) -> Array:
    """Stream frames trajectory by trajectory (as a sequential reader
    does) into a buffer of `buffer` frames; each batch takes `batch`
    frames at random from the buffer, and the reader refills it.
    Returns the trajectory index of every frame in every batch,
    (n_batches, batch)."""
    order = rng.permutation(n_traj)          # shards are visited shuffled
    stream = np.repeat(order, traj_len)      # frames arrive in sequence
    buf, head = list(stream[:buffer]), buffer
    out = []
    for _ in range(n_batches):
        pick = rng.choice(len(buf), batch, replace=False)
        out.append([buf[i] for i in pick])
        for i in sorted(pick, reverse=True):   # replace the frames taken
            buf[i] = stream[head % len(stream)]
            head += 1
    return np.array(out)


def distinct_per_batch(batches) -> float:
    """Mean number of different trajectories in a batch."""
    return float(np.mean([len(set(b)) for b in batches]))


# --- Worked example: Gato's tokens for continuous values ----------------

def mu_law(x, mu=100.0, M=256.0) -> Array:
    """Gato's companding, F(x) = sgn(x) log(|x| mu + 1) / log(M mu + 1),
    then clipped to [-1, 1]: small values get spread out, large ones
    squeezed together."""
    x = np.asarray(x, dtype=float)
    f = np.sign(x) * np.log(np.abs(x) * mu + 1) / np.log(M * mu + 1)
    return np.clip(f, -1.0, 1.0)


def gato_tokens(x, bins=1024, text_vocab=32_000) -> Array:
    """Continuous values -> token ids: mu-law, then one of `bins` equal
    bins on [-1, 1] (the top edge goes into the last bin), shifted past
    the text vocabulary so the two never collide."""
    b = np.floor((mu_law(x) + 1) / 2 * bins).astype(int)
    return np.minimum(b, bins - 1) + text_vocab


# --- Worked example: advantage conditioning in a corridor ---------------

def corridor(L, p) -> dict:
    """A corridor of cells 0..L with the goal at L. The operators step
    forward with probability p and back with 1 - p (staying put at 0).
    Reward -1 per step, so the value V(s) is minus the expected number
    of steps to the goal, found by solving the linear equations
    E[s] = 1 + p E[s+1] + (1-p) E[max(s-1, 0)], E[L] = 0.

    Returns V (L+1,), the one-step advantages of the two actions at each
    cell (L,), and the expected steps from cell 0 under the operators'
    own policy, which is what plain cloning reproduces."""
    M, b = np.eye(L + 1), np.zeros(L + 1)
    for s in range(L):
        M[s, s + 1] -= p
        M[s, max(s - 1, 0)] -= 1 - p
        b[s] = 1.0
    E = np.linalg.solve(M, b)
    V = -E
    s = np.arange(L)
    adv_fwd = -1 + V[s + 1] - V[s]
    adv_back = -1 + V[np.maximum(s - 1, 0)] - V[s]
    return {"V": V, "adv_fwd": adv_fwd, "adv_back": adv_back,
            "steps_bc": float(E[0])}

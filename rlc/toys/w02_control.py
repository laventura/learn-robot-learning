"""Week 2 toys: from robot control to Markov decision processes.

Toy 1, a two-link planar arm (a "2R" arm). Forward kinematics, the
configuration space (a torus, drawn as a square with wrap-around edges), how
one round obstacle in the workspace becomes a region of forbidden joint
angles, a grid search for a path around that region, and inverse kinematics
by the lecture's Jacobian-transpose gradient steps, which can land on either
of the two solutions.

Toy 2, one joint lifting a link against gravity. A reference trajectory from
linear interpolation (LERP) or quintic time scaling, tracked by a discrete
P, PI or PID controller running at a fixed control frequency. Shows the
steady sag of P-only control, the torque spike LERP causes, and how a slow
control loop turns the damping term against you.

Toy 3, a corridor MDP. A robot in a row of cells can go left to a small
reward close by or right to a large reward further away; each move can slip
the wrong way. Trajectory probabilities, the expected return J(pi) computed
two ways, and the discount factor at which the better policy flips.

Toy 4, basic random search (Mania, Guy & Recht 2018, Algorithm 1) training a
linear reaching policy with a dense reward (distance to the goal) or a sparse
one (1 if the robot ends at the goal, else 0).
"""

from collections import deque

import numpy as np

Array = np.ndarray

# --- Toy 1: the 2R arm ------------------------------------------------------

L1, L2 = 1.0, 0.8                   # link lengths
OBSTACLE = np.array([0.95, 0.75])    # centre of a round obstacle ...
OBSTACLE_R = 0.25                    # ... and its radius
GRID = 180                           # C-space cells per joint (2 degrees each)


def fk(q) -> tuple[Array, Array]:
    """Forward kinematics: elbow and hand positions for joint angles q.

    q has shape (..., 2): q[..., 0] is the shoulder angle from the x-axis,
    q[..., 1] the elbow angle relative to the first link. Returns two arrays
    of shape (..., 2)."""
    q = np.asarray(q, dtype=float)
    a1, a12 = q[..., 0], q[..., 0] + q[..., 1]
    elbow = np.stack([L1 * np.cos(a1), L1 * np.sin(a1)], axis=-1)
    hand = elbow + np.stack([L2 * np.cos(a12), L2 * np.sin(a12)], axis=-1)
    return elbow, hand


def jacobian(q) -> Array:
    """d(hand)/dq, shape (2, 2): column j is how the hand moves per radian
    of joint j."""
    a1, a12 = q[0], q[0] + q[1]
    return np.array([[-L1 * np.sin(a1) - L2 * np.sin(a12), -L2 * np.sin(a12)],
                     [L1 * np.cos(a1) + L2 * np.cos(a12), L2 * np.cos(a12)]])


def segment_distance(p, a, b) -> Array:
    """Distance from point p (2,) to the segments a-b; a, b shape (..., 2)."""
    ab, ap = b - a, p - a
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    return np.linalg.norm(ap - t[..., None] * ab, axis=-1)


def collides(q, center=OBSTACLE, radius=OBSTACLE_R) -> Array:
    """True where either link passes through the obstacle; q shape (..., 2)."""
    elbow, hand = fk(q)
    base = np.zeros_like(elbow)
    return (segment_distance(center, base, elbow) < radius) | \
           (segment_distance(center, elbow, hand) < radius)


def grid_angles(n=GRID) -> Array:
    """The n cell-centre angles in [-pi, pi)."""
    return -np.pi + (np.arange(n) + 0.5) * 2 * np.pi / n


def c_obstacle(n=GRID) -> Array:
    """Boolean (n, n) grid of forbidden configurations; [i, j] is shoulder
    angle i, elbow angle j."""
    th = grid_angles(n)
    q = np.stack(np.meshgrid(th, th, indexing="ij"), axis=-1)   # (n, n, 2)
    return collides(q)


def wrap(a) -> Array:
    """Angle(s) mapped into [-pi, pi). On the torus, -179 and +179 degrees
    are 2 degrees apart, not 358."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def to_cell(q, n=GRID) -> tuple[int, int]:
    i = np.floor((wrap(np.asarray(q)) + np.pi) / (2 * np.pi) * n).astype(int) % n
    return int(i[0]), int(i[1])


def plan_path(q_start, q_goal, blocked=None, n=GRID) -> Array:
    """Shortest 8-connected path of free cells from q_start to q_goal on the
    wrap-around grid (breadth-first search, unit cost per move), returned
    as unwrapped joint angles (k, 2) so consecutive rows differ by at most
    one cell. This is the 1990s recipe from the lecture: build the C-space,
    then search it."""
    blocked = c_obstacle(n) if blocked is None else blocked
    start, goal = to_cell(q_start, n), to_cell(q_goal, n)
    prev = {start: None}
    queue = deque([start])
    moves = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
    while queue:
        c = queue.popleft()
        if c == goal:
            break
        for di, dj in moves:                     # neighbours wrap around
            nb = ((c[0] + di) % n, (c[1] + dj) % n)
            if nb not in prev and not blocked[nb]:
                prev[nb] = c
                queue.append(nb)
    if goal not in prev:
        raise ValueError("no collision-free path")
    cells = [goal]
    while prev[cells[-1]] is not None:
        cells.append(prev[cells[-1]])
    th = grid_angles(n)
    q = np.array([[th[i], th[j]] for i, j in cells[::-1]])
    return np.unwrap(q, axis=0)                  # no jumps at the seams


def straight_path(q_start, q_goal, k=200) -> Array:
    """The joint-space straight line, along the short way round each joint."""
    d = wrap(np.asarray(q_goal) - np.asarray(q_start))
    return np.asarray(q_start) + np.linspace(0, 1, k)[:, None] * d


def ik_analytic(target) -> Array:
    """Both exact solutions (2, 2) for a reachable hand target: row 0 has the
    elbow bent one way (q2 > 0), row 1 the other way."""
    x, y = target
    c2 = (x * x + y * y - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    if abs(c2) > 1:
        raise ValueError("target out of reach")
    sols = []
    for q2 in (np.arccos(c2), -np.arccos(c2)):
        q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        sols.append([wrap(q1), q2])
    return np.array(sols)


def ik_jacobian_transpose(q0, target, alpha=0.3, iters=200) -> Array:
    """Gradient descent on L(q) = 1/2 ||f(q) - target||^2. The gradient is
    J(q)^T (f(q) - target), so each step is q <- q - alpha J^T e.
    Returns every iterate, shape (iters + 1, 2)."""
    q = np.array(q0, dtype=float)
    path = [q.copy()]
    for _ in range(iters):
        e = fk(q)[1] - target                    # hand error (2,)
        q = q - alpha * jacobian(q).T @ e
        path.append(q.copy())
    return np.array(path)


def homogeneous(theta, tx, ty) -> Array:
    """3x3 SE(2) transform: rotate by theta, then translate by (tx, ty)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]])


# --- Toy 2: one joint against gravity ---------------------------------------

MASS, LENGTH = 1.0, 0.5             # a uniform rod pivoting at one end
INERTIA = MASS * LENGTH ** 2 / 3    # about the pivot
G = 9.81
FRICTION = 0.05                     # viscous joint friction (N m s / rad)
SIM_DT = 1e-3                       # physics step (s)
Q_START, Q_END, MOVE_T = np.deg2rad(-30.0), np.deg2rad(60.0), 1.0
T_TOTAL = 2.0                       # simulate the move, then hold
GAINS = {"P": (30.0, 0.0, 0.0), "PD": (30.0, 0.0, 2.0), "PID": (30.0, 60.0, 2.0)}
HOLD = 0.5                          # the last HOLD seconds measure the steady error


def gravity_torque(q) -> Array:
    """Torque gravity applies at the pivot; q = 0 is horizontal."""
    return -MASS * G * (LENGTH / 2) * np.cos(q)


def lerp(s) -> Array:
    return np.asarray(s, dtype=float)


def quintic(s) -> Array:
    """Quintic time scaling f(s) = 10 s^3 - 15 s^4 + 6 s^5: f(0) = 0,
    f(1) = 1, with zero velocity and acceleration at both ends."""
    s = np.asarray(s, dtype=float)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def reference(t, scaling=quintic) -> Array:
    """Desired joint angle at times t: the move takes MOVE_T, then holds."""
    s = np.clip(np.asarray(t, dtype=float) / MOVE_T, 0.0, 1.0)
    return Q_START + (Q_END - Q_START) * scaling(s)


def track(gains, freq=100.0, scaling=quintic, gravity=True, t_total=T_TOTAL) -> dict:
    """Simulate the joint following reference() under discrete PID control.

    The controller runs every 1/freq s with the lecture's discrete form
    u_k = Kp e_k + Ki sum_j e_j dt + Kd (e_k - e_{k-1}) / dt and holds u
    between updates (zero-order hold); physics runs every SIM_DT. Starts at
    rest at Q_START. Returns per-physics-step arrays t, q, ref, u."""
    kp, ki, kd = gains
    n_sub = int(round(1.0 / (freq * SIM_DT)))   # physics steps per control step
    dt_c = n_sub * SIM_DT
    n = int(round(t_total / SIM_DT))
    q, qd = Q_START, 0.0
    integral, e_prev, u = 0.0, 0.0, 0.0
    ts, qs, refs, us = (np.empty(n) for _ in range(4))
    for k in range(n):
        t = k * SIM_DT
        if k % n_sub == 0:                       # a control update
            e = float(reference(t, scaling)) - q
            integral += e * dt_c
            u = kp * e + ki * integral + kd * (e - e_prev) / dt_c
            e_prev = e
        tau = u - FRICTION * qd + (gravity_torque(q) if gravity else 0.0)
        qd += tau / INERTIA * SIM_DT             # semi-implicit Euler
        q += qd * SIM_DT
        ts[k], qs[k], refs[k], us[k] = t, q, reference(t, scaling), u
        if not np.isfinite(q) or abs(q) > 1e3:   # unstable: stop early
            return {"t": ts[:k + 1], "q": qs[:k + 1], "ref": refs[:k + 1],
                    "u": us[:k + 1], "unstable": True}
    return {"t": ts, "q": qs, "ref": refs, "u": us, "unstable": False}


def summary(run) -> dict:
    """Peak |error| during the move; the mean error and the error's range
    (max - min) over the final HOLD seconds, which show a steady sag and a
    lasting oscillation; peak |torque|; and the largest torque change
    between consecutive physics steps."""
    t, err = run["t"], run["ref"] - run["q"]
    if run["unstable"]:
        hold_mean = hold_swing = np.inf
    else:
        tail = err[t >= t[-1] - HOLD]
        hold_mean, hold_swing = tail.mean(), tail.max() - tail.min()
    return {"peak_err": float(np.abs(err[t <= MOVE_T]).max()),
            "hold_mean": float(hold_mean), "hold_swing": float(hold_swing),
            "peak_u": float(np.abs(run["u"]).max()),
            "max_du": float(np.abs(np.diff(run["u"])).max())}


def p_only_sag(kp=GAINS["P"][0], q=Q_END) -> float:
    """Steady error of proportional control holding target angle q (any
    controller without an integral term settles here once the motion stops).

    At rest the motor torque Kp e must cancel gravity at the angle the joint
    actually reaches, q - e: Kp e = m g (l/2) cos(q - e). Solved by
    fixed-point iteration; the first guess, m g (l/2) cos(q) / Kp, ignores
    that the sagging link's lever arm grows."""
    e = 0.0
    for _ in range(50):
        e = float(-gravity_torque(q - e) / kp)
    return e


# --- Toy 3: the corridor MDP -------------------------------------------------

N_CELLS = 9                  # cells 0 .. 8; 0 and 8 end the episode
START_CELL = 2
R_NEAR, R_FAR = 1.0, 10.0    # reward for entering cell 0 / cell 8
LEFT, RIGHT = -1, +1


def step_dist(s, a, slip) -> list[tuple[int, float, float]]:
    """Transition P(s' | s, a) with rewards: (s', probability, reward) for
    each outcome. The robot moves the way it chose with probability 1 - slip
    and the other way with probability slip."""
    out = []
    for move, p in ((a, 1.0 - slip), (-a, slip)):
        if p == 0:
            continue
        s2 = s + move
        r = R_NEAR if s2 == 0 else R_FAR if s2 == N_CELLS - 1 else 0.0
        out.append((s2, p, r))
    return out


def terminal(s) -> bool:
    return s == 0 or s == N_CELLS - 1


def enumerate_trajectories(policy, slip, horizon, s0=START_CELL):
    """Every trajectory of up to `horizon` steps with its probability
    Pr(tau) = p(s0) prod_t pi(a_t | s_t) P(s_t+1 | s_t, a_t) and its rewards.

    policy maps a cell to an action (deterministic, so pi(a|s) = 1 for that
    action). Yields (states, rewards, prob); an episode stops early on
    reaching an end cell."""
    stack = [([s0], [], 1.0)]
    while stack:
        states, rewards, prob = stack.pop()
        s = states[-1]
        if terminal(s) or len(rewards) == horizon:
            yield states, rewards, prob
            continue
        for s2, p, r in step_dist(s, policy(s), slip):
            stack.append((states + [s2], rewards + [r], prob * p))


def discounted_return(rewards, gamma) -> float:
    """G = sum_t gamma^t r_t."""
    return float(sum(r * gamma ** t for t, r in enumerate(rewards)))


def expected_return_enum(policy, slip, gamma, horizon) -> float:
    """J(pi) = sum over trajectories of Pr(tau) G(tau), by brute force."""
    return sum(p * discounted_return(r, gamma)
               for _, r, p in enumerate_trajectories(policy, slip, horizon))


def expected_return(policy, slip, gamma, horizon=2000) -> float:
    """The same sum, organised by time: carry the probability of being in
    each cell at step t forward, and add the expected reward of step t.
    Equal to expected_return_enum, but linear in the horizon."""
    dist = np.zeros(N_CELLS)
    dist[START_CELL] = 1.0
    total = 0.0
    for t in range(horizon):
        new = np.zeros(N_CELLS)
        for s in range(1, N_CELLS - 1):    # episodes in end cells stopped
            if dist[s] == 0:
                continue
            for s2, p, r in step_dist(s, policy(s), slip):
                new[s2] += dist[s] * p
                total += gamma ** t * dist[s] * p * r
        dist = new
        dist[0] = dist[N_CELLS - 1] = 0.0
    return float(total)


def monte_carlo_return(policy, slip, gamma, n, rng, horizon=2000) -> Array:
    """Discounted returns of n sampled episodes."""
    out = np.empty(n)
    for i in range(n):
        s, g, t = START_CELL, 0.0, 0
        while not terminal(s) and t < horizon:
            a = policy(s)
            move = a if rng.random() >= slip else -a
            s += move
            r = R_NEAR if s == 0 else R_FAR if s == N_CELLS - 1 else 0.0
            g += gamma ** t * r
            t += 1
        out[i] = g
    return out


def go_left(s) -> int:
    return LEFT


def go_right(s) -> int:
    return RIGHT


def flip_gamma(slip, lo=0.01, hi=0.999, tol=1e-6) -> float:
    """The discount factor above which going right beats going left, found
    by bisection on J(right) - J(left)."""
    def gap(g):
        return (expected_return(go_right, slip, g)
                - expected_return(go_left, slip, g))
    if gap(hi) <= 0:
        return np.nan
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) <= 0 else (lo, mid)
    return (lo + hi) / 2


def flip_gamma_deterministic() -> float:
    """Without slips: left pays R_NEAR after d_near steps, right pays R_FAR
    after d_far steps, so right wins when gamma^(d_far - d_near) >
    R_NEAR / R_FAR."""
    d_near, d_far = START_CELL, N_CELLS - 1 - START_CELL
    return float((R_NEAR / R_FAR) ** (1.0 / (d_far - d_near)))


# --- Toy 4: random search, dense vs sparse reward -----------------------------

REACH_H = 30            # steps per episode
REACH_AMAX = 0.1        # each action component is clipped to +-REACH_AMAX
REACH_NOISE = 0.01      # motor noise per step
REACH_TOL = 0.05        # "at the goal": final distance below this


def reach_episode(M, rng, sparse) -> tuple[float, float]:
    """One episode of the linear policy a = clip(M (g - x)).

    The robot starts on the unit circle around the goal g = 0, at a random
    angle. Returns (reward, final distance): the dense reward is minus the
    mean distance to the goal over the episode; the sparse reward is 1 if
    the final distance is below REACH_TOL, else 0."""
    ang = rng.uniform(0, 2 * np.pi)
    x = np.array([np.cos(ang), np.sin(ang)])
    dists = []
    for _ in range(REACH_H):
        a = np.clip(M @ (-x), -REACH_AMAX, REACH_AMAX)   # observation: g - x
        x = x + a + REACH_NOISE * rng.standard_normal(2)
        dists.append(np.linalg.norm(x))
    final = dists[-1]
    reward = float(final < REACH_TOL) if sparse else -float(np.mean(dists))
    return reward, final


def success_rate(M, rng, n=50) -> float:
    """Share of n fresh episodes that end within REACH_TOL of the goal."""
    finals = [reach_episode(M, rng, sparse=True)[1] for _ in range(n)]
    return float(np.mean(np.array(finals) < REACH_TOL))


def basic_random_search(sparse, rng, iters=80, n_dirs=8, nu=0.05,
                        alpha=0.5) -> dict:
    """Basic random search (Mania et al. 2018, Algorithm 1) on M (2, 2).

    Each iteration samples n_dirs Gaussian directions delta, runs one
    episode at M + nu delta and one at M - nu delta, and moves M by
    alpha / n_dirs * sum (r+ - r-) delta. A pair whose two episodes score
    the same adds nothing; with a sparse reward that is almost every pair
    until some nudge happens to reach the goal.

    Returns "success": success_rate of M after each iteration, shape
    (iters + 1,), and "informative": the share of pairs with r+ != r-
    in each iteration, shape (iters,)."""
    M = np.zeros((2, 2))                   # starts by not moving at all
    curve, informative = [success_rate(M, rng)], []
    for _ in range(iters):
        deltas = rng.standard_normal((n_dirs, 2, 2))   # (N, 2, 2)
        step, n_info = np.zeros((2, 2)), 0
        for d in deltas:
            r_plus = reach_episode(M + nu * d, rng, sparse)[0]
            r_minus = reach_episode(M - nu * d, rng, sparse)[0]
            # (r+ - r-) / (2 nu) estimates the slope of the reward along d;
            # the constant 1 / (2 nu) is folded into alpha.
            step += (r_plus - r_minus) * d
            n_info += r_plus != r_minus
        M = M + alpha / n_dirs * step
        curve.append(success_rate(M, rng))
        informative.append(n_info / n_dirs)
    return {"success": np.array(curve), "informative": np.array(informative)}


def random_search_study(n_seeds=20, **kw) -> dict:
    """basic_random_search for both rewards over seeds 0 .. n_seeds - 1.
    Returns, per reward, success curves (n_seeds, iters + 1) and
    informative shares (n_seeds, iters)."""
    out = {}
    for name, sparse in (("dense", False), ("sparse", True)):
        runs = [basic_random_search(sparse, np.random.default_rng(s), **kw)
                for s in range(n_seeds)]
        out[name] = {"success": np.array([r["success"] for r in runs]),
                     "informative": np.array([r["informative"] for r in runs])}
    return out


def first_solved(success, level=0.9) -> Array:
    """Per seed, the first iteration whose success rate reaches `level`
    (np.nan if never); success has shape (n_seeds, iters + 1)."""
    hit = success >= level
    return np.where(hit.any(axis=1), hit.argmax(axis=1), np.nan)

"""Week 4 toys: value functions, Q-learning, and acting in continuous spaces.

Toy 1, smoothing by expectation. A car drives along a cliff edge: reward +1
on the road, -1 over the edge, a step function of the car's sideways
position. A Gaussian policy over that position turns the step into a smooth
expected reward.

Toy 2, dynamic programming on a slippery warehouse grid (values, value
iteration, policy iteration). The page shows results only: implementing
these algorithms is the course's homework.

Toy 3, Q-learning versus SARSA on a cliff walk, the lecture's example of
off-policy versus on-policy learning.

Toy 4, overestimation: the maximum of noisy value estimates is biased
upwards; a second, independent estimate removes the bias (double Q-learning).

Toy 5, finding argmax_a Q(s, a) over a 7-dimensional continuous action:
uniform sampling, the cross-entropy method, and gradient ascent.
"""

import numpy as np
from scipy.stats import norm

Array = np.ndarray

# --- Toy 1: smoothing a step reward ----------------------------------------------

EDGE = 1.0                       # the cliff edge, in sideways position


def step_reward(y) -> Array:
    """+1 on the road (y < EDGE), -1 over the edge."""
    return np.where(np.asarray(y) < EDGE, 1.0, -1.0)


def expected_reward(mu, sigma) -> Array:
    """E[r(y)] for y ~ N(mu, sigma^2): P(on road) - P(off) = 2 Phi((EDGE-mu)/sigma) - 1."""
    return 2 * norm.cdf((EDGE - np.asarray(mu)) / sigma) - 1


def expected_reward_grad(mu, sigma) -> Array:
    """d/dmu of expected_reward: nonzero everywhere, largest near the edge."""
    return -2 / sigma * norm.pdf((EDGE - np.asarray(mu)) / sigma)


# --- Toy 2: the warehouse grid -----------------------------------------------------

# The short route from S to G runs along row 2, right beside a row of
# hazards; the long route goes round the top, beside walls, which are safe
# to bump into. Which route is optimal depends on how slippery the floor is.
LAYOUT = [".......",
          ".####..",
          "S.....G",
          ".XXXX..",
          "......."]
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]       # up, right, down, left
ARROWS = "↑→↓←"
GOAL_R, HAZARD_R = 1.0, -1.0


def grid_cells():
    """Free cells (row, col) in reading order, and lookups."""
    cells = [(r, c) for r, row in enumerate(LAYOUT) for c, ch in enumerate(row) if ch != "#"]
    index = {rc: i for i, rc in enumerate(cells)}
    return cells, index


CELLS, INDEX = grid_cells()
N_S, N_A = len(CELLS), len(ACTIONS)
START = INDEX[next((r, c) for r, row in enumerate(LAYOUT) for c, ch in enumerate(row) if ch == "S")]
GOAL = INDEX[next((r, c) for r, row in enumerate(LAYOUT) for c, ch in enumerate(row) if ch == "G")]
HAZARDS = {INDEX[(r, c)] for r, row in enumerate(LAYOUT) for c, ch in enumerate(row) if ch == "X"}
TERMINAL = {GOAL} | HAZARDS


def move(s, a) -> int:
    """Deterministic move; bumping into a wall or the border stays put."""
    r, c = CELLS[s]
    nr, nc = r + ACTIONS[a][0], c + ACTIONS[a][1]
    return INDEX.get((nr, nc), s)


def transition_model(slip=0.2) -> tuple[Array, Array]:
    """P (S, A, S') and expected reward R (S, A).

    The intended move happens with probability 1 - slip; each of the two
    perpendicular moves with slip / 2 (the lecture's 80/10/10 for slip 0.2).
    Entering the goal pays +1, a hazard -1; both end the episode, so
    terminal states transition to themselves with reward 0."""
    P = np.zeros((N_S, N_A, N_S))
    R = np.zeros((N_S, N_A))
    for s in range(N_S):
        for a in range(N_A):
            if s in TERMINAL:
                P[s, a, s] = 1.0
                continue
            for a2, p in ((a, 1 - slip), ((a + 1) % 4, slip / 2), ((a + 3) % 4, slip / 2)):
                s2 = move(s, a2)
                P[s, a, s2] += p
                R[s, a] += p * (GOAL_R if s2 == GOAL else HAZARD_R if s2 in HAZARDS else 0.0)
    return P, R


def value_iteration(P, R, gamma=0.95, tol=1e-8, max_sweeps=10_000) -> list[Array]:
    """Every value function V_0 = 0, V_1, ... of value iteration, until the
    largest change in a sweep is below tol."""
    V = np.zeros(N_S)
    history = [V]
    for _ in range(max_sweeps):
        V_new = (R + gamma * P @ V).max(axis=1)
        history.append(V_new)
        if np.abs(V_new - V).max() < tol:
            break
        V = V_new
    return history


def greedy(P, R, V, gamma=0.95) -> Array:
    """One-step lookahead policy (ties broken towards the lowest action index)."""
    return (R + gamma * P @ V).argmax(axis=1)


def evaluate(P, R, policy, gamma=0.95) -> Array:
    """Exact V^pi by solving the linear Bellman equation (I - gamma P_pi) V = R_pi."""
    idx = np.arange(N_S)
    P_pi, R_pi = P[idx, policy], R[idx, policy]
    return np.linalg.solve(np.eye(N_S) - gamma * P_pi, R_pi)


def policy_iteration(P, R, gamma=0.95, policy0=None) -> list[tuple[Array, Array]]:
    """(policy, V^policy) for every iteration until the policy stops changing."""
    policy = np.zeros(N_S, dtype=int) if policy0 is None else policy0.copy()
    history = []
    while True:
        V = evaluate(P, R, policy, gamma)
        history.append((policy.copy(), V))
        new = greedy(P, R, V, gamma)
        # Keep the old action when it is (numerically) as good: avoids cycling on ties.
        Q = R + gamma * P @ V
        keep = Q[np.arange(N_S), policy] >= Q.max(axis=1) - 1e-12
        new[keep] = policy[keep]
        if np.array_equal(new, policy):
            return history
        policy = new


def rollout_grid(policy, slip, rng, s0=START, max_steps=100) -> list[int]:
    """Cells visited by one episode following `policy` with slips."""
    path = [s0]
    s = s0
    for _ in range(max_steps):
        if s in TERMINAL:
            break
        a = policy[s]
        u = rng.random()
        a2 = a if u < 1 - slip else (a + 1) % 4 if u < 1 - slip / 2 else (a + 3) % 4
        s = move(s, a2)
        path.append(s)
    return path


# --- Toy 3: cliff walking, Q-learning versus SARSA ------------------------------------

CLIFF_ROWS, CLIFF_COLS = 4, 10
CLIFF_START, CLIFF_GOAL = (3, 0), (3, CLIFF_COLS - 1)


def cliff_step(pos, a) -> tuple[tuple[int, int], float, bool]:
    """One move on the cliff grid: -1 per step; stepping into the cliff (the
    bottom row between start and goal) costs -100 and sends the agent back
    to the start; reaching the goal ends the episode."""
    r = min(max(pos[0] + ACTIONS[a][0], 0), CLIFF_ROWS - 1)
    c = min(max(pos[1] + ACTIONS[a][1], 0), CLIFF_COLS - 1)
    if r == CLIFF_ROWS - 1 and 0 < c < CLIFF_COLS - 1:
        return CLIFF_START, -100.0, False
    return (r, c), -1.0, (r, c) == CLIFF_GOAL


def eps_greedy(q_row, eps, rng) -> int:
    """Random action with probability eps, else the best (random tie-break)."""
    if rng.random() < eps:
        return int(rng.integers(4))
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def td_control(method, rng, episodes=500, eps=0.1, alpha=0.5, gamma=1.0):
    """Tabular Q-learning ("q") or SARSA ("sarsa") on the cliff grid.

    Both act epsilon-greedily. Q-learning's target uses the best next
    action (off-policy); SARSA's uses the next action it will actually take
    (on-policy). Returns the Q-table (rows, cols, 4) and each episode's
    return, shape (episodes,)."""
    Q = np.zeros((CLIFF_ROWS, CLIFF_COLS, 4))
    returns = np.empty(episodes)
    for ep in range(episodes):
        pos, total = CLIFF_START, 0.0
        a = eps_greedy(Q[pos], eps, rng)
        for _ in range(1000):                       # cap for very early episodes
            nxt, r, done = cliff_step(pos, a)
            total += r
            a2 = eps_greedy(Q[nxt], eps, rng)
            if done:
                target = r                          # no value beyond the goal
            elif method == "q":
                target = r + gamma * Q[nxt].max()   # best next action
            else:
                target = r + gamma * Q[nxt][a2]     # the action SARSA will take
            Q[pos][a] += alpha * (target - Q[pos][a])
            pos, a = nxt, a2
            if done:
                break
        returns[ep] = total
    return Q, returns


def greedy_path(Q, max_steps=60) -> list[tuple[int, int]]:
    """Cells visited by the greedy policy from the start (no exploration)."""
    pos, path = CLIFF_START, [CLIFF_START]
    for _ in range(max_steps):
        pos, _, done = cliff_step(pos, int(Q[pos].argmax()))
        path.append(pos)
        if done:
            break
    return path


# --- Toy 4: overestimation -------------------------------------------------------

def max_bias(n_actions, sigma, rng, trials=20_000) -> tuple[float, float]:
    """True Q-values are all 0. Each action's estimate is 0 plus N(0, sigma^2)
    noise. Returns the mean of (single) the max of one set of estimates, and
    (double) the estimate from a second, independent set at the first set's
    argmax. The true max is 0, so these are the biases."""
    A = sigma * rng.standard_normal((trials, n_actions))
    B = sigma * rng.standard_normal((trials, n_actions))
    single = A.max(axis=1)
    double = B[np.arange(trials), A.argmax(axis=1)]
    return float(single.mean()), float(double.mean())


def expected_max_normal(n, grid=4001) -> float:
    """E[max of n standard normals], by numerical integration of
    n x phi(x) Phi(x)^(n-1)."""
    x = np.linspace(-8, 8, grid)
    return float(np.trapezoid(x * n * norm.pdf(x) * norm.cdf(x) ** (n - 1), x))


# --- Toy 5: argmax over a continuous action ------------------------------------------

DIM = 7
PEAK = np.full(DIM, 0.5)                 # the global maximum, Q = 1
DECOY = np.full(DIM, -0.5)               # a lower local maximum, Q = 0.7
WIDTH = 0.7                              # broad enough that gradients reach most starts


def q_landscape(a) -> Array:
    """A smooth Q(s, a) for one fixed state; a has shape (..., DIM) in [-1, 1]."""
    a = np.asarray(a)
    d1 = ((a - PEAK) ** 2).sum(-1)
    d2 = ((a - DECOY) ** 2).sum(-1)
    return np.exp(-d1 / (2 * WIDTH ** 2)) + 0.7 * np.exp(-d2 / (2 * WIDTH ** 2))


def q_grad(a) -> Array:
    """dQ/da, the direction DDPG's actor follows."""
    a = np.asarray(a)
    g1 = np.exp(-((a - PEAK) ** 2).sum(-1) / (2 * WIDTH ** 2))[..., None] * -(a - PEAK) / WIDTH ** 2
    g2 = 0.7 * np.exp(-((a - DECOY) ** 2).sum(-1) / (2 * WIDTH ** 2))[..., None] * -(a - DECOY) / WIDTH ** 2
    return g1 + g2


def random_search_argmax(n, rng) -> float:
    """Best Q among n uniform samples."""
    return float(q_landscape(rng.uniform(-1, 1, (n, DIM))).max())


def cem_argmax(rng, n=64, elite=8, iters=10, init_std=0.6):
    """Cross-entropy method: sample n actions from a Gaussian, keep the
    `elite` best by Q, refit the mean and std to them, repeat.
    Returns the final mean action and its Q; costs n * iters evaluations."""
    mu, std = np.zeros(DIM), np.full(DIM, init_std)
    for _ in range(iters):
        a = np.clip(mu + std * rng.standard_normal((n, DIM)), -1, 1)   # (n, DIM)
        best = a[np.argsort(q_landscape(a))[-elite:]]                    # (elite, DIM)
        mu, std = best.mean(axis=0), best.std(axis=0) + 1e-3
    return mu, float(q_landscape(mu))


def gradient_argmax(a0, lr=0.05, steps=300) -> tuple[Array, float]:
    """Gradient ascent on Q from a0, the inner loop of DDPG's actor update."""
    a = np.array(a0, dtype=float)
    for _ in range(steps):
        a = np.clip(a + lr * q_grad(a), -1, 1)
    return a, float(q_landscape(a))

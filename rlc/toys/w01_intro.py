"""Week 1 toys: why robots need feedback, and robot actions as tokens.

Toy 1, sense-plan-act vs closed loop. A robot in a room must pass through a
doorway to reach a goal. The sense-plan-act robot (like Shakey) estimates its
position once, plans a path from that estimate, and executes it blind. The
closed-loop robot re-estimates its position at every step and heads for the
next waypoint from wherever it is. Both see the same noisy position
estimates and the same small motor noise.

Toy 2, actions as tokens. A smooth reaching motion is turned into discrete
tokens the way RT-2 does it: each action dimension is cut into B equal bins
and each step's action becomes the index of its bin. Replaying the tokens
blindly accumulates the rounding error; a robot that re-observes where it is
and tokenizes the correction keeps the error within one bin.

The reading-time arithmetic behind the lecture's data-scale comparison is at
the end.
"""

import numpy as np

Array = np.ndarray

# --- Toy 1: the room ------------------------------------------------------

START = np.array([0.0, -1.5])
GOAL = np.array([8.0, 1.5])
WALL_X = 5.0                  # a wall runs along x = WALL_X ...
DOOR_Y, DOOR_HALF = 0.5, 0.4  # ... with a doorway |y - DOOR_Y| < DOOR_HALF
STEP = 0.2                    # distance moved per step
GOAL_TOL = 0.3                # success: within this distance of the goal
T_MAX = 120                   # steps allowed
MOTOR_NOISE = 0.02            # each step lands off by N(0, MOTOR_NOISE^2)
# Waypoints: just before the door, just after it, then the goal.
WAYPOINTS = np.array([[WALL_X - 0.5, DOOR_Y], [WALL_X + 0.5, DOOR_Y], GOAL])


def crashes(p, q) -> bool:
    """True if the move from p to q crosses the wall outside the doorway."""
    if (p[0] - WALL_X) * (q[0] - WALL_X) > 0 or p[0] == q[0]:
        return False                              # did not cross x = WALL_X
    y = p[1] + (q[1] - p[1]) * (WALL_X - p[0]) / (q[0] - p[0])
    return abs(y - DOOR_Y) >= DOOR_HALF


def plan(start_estimate) -> Array:
    """Sense-plan-act: the whole sequence of moves (n, 2) along the
    waypoint path, computed once from the estimated start."""
    moves, p = [], np.asarray(start_estimate, dtype=float)
    for w in WAYPOINTS:
        d = w - p
        n = int(np.ceil(np.linalg.norm(d) / STEP))
        moves += [d / n] * n                      # n equal moves to w
        p = w
    return np.array(moves)


def next_waypoint(p_est) -> Array:
    """Closed loop: the first waypoint the robot believes it hasn't passed."""
    for w in WAYPOINTS[:2]:
        if p_est[0] < w[0]:
            return w
    return GOAL


def run(mode, sigma, rng):
    """One episode. mode is "open" (plan once from a noisy estimate of the
    start) or "closed" (a fresh noisy estimate every step).
    Returns (path (k, 2), outcome), outcome "goal", "crash" or "timeout"."""
    p = START.copy()
    path = [p]
    if mode == "open":
        moves = plan(START + sigma * rng.normal(size=2))
    for t in range(T_MAX):
        if mode == "open":
            if t >= len(moves):
                break                             # plan finished
            move = moves[t]
        else:
            est = p + sigma * rng.normal(size=2)  # sense
            d = next_waypoint(est) - est          # plan one step
            move = d * min(1.0, STEP / max(np.linalg.norm(d), 1e-9))
        q = p + move + MOTOR_NOISE * rng.normal(size=2)   # act
        if crashes(p, q):
            return np.array(path + [q]), "crash"
        p = q
        path.append(p)
        if mode == "closed" and np.linalg.norm(p - GOAL) < GOAL_TOL:
            break
    reached = np.linalg.norm(p - GOAL) < GOAL_TOL
    return np.array(path), "goal" if reached else "timeout"


def success_rate(mode, sigma, n=400, seed=0) -> dict:
    """Share of n episodes ending at the goal, in a crash, or elsewhere."""
    rng = np.random.default_rng(seed)
    outcomes = [run(mode, sigma, rng)[1] for _ in range(n)]
    return {k: outcomes.count(k) / n for k in ("goal", "crash", "timeout")}


# --- Toy 2: actions as tokens ---------------------------------------------

def reach_path(T=100) -> Array:
    """A smooth 2-D end-effector path (T+1, 2) in metres: a 0.6 m reach
    that speeds up and slows down (x), with a sideways arc that leans
    towards the start (y). Deliberately not symmetric in time, so rounding
    errors don't cancel by symmetry."""
    u = np.linspace(0.0, 1.0, T + 1)
    x = 0.6 * (10 * u**3 - 15 * u**4 + 6 * u**5)    # minimum-jerk profile
    y = 0.3 * np.sin(np.pi * u) * (1 - 0.5 * u) + 0.1 * u
    return np.stack([x, y], axis=1)


def action_range(moves, margin=0.1):
    """Per-dimension token range: the demonstrated moves' min and max,
    widened by `margin` of their span on each side, so that a small
    correction just outside the demonstrated range can still be expressed."""
    lo, hi = moves.min(axis=0), moves.max(axis=0)
    pad = margin * (hi - lo)
    return lo - pad, hi + pad


def tokenize(a, lo, hi, bins) -> Array:
    """Index of the bin each value falls in: [lo, hi] cut into `bins` equal
    bins; values outside are clipped to the end bins. Same shape as a."""
    k = np.floor((np.asarray(a) - lo) / (hi - lo) * bins).astype(int)
    return np.clip(k, 0, bins - 1)


def detokenize(k, lo, hi, bins) -> Array:
    """The centre of bin k: the value a token stands for."""
    return lo + (np.asarray(k) + 0.5) * (hi - lo) / bins


def replay(path, bins, closed_loop=False):
    """Execute the path's moves after a round trip through tokens.
    Open loop: tokenize the demonstrated moves and add them up.
    Closed loop: at each step tokenize the move from where the robot
    actually is to the next point on the path.
    Returns the executed path (T+1, 2)."""
    moves = np.diff(path, axis=0)                   # (T, 2) actions
    lo, hi = action_range(moves)                    # per-dimension range
    p, out = path[0].copy(), [path[0].copy()]
    for t in range(len(moves)):
        want = path[t + 1] - p if closed_loop else moves[t]
        p = p + detokenize(tokenize(want, lo, hi, bins), lo, hi, bins)
        out.append(p.copy())
    return np.array(out)


def replay_error(path, bins, closed_loop=False) -> float:
    """Largest distance between the replayed and the demonstrated path."""
    gap = replay(path, bins, closed_loop) - path
    return float(np.linalg.norm(gap, axis=1).max())


# --- Reading-time arithmetic ----------------------------------------------

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def reading_years(units, seconds_per_unit) -> float:
    """Years of non-stop reading (or looking) to get through a data set."""
    return units * seconds_per_unit / SECONDS_PER_YEAR

"""Week 11 toys: going beyond the demonstrations, the data flywheel, and
knowing what you don't know.

The shared world for toys 1 and 2 is a small building on a grid: two rooms
joined by an opening, and a corridor below reached through two doors. Moves
are deterministic (right, left, down, up; walls block). With deterministic
moves and one unit of cost per step, the best policy that offline RL can
extract from a data set is the shortest path through the transitions the
data contain, so "offline RL" here is a breadth-first search over observed
transitions.

Toy 1, stitching. Operators walk between landmarks, a little noisily. A
goal-conditioned behavior-cloning policy (with hindsight relabelling) can
only repeat routes that some trajectory took from here to the goal; offline
RL can join pieces of different trajectories. How many landmark-to-landmark
tasks can each solve, as the data grow?

Toy 2, the data flywheel. A human operator demonstrates one task the long
way round. The robot is then deployed from random starts, with some
exploration noise, and its rollouts are added to the data each round. Three
ways to retrain: clone everything (naive co-training), clone the expert plus
only the successful rollouts, or offline RL on everything.

Toy 3, introspection. An ensemble of cloned policies on Week 9's post task:
does their disagreement flag the situations they would fail in? It flags
inputs unlike the training data, but not a change the inputs do not show
(the post has moved).
"""

from collections import deque

import numpy as np

from rlc import nn
from rlc.toys import w09_generalist as w9

Array = np.ndarray

# --- The building ------------------------------------------------------------

MAP = (
    "###############",
    "#.....#.......#",
    "#.....#.......#",
    "#.............#",
    "#.....#.......#",
    "#.....#.......#",
    "###.#####.#####",
    "#.............#",
    "#.............#",
    "###############",
)
FREE = [(r, c) for r, row in enumerate(MAP) for c, ch in enumerate(row)
        if ch == "."]
IDX = {s: i for i, s in enumerate(FREE)}
MOVES = ((0, 1), (0, -1), (1, 0), (-1, 0))      # (row, col): right, left, down, up


def step(s, a):
    """Deterministic move; into a wall means staying put."""
    n = (s[0] + MOVES[a][0], s[1] + MOVES[a][1])
    return n if n in IDX else s


def distances(goal, edges=None) -> dict:
    """Steps to `goal` from every cell that can reach it: with all moves
    (edges=None), or only along observed transitions {(s, a, s2)}."""
    back = {}
    pairs = (((s, step(s, a)) for s in FREE for a in range(4)) if edges is None
             else ((s, s2) for s, _, s2 in edges))
    for s, s2 in pairs:
        if s != s2:
            back.setdefault(s2, set()).add(s)
    d, todo = {goal: 0}, deque([goal])
    while todo:
        c = todo.popleft()
        for p in back.get(c, ()):
            if p not in d:
                d[p] = d[c] + 1
                todo.append(p)
    return d


def downhill(s, d, rng):
    """A random move that brings s one step closer under distances d."""
    best = [a for a in range(4) if d.get(step(s, a), 10**9) < d.get(s, 10**9)]
    return int(rng.choice(best)) if best else int(rng.integers(4))


def rollout(policy, start, goal, horizon, rng):
    """Run policy(s, rng) -> action from start. Returns the transitions
    [(s, a, s2)], whether the goal was reached, and the steps taken."""
    s, traj = start, []
    for t in range(horizon):
        a = policy(s, rng)
        s2 = step(s, a)
        traj.append((s, a, s2))
        s = s2
        if s == goal:
            return traj, True, t + 1
    return traj, False, horizon


def offline_rl_policy(edges, goal):
    """Greedy policy on the shortest path to goal through observed
    transitions; cells with no observed route act at random."""
    d = distances(goal, edges)
    best = {}
    for s, a, s2 in edges:
        if s2 in d and (s not in best or d[s2] < best[s][1]):
            best[s] = (a, d[s2])
    return lambda s, rng: best[s][0] if s in best else int(rng.integers(4))


def bc_policy(pairs):
    """Tabular behavior cloning: at s, sample an action in proportion to
    how often the data took it there; unseen cells act at random."""
    counts = np.zeros((len(FREE), 4))
    for s, a in pairs:
        counts[IDX[s], a] += 1

    def act(s, rng):
        c = counts[IDX[s]]
        if c.sum() == 0:
            return int(rng.integers(4))
        return int(rng.choice(4, p=c / c.sum()))
    return act


# --- Toy 1: stitching --------------------------------------------------------

LANDMARKS = {"A": (2, 2), "B": (8, 2), "C": (8, 12), "D": (2, 12),
             "E": (5, 9), "F": (4, 4)}
P_STRAY = 0.25          # chance an operator's step is a random move
HORIZON_1 = 40          # steps allowed per task (longest shortest path: 22)
DATA_SIZES = (2, 4, 8, 16, 32)


def operator_walk(start, goal, rng, cap=80):
    """An operator heading for goal: a shortest-path move, except with
    chance P_STRAY a random one."""
    d = distances(goal)

    def act(s, g):
        return int(g.integers(4)) if g.random() < P_STRAY else downhill(s, d, g)
    return rollout(act, start, goal, cap, rng)[0]


def operator_data(n, rng):
    """n operator trajectories between random ordered landmark pairs."""
    names = list(LANDMARKS)
    out = []
    for _ in range(n):
        i, j = rng.choice(len(names), 2, replace=False)
        out.append(operator_walk(LANDMARKS[names[i]], LANDMARKS[names[j]], rng))
    return out


def relabelled_bc(trajs, goal):
    """Goal-conditioned BC with hindsight relabelling: every (s, a) taken
    on the way to a later visit of `goal`, in any trajectory, becomes a
    training pair for that goal."""
    pairs = []
    for traj in trajs:
        visits = [t for t, (_, _, s2) in enumerate(traj) if s2 == goal]
        if visits:
            last = visits[-1]
            pairs += [(s, a) for s, a, _ in traj[:last + 1]]
    return bc_policy(pairs)


def stitching_scores(trajs, seed=0):
    """Every ordered landmark pair as a task: success rate and mean extra
    steps over the shortest path (successes only) for BC and offline RL."""
    edges = {e for traj in trajs for e in traj}
    rng = np.random.default_rng(seed)
    out = {m: {"ok": [], "extra": []} for m in ("bc", "offline")}
    for a, sa in LANDMARKS.items():
        for b, sb in LANDMARKS.items():
            if a == b:
                continue
            opt = distances(sb)[sa]
            for m, pol in (("bc", relabelled_bc(trajs, sb)),
                           ("offline", offline_rl_policy(edges, sb))):
                _, ok, steps = rollout(pol, sa, sb, HORIZON_1, rng)
                out[m]["ok"].append(ok)
                if ok:
                    out[m]["extra"].append(steps - opt)
    return {m: (float(np.mean(v["ok"])),
                float(np.mean(v["extra"])) if v["extra"] else float("nan"))
            for m, v in out.items()}


def stitching_curve(seed):
    """stitching_scores for each data size; data for size n are the first
    n trajectories of one stream, so larger sets contain smaller ones."""
    trajs = operator_data(max(DATA_SIZES), np.random.default_rng(seed))
    return {n: stitching_scores(trajs[:n], seed) for n in DATA_SIZES}


def lecture_example(seed=0):
    """The lecture's picture: one demo A -> B and one B -> C, task A -> C.
    Returns the demos and one rollout of each policy."""
    rng = np.random.default_rng(seed)
    demos = [operator_walk(LANDMARKS["A"], LANDMARKS["B"], rng),
             operator_walk(LANDMARKS["B"], LANDMARKS["C"], rng)]
    edges = {e for traj in demos for e in traj}
    goal = LANDMARKS["C"]
    runs = {m: rollout(pol, LANDMARKS["A"], goal, HORIZON_1, rng)
            for m, pol in (("bc", relabelled_bc(demos, goal)),
                           ("offline", offline_rl_policy(edges, goal)))}
    return demos, runs


# --- Toy 2: the data flywheel ------------------------------------------------

GOAL_2 = (1, 13)
HORIZON_2 = 30
EPS = 0.2               # exploration noise in deployments
N_EXPERT, PER_ROUND, ROUNDS = 6, 10, 8
EXPERT_STARTS = ((2, 1), (3, 2), (4, 1))
FLY_METHODS = ("naive", "filtered", "offline")


def operator_detour(rng):
    """The human operator's habit: from the left room, always down through
    the left door, along the corridor and up through the right door, even
    though the opening between the rooms is shorter. A Markov policy."""
    d_door, d_way, d_goal = distances((7, 3)), distances((7, 9)), distances(GOAL_2)

    def act(s, g):
        r, c = s
        if r <= 5 and c <= 6:
            d = d_door
        elif r >= 6 and s not in ((7, 9), (6, 9)):
            d = d_way
        else:
            d = d_goal
        return downhill(s, d, g)
    return act


def fly_policy(method, expert, autonomous):
    """Retrain from the data so far. Each episode is (traj, ok, steps)."""
    if method == "naive":                   # clone everything
        eps_ = expert + autonomous
    elif method == "filtered":              # clone expert + successes only
        eps_ = expert + [e for e in autonomous if e[1]]
    else:                                   # offline RL on everything
        edges = {t for e in expert + autonomous for t in e[0]}
        return offline_rl_policy(edges, GOAL_2)
    return bc_policy([(s, a) for e in eps_ for s, a, _ in e[0]])


def fly_evaluate(policy, seed, reps=4):
    """Every free cell as a start, reps times: success rate, and mean
    steps over the shortest path, a failure counting as HORIZON_2."""
    rng = np.random.default_rng(seed)
    d = distances(GOAL_2)
    starts = [s for s in FREE if s != GOAL_2] * reps
    res = [rollout(policy, s, GOAL_2, HORIZON_2, rng) for s in starts]
    return (float(np.mean([ok for _, ok, _ in res])),
            float(np.mean([n - d[s] for (_, _, n), s in zip(res, starts)])))


def flywheel(method, seed):
    """Run ROUNDS rounds of deploy-and-retrain. Returns (success, extra
    steps) after each round, including round 0 (expert data only)."""
    rng = np.random.default_rng(seed)
    op = operator_detour(rng)
    expert = [rollout(op, EXPERT_STARTS[i % 3], GOAL_2, HORIZON_2, rng)
              for i in range(N_EXPERT)]
    autonomous, out = [], []
    for r in range(ROUNDS + 1):
        pol = fly_policy(method, expert, autonomous)
        out.append(fly_evaluate(pol, 1000 * seed + r))

        def noisy(s, g, p=pol):             # deployments explore a little
            return int(g.integers(4)) if g.random() < EPS else p(s, g)
        starts = [FREE[i] for i in rng.integers(len(FREE), size=PER_ROUND)]
        autonomous += [rollout(noisy, s, GOAL_2, HORIZON_2, rng) for s in starts]
    return out


def expert_extra(seed=0):
    """Extra steps of the operator's own demos over the shortest path."""
    rng = np.random.default_rng(seed)
    op, d = operator_detour(rng), distances(GOAL_2)
    return [rollout(op, s, GOAL_2, HORIZON_2, rng)[2] - d[s] for s in EXPERT_STARTS]


# --- Toy 3: knowing what you don't know --------------------------------------

M_ENS, EPISODES_3, STEPS_3 = 5, 60, 3000
NEW_POST = np.array([0.0, 0.3])          # where the post moves to


def post_demos(rng, n_ep):
    """Week 9's expert driving a plain point robot round the post:
    observations [x, g] (n, 4) and velocities (n, 2)."""
    x, g = w9.sample_tasks(rng, n_ep)
    path, _, _ = w9.run_policy(w9.expert_velocity, np.eye(2), x, g)
    X = np.concatenate([np.c_[p, g] for p in path[:-1]])
    Y = np.concatenate([w9.expert_velocity(p, g) for p in path[:-1]])
    return X, Y


def train_ensemble(X, Y, seed):
    """M_ENS MLPs, each on its own bootstrap resample of the data and its
    own initialization, so they agree where the data pin them down."""
    nets = []
    for m in range(M_ENS):
        rng = np.random.default_rng(1000 * seed + m)
        boot = rng.integers(0, len(X), len(X))
        p = nn.init_mlp([4, 64, 64, 2], rng)
        opt = nn.Adam(p, 0.003)
        for _ in range(STEPS_3):
            b = boot[rng.integers(0, len(boot), 256)]
            Yh, cache = nn.forward(p, X[b])
            p = opt.step(p, nn.backward(p, cache, 2 * (Yh - Y[b]) / len(b)))
        nets.append(p)
    return nets


def ensemble_act(nets, x, g):
    """Mean action (n, 2) and disagreement (n,): the members' standard
    deviation, summed over the two action dimensions."""
    outs = np.stack([nn.predict(p, np.c_[x, g]) for p in nets])   # (M, n, 2)
    return outs.mean(0), outs.std(0).sum(-1)


def run_ensemble(nets, x, g, post=w9.POST):
    """Roll out the ensemble's mean action. Returns success per episode
    (goal reached without touching `post`) and the disagreement averaged
    over each episode."""
    seen = []

    def act(xx, gg):
        a, u = ensemble_act(nets, xx, gg)
        seen.append(u)
        return a
    path, _, t_goal = w9.run_policy(act, np.eye(2), x, g)
    hit = (np.linalg.norm(path - post, axis=2) < w9.R_POST).any(0)
    return (t_goal < w9.T_RUN) & ~hit, np.mean(seen, axis=0)


def conditions(seed):
    """Test tasks: as in training; mirrored (right to left, which the data
    never show); turned (bottom to top); and as in training but with the
    post moved, which nothing in the input reveals."""
    x, g = w9.sample_tasks(np.random.default_rng(5000 + seed), 300)
    swap = lambda v: v[:, ::-1].copy()                  # (x, y) -> (y, x)
    return {"as trained": (x, g, w9.POST), "mirrored": (g.copy(), x.copy(), w9.POST),
            "turned": (swap(x), swap(g), w9.POST), "post moved": (x, g, NEW_POST)}


def introspection(seed):
    """Per condition: success per episode and disagreement per episode."""
    X, Y = post_demos(np.random.default_rng(seed), EPISODES_3)
    nets = train_ensemble(X, Y, seed)
    return {k: run_ensemble(nets, x, g, post) for k, (x, g, post)
            in conditions(seed).items()}, nets


def risk_coverage(ok, unc):
    """Act on the share `coverage` of episodes with the lowest
    disagreement and hand the rest to a person. Returns coverage levels
    and the success rate among the episodes acted on."""
    order = np.argsort(unc, kind="stable")
    cov = np.arange(1, len(ok) + 1) / len(ok)
    return cov, np.cumsum(ok[order]) / np.arange(1, len(ok) + 1)


# --- Worked examples -----------------------------------------------------------

def memory_tokens(seconds, fps, tokens_per_frame) -> int:
    """Tokens needed to keep every frame of a video in context."""
    return int(seconds * fps * tokens_per_frame)


def hard_negative_chance(f, batch) -> float:
    """Chance that a batch holds at least one other example with property
    f (e.g. another cat-and-dog picture), besides the one we look at."""
    return float(1 - (1 - f) ** (batch - 1))


def contrastive_losses(S, t=10.0, b=-10.0):
    """CLIP's softmax loss and SigLIP's sigmoid loss for a similarity
    matrix S (n, n) whose diagonal holds the matching pairs."""
    logits = t * S
    n = len(S)
    lsm_r = logits - np.log(np.exp(logits).sum(1, keepdims=True))   # image -> text
    lsm_c = logits - np.log(np.exp(logits).sum(0, keepdims=True))   # text -> image
    clip = -(np.trace(lsm_r) + np.trace(lsm_c)) / (2 * n)
    z = 2 * np.eye(n) - 1                                # +1 match, -1 not
    sig = np.log1p(np.exp(-z * (logits + b))).sum() / n
    return float(clip), float(sig)

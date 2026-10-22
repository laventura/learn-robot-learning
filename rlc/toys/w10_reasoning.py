"""Week 10 toys: reasoning before acting, searching at test time, and
learning to reason with group-relative policy optimization (GRPO).

Toy 1, reason then act (ECoT and ECoT-Lite). A gripper must reach the
object whose colour the instruction names; the scene holds K objects of
different colours. Demonstrated actions come from a human and are noisy;
the "reasoning" label (where the named object is) comes from a detector
and is precise, as ECoT's labels come from foundation models. One MLP
with two output heads plays the part of a decoder that writes reasoning
tokens and then action tokens: pass 1 sees an empty reasoning slot and
writes the reasoning; pass 2 sees the reasoning in the slot and writes
the action. The ECoT-Lite variants change which losses are used and
whether the slot is filled.

Toy 2, best-of-N through a gap. A planner proposes paths through a gap
in a wall; a verifier picks one. Difficulty is set by how often the
planner's own samples get through (as Snell et al. bin questions). A
bigger planner costs more per sample. Who wins at equal compute, and
what an imperfect verifier caps.

Toy 3, GRPO on how long to think. A policy picks how many reasoning
steps to take before answering; an answer is right only if it took
enough steps for that kind of problem. Rewards are 0 or 1 and the
advantage is group-relative. Does thinking longer emerge, and what
happens when the starting policy almost never gets a hard problem right?
"""

import numpy as np
from scipy.stats import norm

from rlc import nn

Array = np.ndarray

# --- Toy 1: reason, then act ---------------------------------------------

K, C = 2, 4                 # objects per scene, colours
REACH = 0.25                # success: the gripper ends this close
NOISE = 0.2                 # std of the teleop noise on demo actions
HIDDEN, STEPS, BATCH = 64, 6000, 128
P_DROP = 0.5                # reasoning dropout probability
METHODS = ("standard", "ecot", "dropout", "pretrain", "scaffold")


def scenes(rng, n):
    """n scenes: object positions (n, K, 2), colours (n, K) all
    different, the instructed object's index (n,), gripper (n, 2)."""
    pos = rng.uniform(-1, 1, (n, K, 2))
    cols = np.stack([rng.permutation(C)[:K] for _ in range(n)])
    tgt = rng.integers(0, K, n)
    grip = rng.uniform(-1, 1, (n, 2))
    return pos, cols, tgt, grip


def observe(pos, cols, tgt, grip) -> Array:
    """What the policy sees: object positions, their colours one-hot,
    the instructed colour one-hot, and the gripper. (n, K*(2+C)+C+2)"""
    n = len(pos)
    onehot = np.eye(C)[cols]                             # (n, K, C)
    instr = np.eye(C)[cols[np.arange(n), tgt]]           # (n, C)
    return np.concatenate([pos.reshape(n, -1), onehot.reshape(n, -1),
                           instr, grip], axis=1)


def demos(rng, n, noise=NOISE):
    """n one-step demos: observations, reasoning labels (the target's
    position, (n, 2)), noisy actions (n, 2), and the scenes."""
    pos, cols, tgt, grip = scenes(rng, n)
    target = pos[np.arange(n), tgt]
    action = target - grip + noise * rng.standard_normal((n, 2))
    return observe(pos, cols, tgt, grip), target, action, (pos, tgt, grip)


def with_slot(O, slot, filled) -> Array:
    """Append the reasoning slot (n, 2) and a 0/1 flag saying whether it
    is filled. An empty slot is zeros with flag 0."""
    return np.concatenate([O, slot, filled[:, None]], axis=1)


def train_reasoner(O, R, A, method, rng, steps=STEPS):
    """One MLP, outputs [reasoning (2), action (2)]. Each method decides
    which pass gets which loss (a mask picks the trained outputs):

    standard  empty slot -> action
    ecot      empty slot -> reasoning;  true reasoning in slot -> action
    dropout   as ecot, but the slot is emptied for half the action rows
    pretrain  empty slot -> reasoning for half the steps, then -> action
    scaffold  true reasoning in slot (or emptied) -> action; no
              reasoning loss, so the network never learns to write it
    Every method takes the same number of gradient steps."""
    params = nn.init_mlp([O.shape[1] + 3, HIDDEN, HIDDEN, 4], rng)
    opt = nn.Adam(params, 0.003)
    for s in range(steps):
        idx = rng.integers(0, len(O), min(BATCH, len(O)))
        o, r, y = O[idx], R[idx], np.c_[R[idx], A[idx]]
        b = len(idx)
        empty, no, yes = np.zeros((b, 2)), np.zeros(b), np.ones(b)
        on_r = np.c_[np.ones((b, 2)), np.zeros((b, 2))]  # train reasoning
        on_a = 1 - on_r                                  # train action
        keep = rng.random(b) > P_DROP                    # dropout draw
        dropped = np.where(keep[:, None], r, 0.0)
        passes = {                       # (slot, flag, loss mask) per pass
            "standard": [(empty, no, on_a)],
            "ecot": [(empty, no, on_r), (r, yes, on_a)],
            "dropout": [(empty, no, on_r), (dropped, keep * 1.0, on_a)],
            "pretrain": [(empty, no, on_r if s < steps // 2 else on_a)],
            "scaffold": [(dropped, keep * 1.0, on_a)],
        }[method]
        X = np.concatenate([with_slot(o, sl, f) for sl, f, _ in passes])
        M = np.concatenate([m for *_, m in passes])
        Y = np.concatenate([y] * len(passes))
        Yh, cache = nn.forward(params, X)
        dY = 2 * M * (Yh - Y) / len(X)                   # masked MSE
        params = opt.step(params, nn.backward(params, cache, dY))
    return params


def reason(params, O) -> Array:
    """Pass 1: the reasoning the network writes into an empty slot."""
    n = len(O)
    out = nn.predict(params, with_slot(O, np.zeros((n, 2)), np.zeros(n)))
    return out[:, :2]


def act(params, O, think, slot=None) -> Array:
    """The action. think=False: one pass with an empty slot.
    think=True: write the reasoning first (or use a given slot, e.g. a
    human's correction), then a second pass reads it."""
    n = len(O)
    if not think:
        empty = with_slot(O, np.zeros((n, 2)), np.zeros(n))
        return nn.predict(params, empty)[:, 2:]
    slot = reason(params, O) if slot is None else slot
    return nn.predict(params, with_slot(O, slot, np.ones(n)))[:, 2:]


def reach_success(params, think, test, slot=None) -> float:
    """Share of test scenes where the gripper ends within REACH of the
    instructed object."""
    O, _, _, (pos, tgt, grip) = test
    end = grip + act(params, O, think, slot)
    target = pos[np.arange(len(O)), tgt]
    return float((np.linalg.norm(end - target, axis=1) < REACH).mean())


def test_set():
    """The fixed held-out scenes every toy-1 policy is scored on."""
    return demos(np.random.default_rng(999), 2000)


THINKS = {"standard": False, "ecot": True, "dropout": False,
          "pretrain": False, "scaffold": False}


def compare_methods(n, seed, noise=NOISE, test=None):
    """Success of every method trained on n demos (data seed `seed`).
    Adds 'dropout+think' (the dropout network with its reasoning on) and
    the reasoning error of the networks that learn to reason."""
    test = test_set() if test is None else test
    O, R, A, _ = demos(np.random.default_rng(seed), n, noise)
    out = {}
    for m in METHODS:
        p = train_reasoner(O, R, A, m, np.random.default_rng(100 + seed))
        out[m] = reach_success(p, THINKS[m], test)
        if m == "dropout":
            out["dropout+think"] = reach_success(p, True, test)
        if m in ("ecot", "dropout", "pretrain"):
            out[m + " reasoning error"] = float(np.linalg.norm(
                reason(p, test[0]) - test[1], axis=1).mean())
    return out


# --- Toy 2: best-of-N through a gap --------------------------------------

GAP_W = 0.2                                 # gap width
PLANNERS = {"small": dict(tau=0.25, sigma=0.12, cost=1),
            "big": dict(tau=0.12, sigma=0.08, cost=4)}
SIGMA_V = 0.05              # learned verifier's error on the gap centre
BUDGETS = (4, 8, 16, 32, 64, 128, 256)      # samples of the small planner
BINS = {"easy": (0.3, 1.0), "medium": (0.02, 0.3), "hard": (0.0, 0.02)}


def gap_problems(rng, n):
    """Gap centres (n,) and a shared 'knowledge error' z (n,): a planner
    believes the gap is at centre + tau * z. The same z for both
    planners, so a problem one finds hard the other finds hard too."""
    return rng.uniform(-0.6, 0.6, n), rng.standard_normal(n)


def p_through(z, planner) -> Array:
    """Exact chance that one sample of `planner` gets through the gap:
    a sample is N(tau z, sigma^2) around the centre."""
    tau, sig = PLANNERS[planner]["tau"], PLANNERS[planner]["sigma"]
    e = tau * z
    return norm.cdf((GAP_W / 2 - e) / sig) - norm.cdf((-GAP_W / 2 - e) / sig)


def difficulty(z) -> Array:
    """Bin label per problem from the small planner's own success."""
    p = p_through(z, "small")
    labels = np.empty(len(z), dtype=object)
    for name, (lo, hi) in BINS.items():
        labels[(p > lo) & (p <= hi) if lo > 0 else p <= hi] = name
    return labels


def best_of_n(rng, gap, z, planner, N, verifier):
    """Sample N candidates per problem and let the verifier pick one.
    Returns the candidates (n, N), whether each gets through, and the
    picked index. A perfect verifier picks a candidate that gets through
    if there is one; the learned one believes the gap is at centre +
    N(0, SIGMA_V^2) and picks the candidate closest to that."""
    tau, sig = PLANNERS[planner]["tau"], PLANNERS[planner]["sigma"]
    cand = gap[:, None] + tau * z[:, None] + sig * rng.standard_normal(
        (len(gap), N))
    ok = np.abs(cand - gap[:, None]) < GAP_W / 2
    if verifier == "perfect":
        pick = np.argmax(ok, axis=1)        # first that gets through, or 0
    else:
        belief = gap + SIGMA_V * rng.standard_normal(len(gap))
        pick = np.argmin(np.abs(cand - belief[:, None]), axis=1)
    return cand, ok, pick


def search_curves(n=20000, seed=0):
    """Success per difficulty bin against compute budget (in small-planner
    samples), for each planner and verifier. The big planner gets
    budget / cost samples."""
    gap, z = gap_problems(np.random.default_rng(seed), n)
    lab = difficulty(z)
    out = {}
    for planner, cfg in PLANNERS.items():
        for ver in ("perfect", "learned"):
            rows = []
            for B in BUDGETS:
                N = max(B // cfg["cost"], 1)
                _, ok, pick = best_of_n(np.random.default_rng(seed + B), gap,
                                        z, planner, N, ver)
                got = ok[np.arange(n), pick]
                rows.append({b: float(got[lab == b].mean()) for b in BINS})
            out[planner, ver] = rows
    return out, {b: float((lab == b).mean()) for b in BINS}


def verifier_cap() -> float:
    """Best-of-infinity with the learned verifier: the pick converges to
    the verifier's belief, which is inside the gap with this chance."""
    return float(2 * norm.cdf(GAP_W / 2 / SIGMA_V) - 1)


# --- Toy 3: GRPO on how long to think ------------------------------------

L_MAX = 10                                  # longest reasoning allowed
NEED = {"easy": 1, "medium": 4, "hard": 8}  # steps a problem needs
P_RIGHT = 0.9               # chance of a right answer with enough steps
G, BATCH_Q, LR, GRPO_STEPS = 8, 16, 0.5, 300


def grpo_advantages(r) -> Array:
    """Group-relative advantages: (r - mean) / std within one group of
    answers to the same question. All equal rewards give all zeros:
    nothing in the group to prefer."""
    sd = r.std()
    return (r - r.mean()) / sd if sd > 0 else np.zeros_like(r)


def start_logits(lam) -> Array:
    """A base policy that prefers short answers: logit -lam * L, the
    same for every kind of problem. (kinds, L_MAX + 1)"""
    return np.tile(-lam * np.arange(L_MAX + 1.0), (len(NEED), 1))


def accuracy(theta) -> Array:
    """Exact accuracy per kind of problem under logits theta."""
    pi = nn.softmax(theta)
    right = np.array([np.where(np.arange(L_MAX + 1) >= k, P_RIGHT, 0.0)
                      for k in NEED.values()])
    return (pi * right).sum(1)


def grpo_train(lam, rng, steps=GRPO_STEPS):
    """GRPO with one gradient step per batch, so the policy ratio is 1
    and PPO's clip never acts; no KL term. Per step: BATCH_Q questions,
    a group of G answers each, binary rewards, and the gradient of
    sum A_i log pi(L_i). Returns accuracy and mean length per kind
    after every step, (steps, kinds) each."""
    theta = start_logits(lam)
    kinds = list(NEED.values())
    acc, length = [], []
    for _ in range(steps):
        grad = np.zeros_like(theta)
        for _ in range(BATCH_Q):
            k = rng.integers(len(kinds))
            pi = nn.softmax(theta[k])
            L = rng.choice(L_MAX + 1, G, p=pi)           # G answers
            r = (L >= kinds[k]) & (rng.random(G) < P_RIGHT)
            A = grpo_advantages(r.astype(float))
            # d log softmax(theta)[L] / d theta = onehot(L) - pi
            onehot = np.eye(L_MAX + 1)[L]                # (G, L_MAX + 1)
            grad[k] += A @ (onehot - pi) / G
        theta = theta + LR * grad / BATCH_Q              # gradient ascent
        pi = nn.softmax(theta)
        acc.append(accuracy(theta))
        length.append(pi @ np.arange(L_MAX + 1.0))
    return np.array(acc), np.array(length)


def base_success(lam) -> dict:
    """Accuracy of the starting policy on each kind of problem."""
    return dict(zip(NEED, accuracy(start_logits(lam))))


def steps_to(acc, level=0.8) -> int:
    """First step at which accuracy reaches `level` (len(acc) if never)."""
    hit = np.nonzero(acc >= level)[0]
    return int(hit[0]) + 1 if len(hit) else len(acc)


def group_has_success(p, g=G) -> float:
    """Chance that a group of g answers holds at least one right one,
    when each is right with chance p: only then is there a signal."""
    return float(1 - (1 - p) ** g)


# --- Worked example: Elo and test-time search ----------------------------

def elo_win(delta) -> float:
    """Chance that the player rated delta points higher wins."""
    return float(1 / (1 + 10 ** (-delta / 400)))

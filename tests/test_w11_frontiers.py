import numpy as np

from rlc.toys import w11_frontiers as w


# --- The building ------------------------------------------------------------

def test_moves_respect_walls_and_distances_are_consistent():
    assert w.step((1, 5), 0) == (1, 5)                  # wall at column 6
    assert w.step((3, 5), 0) == (3, 6)                  # the opening on row 3
    d = w.distances((1, 13))
    assert len(d) == len(w.FREE)                         # every cell connected
    for s in w.FREE:
        if s != (1, 13):
            assert min(d[w.step(s, a)] for a in range(4)) == d[s] - 1


def test_offline_rl_on_all_transitions_takes_shortest_paths():
    edges = {(s, a, w.step(s, a)) for s in w.FREE for a in range(4)}
    goal = (1, 13)
    pol, d = w.offline_rl_policy(edges, goal), w.distances(goal)
    rng = np.random.default_rng(0)
    for s in w.FREE[::7]:
        if s != goal:
            assert w.rollout(pol, s, goal, 60, rng)[2] == d[s]


# --- Toy 1: stitching --------------------------------------------------------

def test_lecture_example_bc_fails_and_offline_rl_stitches():
    demos, runs = w.lecture_example()
    assert all(t[-1][2] == g for t, g in zip(demos, (w.LANDMARKS["B"], w.LANDMARKS["C"])))
    assert not runs["bc"][1] and runs["offline"][1]


def test_offline_rl_solves_more_tasks_than_bc():
    curves = [w.stitching_curve(s) for s in range(4)]
    for n in (8, 16):
        bc = np.mean([c[n]["bc"][0] for c in curves])
        off = np.mean([c[n]["offline"][0] for c in curves])
        assert off > bc + 0.15          # measured: 0.72 vs 0.44, 0.93 vs 0.62


# --- Toy 2: the flywheel -----------------------------------------------------

def test_flywheel_naive_degrades_and_offline_rl_improves():
    rows = {m: np.array([w.flywheel(m, s) for s in range(4)]) for m in w.FLY_METHODS}
    succ = {m: r[..., 0].mean(0) for m, r in rows.items()}
    assert succ["naive"][-1] < succ["naive"][0] - 0.1
    assert succ["offline"][-1] > 0.97
    assert succ["filtered"][-1] > succ["naive"][-1] + 0.1
    extra = rows["offline"][..., 1].mean(0)
    assert extra[-1] < min(w.expert_extra()) - 2      # beats the operator's habit


# --- Toy 3: introspection ----------------------------------------------------

def test_disagreement_flags_new_inputs_but_not_a_moved_post():
    res, _ = w.introspection(0)
    ok = {k: v[0].mean() for k, v in res.items()}
    unc = {k: v[1].mean() for k, v in res.items()}
    assert ok["as trained"] > 0.95 and ok["mirrored"] < 0.2 and ok["post moved"] < 0.6
    assert unc["mirrored"] > 5 * unc["as trained"]
    assert np.isclose(unc["post moved"], unc["as trained"])


def test_risk_coverage_is_monotone_at_full_coverage():
    ok = np.array([1, 1, 0, 1, 0], bool)
    cov, succ = w.risk_coverage(ok, np.array([0.1, 0.2, 0.9, 0.3, 0.8]))
    assert np.allclose(succ, [1, 1, 1, 0.75, 0.6]) and cov[-1] == 1


# --- Worked examples -----------------------------------------------------------

def test_worked_examples():
    assert w.memory_tokens(3600, 10, 256) == 9_216_000
    assert abs(w.hard_negative_chance(0.001, 1001) - (1 - 0.999 ** 1000)) < 1e-12
    S = np.eye(3) * 0.3 + 0.1
    clip, sig = w.contrastive_losses(S)
    assert clip > 0 and sig > 0
    clip2, _ = w.contrastive_losses(np.eye(3))       # sharper matches: lower loss
    assert clip2 < clip

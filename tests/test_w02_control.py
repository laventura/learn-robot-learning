import numpy as np
import pytest

from rlc.toys import w02_control as w

Q_A, Q_B = np.deg2rad([-20.0, 30.0]), np.deg2rad([110.0, -40.0])
IK_TARGET = np.array([0.3, 1.4])


# --- Toy 1: the arm -----------------------------------------------------------

def test_fk_straight_and_folded():
    _, hand = w.fk([0.0, 0.0])
    assert np.allclose(hand, [w.L1 + w.L2, 0.0])
    _, hand = w.fk([np.pi / 2, np.pi / 2])
    assert np.allclose(hand, [-w.L2, w.L1])


def test_jacobian_matches_finite_differences():
    q, h = np.array([0.4, -1.1]), 1e-6
    num = np.column_stack([(w.fk(q + h * e)[1] - w.fk(q - h * e)[1]) / (2 * h) for e in np.eye(2)])
    assert np.allclose(w.jacobian(q), num, atol=1e-6)


def test_wrap_measures_the_short_way_round():
    assert np.isclose(abs(w.wrap(np.deg2rad(179.0) - np.deg2rad(-179.0))), np.deg2rad(2.0))


def test_homogeneous_transforms_chain_and_invert():
    T = w.homogeneous(0.3, 1.0, -2.0)
    assert np.allclose(T @ np.linalg.inv(T), np.eye(3))
    R = T[:2, :2]
    assert np.allclose(R.T @ R, np.eye(2)) and np.isclose(np.linalg.det(R), 1.0)


def test_both_ik_solutions_reach_the_target():
    sols = w.ik_analytic(IK_TARGET)
    assert sols[0, 1] > 0 > sols[1, 1]
    for q in sols:
        assert np.allclose(w.fk(q)[1], IK_TARGET)


def test_jacobian_transpose_finds_either_solution_depending_on_start():
    sols = w.ik_analytic(IK_TARGET)
    up = w.ik_jacobian_transpose([0.0, 1.0], IK_TARGET)[-1]
    down = w.ik_jacobian_transpose([1.5, -1.0], IK_TARGET)[-1]
    assert np.allclose(w.wrap(up), sols[0], atol=1e-4)
    assert np.allclose(w.wrap(down), sols[1], atol=1e-4)


def test_straight_joint_path_collides_but_planned_path_does_not():
    assert not w.collides(Q_A) and not w.collides(Q_B)
    assert w.collides(w.straight_path(Q_A, Q_B)).any()
    path = w.plan_path(Q_A, Q_B)
    assert not w.collides(path).any()
    cell = 2 * np.pi / w.GRID
    assert np.abs(np.diff(path, axis=0)).max() <= cell + 1e-9   # no jumps at the seams
    assert np.allclose(w.wrap(path[-1] - Q_B), 0, atol=cell)


# --- Toy 2: the joint ---------------------------------------------------------

def test_quintic_boundary_conditions():
    s, h = np.array([0.0, 1.0]), 1e-5
    assert np.allclose(w.quintic(s), [0, 1])
    vel = (w.quintic(s + h) - w.quintic(s - h)) / (2 * h)
    acc = (w.quintic(s + h) - 2 * w.quintic(s) + w.quintic(s - h)) / h ** 2
    assert np.allclose(vel, 0, atol=1e-6) and np.allclose(acc, 0, atol=1e-3)


def test_p_oscillates_pd_sags_pid_removes_the_sag():
    s = {k: w.summary(w.track(g)) for k, g in w.GAINS.items()}
    assert s["P"]["hold_swing"] > 0.1                      # undamped: still swinging
    assert s["PD"]["hold_swing"] < 1e-3
    assert np.isclose(s["PD"]["hold_mean"], w.p_only_sag(), atol=5e-4)
    assert abs(s["PID"]["hold_mean"]) < 0.1 * w.p_only_sag()


def test_lerp_jerks_the_motor_more_than_quintic():
    lerp = w.summary(w.track(w.GAINS["PID"], scaling=w.lerp))
    quin = w.summary(w.track(w.GAINS["PID"], scaling=w.quintic))
    assert lerp["max_du"] > 3 * quin["max_du"]
    assert lerp["peak_u"] > quin["peak_u"]


def test_slow_control_loop_breaks_pid():
    ok = w.summary(w.track(w.GAINS["PID"], freq=25.0))
    bad = w.summary(w.track(w.GAINS["PID"], freq=20.0))
    assert ok["hold_swing"] < 0.01
    assert bad["hold_swing"] > 1.0


# --- Toy 3: the corridor ------------------------------------------------------

@pytest.mark.parametrize("slip", [0.0, 0.1, 0.3])
def test_two_ways_of_computing_J_agree(slip):
    for pol in (w.go_left, w.go_right):
        a = w.expected_return_enum(pol, slip, 0.9, horizon=14)
        b = w.expected_return(pol, slip, 0.9, horizon=14)
        assert np.isclose(a, b)


def test_trajectory_probabilities_sum_to_one():
    total = sum(p for *_, p in w.enumerate_trajectories(w.go_right, 0.2, 10))
    assert np.isclose(total, 1.0)


def test_monte_carlo_agrees_with_exact_return():
    g = w.monte_carlo_return(w.go_right, 0.2, 0.9, 4000, np.random.default_rng(0))
    exact = w.expected_return(w.go_right, 0.2, 0.9)
    assert abs(g.mean() - exact) < 3 * g.std(ddof=1) / np.sqrt(len(g))


def test_discount_flip_point():
    assert np.isclose(w.flip_gamma(0.0), w.flip_gamma_deterministic(), atol=1e-5)
    flips = [w.flip_gamma(s) for s in (0.0, 0.1, 0.2, 0.3)]
    assert np.all(np.diff(flips) > 0)                      # slips favour the near reward


# --- Toy 4: random search -----------------------------------------------------

def test_sparse_reward_slows_random_search():
    st = w.random_search_study(n_seeds=6, iters=40)
    dense = np.nanmedian(w.first_solved(st["dense"]["success"]))
    sparse = w.first_solved(st["sparse"]["success"])
    assert dense <= 3
    assert np.nanmedian(np.where(np.isnan(sparse), 99, sparse)) > 3 * dense
    assert st["sparse"]["informative"][:, :5].mean() < 0.2
    assert st["dense"]["informative"][:, :5].mean() > 0.9

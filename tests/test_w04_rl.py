import numpy as np
import pytest

from rlc.toys import w04_rl as w


def test_expected_reward_is_smooth_version_of_step():
    mu = np.linspace(-2, 3, 501)
    J = w.expected_reward(mu, 0.3)
    assert np.all(np.diff(J) <= 0)                         # decreasing (saturates in floats)
    assert np.all(w.expected_reward_grad(mu, 0.3) < 0)     # analytic gradient never zero
    # Monte Carlo agrees with the closed form
    rng = np.random.default_rng(0)
    y = 0.8 + 0.3 * rng.standard_normal(200_000)
    assert abs(w.step_reward(y).mean() - w.expected_reward(0.8, 0.3)) < 0.01


@pytest.mark.parametrize("slip", [0.0, 0.2])
def test_transition_rows_are_distributions(slip):
    P, _ = w.transition_model(slip)
    assert np.allclose(P.sum(-1), 1.0)


@pytest.mark.parametrize("slip", [0.0, 0.2])
def test_value_and_policy_iteration_agree(slip):
    P, R = w.transition_model(slip)
    V_vi = w.value_iteration(P, R)[-1]
    pol_pi, V_pi = w.policy_iteration(P, R)[-1]
    assert np.allclose(V_vi, V_pi, atol=1e-6)
    assert np.allclose(w.evaluate(P, R, w.greedy(P, R, V_vi)), V_pi, atol=1e-6)


def test_policy_iteration_improves_monotonically():
    P, R = w.transition_model(0.2)
    vals = np.array([V for _, V in w.policy_iteration(P, R)])
    assert np.all(np.diff(vals, axis=0) >= -1e-10)        # every state, every iteration


def test_slippery_floor_changes_the_route():
    rng = np.random.default_rng(0)
    steps = {}
    for slip in (0.0, 0.2):
        P, R = w.transition_model(slip)
        pol = w.greedy(P, R, w.value_iteration(P, R)[-1])
        paths = [w.rollout_grid(pol, slip, rng) for _ in range(500)]
        steps[slip] = np.mean([len(p) - 1 for p in paths])
        assert np.mean([p[-1] == w.GOAL for p in paths]) > 0.97
    assert steps[0.0] == 6 and steps[0.2] > 10


def test_value_iteration_needs_more_sweeps_when_slippery():
    sweeps = [len(w.value_iteration(*w.transition_model(s))) - 1 for s in (0.0, 0.2)]
    assert sweeps[1] > 5 * sweeps[0]


def test_q_learning_hugs_the_cliff_sarsa_keeps_away():
    q_len, s_len, q_ret, s_ret = [], [], [], []
    for seed in range(6):
        Q, rq = w.td_control("q", np.random.default_rng(seed))
        S, rs = w.td_control("sarsa", np.random.default_rng(seed))
        q_len.append(len(w.greedy_path(Q)) - 1)
        s_len.append(len(w.greedy_path(S)) - 1)
        q_ret.append(rq[-100:].mean()); s_ret.append(rs[-100:].mean())
    assert all(n == w.CLIFF_COLS + 1 for n in q_len)      # along the edge
    assert np.median(s_len) > w.CLIFF_COLS + 1            # a longer, safer route
    assert np.mean(s_ret) > np.mean(q_ret)                # but SARSA earns more while exploring


def test_max_of_noisy_estimates_is_biased_and_double_is_not():
    rng = np.random.default_rng(0)
    for k in (2, 8, 32):
        single, double = w.max_bias(k, 1.0, rng)
        assert abs(single - w.expected_max_normal(k)) < 0.03
        assert abs(double) < 0.03


def test_cem_beats_uniform_sampling_at_equal_cost():
    rng = np.random.default_rng(0)
    cem = np.mean([w.cem_argmax(np.random.default_rng(s))[1] for s in range(40)])
    rand = np.mean([w.random_search_argmax(640, rng) for _ in range(40)])    # same 640 evaluations
    assert cem > rand + 0.15


def test_gradient_ascent_climbs_the_nearest_hill():
    a, q = w.gradient_argmax(w.PEAK + 0.3)
    assert q > 0.99
    a, q = w.gradient_argmax(w.DECOY + 0.3)
    assert 0.65 < q < 0.75                                  # stuck on the lower peak

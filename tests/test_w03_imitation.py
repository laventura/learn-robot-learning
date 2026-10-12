"""Tests for rlc.toys.w03_imitation: the behaviour the Week 3 page relies on."""

import numpy as np

from rlc.toys import w03_imitation as m


def test_calm_demos_stay_inside_the_deadband():
    obs, act = m.expert_data(10, 200, np.random.default_rng(0))
    assert np.abs(obs[:, 0]).max() < m.DEADBAND
    assert obs.shape == (2000, 3) and act.shape == (2000,)


def offset_blind(obs):
    """What BC learns from calm demos: the expert's rule inside the deadband."""
    _, heading_err, turn = obs.T
    return np.clip(turn - m.K_HEADING * heading_err, -m.U_MAX, m.U_MAX)


def test_one_gust_separates_expert_from_offset_blind_driver():
    offsets, _ = m.single_gust([m.expert, offset_blind], T=80)
    expert, blind = np.abs(offsets)
    assert expert.max() < m.HALF_WIDTH          # the expert never leaves the road
    assert expert[-1] < m.DEADBAND + 0.05       # and steers back to the deadband
    assert blind[-1] > m.HALF_WIDTH             # the blind driver stays off it


def test_predicted_cost_matches_an_absorbing_simulation():
    rng = np.random.default_rng(1)
    h, T, n = 0.01, 100, 20_000
    first = rng.geometric(h, n)                 # step of the first departure
    off = np.arange(1, T + 1)[None, :] >= first[:, None]
    sim = off.cumsum(axis=1).mean(axis=0)
    se = off.sum(axis=1).std() / np.sqrt(n)
    assert abs(sim[-1] - m.predicted_cost(h, T)[-1]) < 4 * se


def test_first_departure():
    off = np.array([[0, 0, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0]])
    assert list(m.first_departure(off)) == [2, -1, 0]


def test_dagger_beats_bc_pooled_over_seeds():
    # Single BC seeds vary a lot (some pick up a weak steer-back by accident),
    # so pool four.
    left = np.zeros(2)
    for seed in range(4):
        rng = np.random.default_rng(seed)
        bc = m.behavior_cloning(10, 200, rng)
        dg = m.dagger(10, 200, rng, iters=2)
        x0 = np.random.default_rng(5).uniform(0, m.WAVELEN, 100)
        left += [m.off_road(m.rollout(p, 300, 0.01, np.random.default_rng(6), x0)[0]).any(axis=1).mean()
                 for p in (bc, dg)]
    assert left[1] < left[0] / 3


def test_expert_passes_the_obstacle_on_both_sides():
    path, _ = m.rollout2(None, 2, side=np.array([1.0, -1.0]))
    assert not m.hits(path).any()
    assert path[0, :, 1].max() > 1.5 and path[1, :, 1].min() < -1.5


def test_binned_policy_commits_and_mse_averages_when_modes_are_balanced():
    run = m.experiment2(0.5, seed=0)
    fork = np.array([[2.0, 0.0]])
    probs = run["probs"](fork)[0]
    left, right = probs[m.BINS > 0].sum(), probs[m.BINS < 0].sum()
    assert min(left, right) > 0.3                     # two modes kept
    assert abs(run["mse_policy"](fork)[0]) < 0.02     # their average, near 0
    assert abs(run["binned"][25, 1]) > 1.5            # committed to one side
    assert not run["binned_hits"]


def test_with_one_mode_both_policies_clear_the_obstacle():
    run = m.experiment2(1.0, seed=0)
    assert not run["mse_hits"] and not run["binned_hits"]

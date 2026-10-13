"""Tests for rlc.toys.w06_generative: the behaviour the Week 6 page relies on."""

import numpy as np
import pytest

from rlc.toys import w06_generative as m

DATA = m.crescents(2000, np.random.default_rng(0))


def test_data_has_two_modes_and_an_empty_middle():
    assert m.gap_share(DATA) == 0.0
    assert abs(m.left_share(DATA) - 0.5) < 1e-9
    assert np.abs(DATA.mean(axis=0)).max() < 0.05      # the mean sits in the gap
    assert m.gap_share(DATA.mean(axis=0, keepdims=True)) == 1.0


def test_gmm_restarts_find_the_left_right_split():
    w, mu, _ = m.fit_gmm(DATA, 2, np.random.default_rng(0))
    assert sorted(np.round(mu[:, 0]).tolist()) == [-1.0, 1.0]
    assert np.allclose(w, 0.5, atol=0.05)


def test_gmm_loglik_prefers_the_better_fit():
    good = m.fit_gmm(DATA, 2, np.random.default_rng(0))
    one = m.fit_gmm(DATA, 1, np.random.default_rng(0))
    assert m.gmm_loglik(DATA, good) > m.gmm_loglik(DATA, one)


@pytest.mark.parametrize("schedule", [m.linear_schedule, m.cosine_schedule])
def test_schedules_destroy_the_signal(schedule):
    abar = m.alpha_bar(schedule(50))
    assert np.all(np.diff(abar) < 0) and abar[0] > 0.9 and abar[-1] < 0.01


def test_noise_to_matches_stepping_one_step_at_a_time():
    # Jumping to step t must give the same distribution as t single steps;
    # check mean and variance of a fixed point.
    rng = np.random.default_rng(1)
    betas = m.linear_schedule(20)
    x0 = np.full((20000, 2), 1.5)
    x = x0.copy()
    for b in betas[:10]:
        x = np.sqrt(1 - b) * x + np.sqrt(b) * rng.normal(size=x.shape)
    jump = m.noise_to(x0, 10, m.alpha_bar(betas), rng.normal(size=x0.shape))
    assert abs(x.mean() - jump.mean()) < 0.02 and abs(x.var() - jump.var()) < 0.02


def test_ddpm_learns_both_modes_without_the_gap():
    params, losses = m.train_ddpm(DATA, m.linear_schedule(50), np.random.default_rng(2),
                                  steps=1500)
    assert losses[-200:].mean() < losses[:50].mean()
    s = m.ddpm_sample(params, m.linear_schedule(50), 600, np.random.default_rng(3))
    assert m.gap_share(s) < 0.03 and 0.35 < m.left_share(s) < 0.65


def test_one_euler_step_of_flow_matching_returns_the_mean():
    # v(x, 0) = E[x0] - x, so x + v(x, 0) = E[x0]: the MSE answer again.
    params, _ = m.train_flow(DATA, np.random.default_rng(4), steps=1500)
    s = m.flow_sample(params, 600, np.random.default_rng(5), steps=1)
    assert m.gap_share(s) > 0.8
    assert m.gap_share(m.flow_sample(params, 600, np.random.default_rng(5), steps=20)) < 0.05

import numpy as np
import pytest

from rlc.toys import w05_pg as w


@pytest.fixture(scope="module")
def truth():
    return w.true_gradient(0.3, np.random.default_rng(0))


def batch_stats(c, reps=2000, seed=1):
    rng = np.random.default_rng(seed)
    est = [w.gradient_estimates(0.3, 16, rng, c) for _ in range(reps)]
    return {n: (np.mean([e[n] for e in est]), np.std([e[n] for e in est])) for n in w.ESTIMATORS}


@pytest.mark.parametrize("c", [0.0, 10.0])
def test_all_four_estimators_are_unbiased(truth, c):
    _, g = truth
    for name, (m, s) in batch_stats(c).items():
        assert abs(m - g) < 3.5 * s / np.sqrt(2000), name     # within 3.5 standard errors


def test_variance_ladder_and_offset():
    s0 = {n: s for n, (_, s) in batch_stats(0.0).items()}
    s100 = {n: s for n, (_, s) in batch_stats(100.0).items()}
    stds = [s0[n] for n in w.ESTIMATORS]
    assert all(a > b for a, b in zip(stds, stds[1:]))          # each fix lowers the spread
    assert s100["full return"] > 100 * s0["full return"]       # the offset multiplies the noise
    for n in ("minus mean baseline", "minus fitted V"):
        assert abs(s100[n] / s0[n] - 1) < 0.15                 # a baseline removes the offset


def test_best_gain_is_one():
    # k = 1 lands on the goal in one step; any other k leaves error behind.
    rng = np.random.default_rng(2)
    J = {k: w.true_gradient(k, rng, n=400_000)[0] for k in (0.8, 1.0, 1.2)}
    assert J[1.0] > max(J[0.8], J[1.2])
    assert abs(J[1.0] + w.T * w.SIGMA**2) < 0.01               # J* = -T sigma^2


def test_baselines_make_learning_insensitive_to_the_offset():
    for name in ("minus mean baseline", "minus fitted V"):
        ks = [w.reinforce_reach(name, np.random.default_rng(s), c=10.0)[-1] for s in range(10)]
        assert max(abs(k - 1) for k in ks) < 0.15


def test_bump_return_matches_monte_carlo():
    a = 0.7 + 0.8 * np.random.default_rng(3).standard_normal(400_000)
    assert abs(w.bump(a).mean() - w.bump_return(0.7, 0.8)) < 2e-3


def test_surrogate_gradient_matches_finite_differences():
    rng = np.random.default_rng(4)
    a = rng.standard_normal(32)
    adv = rng.standard_normal(32)
    lp_old = w.log_prob(a, 0.0, 0.0)
    mu, ls, h = 0.3, -0.2, 1e-6

    def L(m, s):
        return np.mean(np.exp(w.log_prob(a, m, s) - lp_old) * adv)

    g, _ = w.surrogate_step(a, adv, lp_old, mu, ls)
    fd = [(L(mu + h, ls) - L(mu - h, ls)) / (2 * h), (L(mu, ls + h) - L(mu, ls - h)) / (2 * h)]
    assert np.allclose(g, fd, rtol=1e-5)


def test_clip_switches_off_samples_outside_the_interval():
    a = np.array([1.0, 1.0, -1.0])
    adv = np.array([1.0, -1.0, 1.0])
    lp_old = w.log_prob(a, 0.0, 0.0)
    # moving mu to 1 raises the ratio at a = 1 to e^0.5 and lowers it at a = -1
    g_clip, ratio = w.surrogate_step(a, adv, lp_old, 1.0, 0.0, eps=0.2)
    assert ratio[0] > 1.2 and ratio[2] < 0.8
    # Sample 0 (adv > 0, ratio above 1 + eps) is switched off. The other two
    # moved against their advantage, so the min keeps their unclipped terms.
    g_rest, _ = w.surrogate_step(a[1:], adv[1:], lp_old[1:], 1.0, 0.0)
    assert np.allclose(g_clip, g_rest * 2 / 3)


def test_reuse_helps_and_clip_prevents_losing_the_bump():
    def finals(K, clip, seeds=50):
        return np.array([w.reuse_run(K, clip, s)["J"][-1] for s in range(seeds)])
    assert finals(10, True).mean() > 4 * finals(1, True).mean()    # reuse is far more data-efficient
    for K in (20, 100):
        assert (finals(K, True) < 0.05).mean() == 0.0              # clipped runs never lose it
    assert (finals(20, False) < 0.05).mean() >= 0.15               # unclipped runs do


def test_trajectory_weight_ess_matches_sampling():
    rng = np.random.default_rng(5)
    for T in (1, 10, 50):
        wts = w.trajectory_weights(T, 0.2, 200_000, rng)
        assert abs(wts.mean() - 1) < 0.02                          # unbiased weights
        assert abs(wts.mean() ** 2 / np.mean(wts**2) - w.trajectory_weight_ess(T, 0.2)) < 0.02


def test_reparameterization_has_lower_variance():
    rng = np.random.default_rng(6)
    est = [w.mu_gradient_estimates(0.5, 0.5, 16, rng) for _ in range(10_000)]
    exact = w.mu_gradient_exact(0.5, 0.5)
    stats = {k: (np.mean([e[k] for e in est]), np.std([e[k] for e in est])) for k in est[0]}
    for m, s in stats.values():
        assert abs(m - exact) < 3.5 * s / 100                      # unbiased
    assert stats["reparameterization"][1] < 0.6 * stats["score function"][1]

"""Week 5 toys: policy gradients, from REINFORCE to clipped off-policy updates.

Toy 1, variance. A point robot on a line reaches for a goal in T steps with
a linear-Gaussian policy, a_t = k (g - x_t) + sigma * noise. The gradient
of the expected return with respect to the gain k is estimated four ways
(full return, reward-to-go, minus a mean baseline, minus a fitted value
function), all unbiased; a constant bonus c added to every reward shows why
the first two are so noisy.

Toy 2, reusing data. A one-step task (a bandit) with a continuous action:
a Gaussian policy over a, a noisy reward bump. Each batch is collected
once and then used for K gradient steps on the importance-weighted
objective, with or without the clipped ratio. The page shows results and
equations only: implementing the clipped objective is the course's
homework. Also: the variance of a whole-trajectory importance weight,
which grows exponentially with the horizon.

Toy 3, two ways to differentiate E[Q(a)] for a ~ N(mu, sigma^2): the score
function (as REINFORCE does) and the reparameterization a = mu + sigma eps
(as SAC does). Both are unbiased; their variances differ.
"""

import numpy as np

Array = np.ndarray

# --- Toy 1: a 1-D reach and four gradient estimators -------------------------------

T, GOAL, SIGMA = 10, 1.0, 0.3            # horizon, goal position, policy noise
ESTIMATORS = ("full return", "reward-to-go", "minus mean baseline", "minus fitted V")


def reach_rollouts(k, n, rng, c=0.0) -> tuple[Array, Array, Array]:
    """n episodes from x_0 = 0 with gain k; reward c - (x_{t+1} - g)^2 per step.

    Returns the errors e_t = g - x_t, the score d/dk log pi(a_t | x_t),
    and the rewards, each of shape (n, T).
    """
    x = np.zeros(n)
    e, a, r = (np.empty((n, T)) for _ in range(3))
    for t in range(T):
        e[:, t] = GOAL - x
        a[:, t] = k * e[:, t] + SIGMA * rng.standard_normal(n)
        x = x + a[:, t]
        r[:, t] = c - (x - GOAL) ** 2
    # log N(a; k e, sigma^2) = -(a - k e)^2 / (2 sigma^2) + const, so its
    # derivative in k is (a - k e) e / sigma^2: noise times state, scaled.
    score = (a - k * e) * e / SIGMA**2
    return e, score, r


def reward_to_go(r) -> Array:
    """sum_{t' >= t} r_t' for every t: a reversed cumulative sum, (n, T)."""
    return np.cumsum(r[:, ::-1], axis=1)[:, ::-1]


def fit_value(e, rtg) -> Array:
    """Per-step least squares V_t(e) = w0 + w1 e^2; returns weights (T, 2)."""
    w = []
    for t in range(T):
        X = np.stack([np.ones(len(e)), e[:, t] ** 2], axis=1)   # (n, 2)
        w.append(np.linalg.lstsq(X, rtg[:, t], rcond=None)[0])
    return np.array(w)


def gradient_estimates(k, n, rng, c=0.0) -> dict[str, float]:
    """One batch of n episodes, four estimates of dJ/dk (same samples)."""
    e, score, r = reach_rollouts(k, n, rng, c)
    rtg = reward_to_go(r)
    # A baseline must not depend on the episode it is subtracted from, or it
    # biases the estimate: use the mean of the *other* n - 1 episodes, and
    # fit V on a separate batch, as a critic trained on earlier data would be.
    loo_mean = (rtg.sum(0) - rtg) / (n - 1)
    e_fit, _, r_fit = reach_rollouts(k, n, rng, c)
    w = fit_value(e_fit, reward_to_go(r_fit))     # (T, 2)
    V = w[:, 0] + w[:, 1] * e**2                  # (n, T), broadcast over n
    weights = (r.sum(1, keepdims=True), rtg, rtg - loo_mean, rtg - V)
    return {name: float((score * wt).sum(1).mean())
            for name, wt in zip(ESTIMATORS, weights)}


def true_gradient(k, rng, n=2_000_000) -> tuple[float, float]:
    """J(k) and dJ/dk at c = 0 by a very large Monte Carlo batch."""
    _, score, r = reach_rollouts(k, n, rng)
    return float(r.sum(1).mean()), float((score * reward_to_go(r)).sum(1).mean())


K_RANGE = (0.0, 2.0)          # beyond k = 2 the error grows every step: x_t diverges


def reinforce_reach(estimator, rng, c=0.0, iters=100, n=16, lr=0.005, k0=0.0) -> Array:
    """Plain gradient ascent on k with one estimator; returns k after each batch."""
    k, ks = k0, [k0]
    for _ in range(iters):
        k = float(np.clip(k + lr * gradient_estimates(k, n, rng, c)[estimator], *K_RANGE))
        ks.append(k)
    return np.array(ks)


# --- Toy 2: a noisy bump, and reusing each batch K times ---------------------------

BUMP_AT, BUMP_WIDTH, NOISE = 2.0, 0.5, 0.3
MU0, SIGMA0 = 0.0, 1.0                   # the first policy is centred away from the bump


def bump(a) -> Array:
    """Mean reward: a bump of height 1 centred at BUMP_AT."""
    return np.exp(-0.5 * ((np.asarray(a) - BUMP_AT) / BUMP_WIDTH) ** 2)


def bump_return(mu, sigma) -> Array:
    """J(mu, sigma) = E[bump(a)] for a ~ N(mu, sigma^2), in closed form.

    A Gaussian times a Gaussian bump integrates to another Gaussian whose
    variance is the sum of the two.
    """
    v = BUMP_WIDTH**2 + np.asarray(sigma) ** 2
    return BUMP_WIDTH / np.sqrt(v) * np.exp(-0.5 * (np.asarray(mu) - BUMP_AT) ** 2 / v)


def log_prob(a, mu, log_sigma) -> Array:
    """log N(a; mu, sigma^2) up to the constant, which cancels in every ratio."""
    return -0.5 * ((a - mu) / np.exp(log_sigma)) ** 2 - log_sigma


def surrogate_step(a, adv, logp_old, mu, log_sigma, eps=None) -> tuple[Array, Array]:
    """Gradient of mean(ratio * adv) in (mu, log sigma); eps clips the ratio.

    With the clip, a sample stops contributing once its ratio has moved
    past 1 +- eps in the direction its advantage favours. Returns the
    gradient (2,) and the ratios (n,).
    """
    sigma = np.exp(log_sigma)
    ratio = np.exp(log_prob(a, mu, log_sigma) - logp_old)
    z = (a - mu) / sigma
    d_mu, d_ls = ratio * z / sigma, ratio * (z**2 - 1)        # d ratio / d parameter
    live = np.ones_like(a, dtype=bool)
    if eps is not None:
        live = ~(((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps)))
    return np.array([(live * adv * d_mu).mean(), (live * adv * d_ls).mean()]), ratio


def reuse_run(K, clip, seed, batches=30, n=32, lr=0.1, eps=0.2, keep=False) -> dict:
    """Collect a batch, take K steps on it, repeat. Records J, mu, sigma per batch.

    K = 1 is the on-policy method (one step per batch, ratio 1). The
    advantage is the reward minus the batch mean. keep=True also returns
    every batch's actions, rewards and advantages, under "batches".
    """
    rng = np.random.default_rng(seed)
    mu, ls = MU0, np.log(SIGMA0)
    rec = {"J": [bump_return(mu, SIGMA0)], "mu": [mu], "sigma": [SIGMA0], "max_ratio": []}
    kept = []
    for _ in range(batches):
        a = mu + np.exp(ls) * rng.standard_normal(n)
        r = bump(a) + NOISE * rng.standard_normal(n)
        adv = r - r.mean()
        logp_old = log_prob(a, mu, ls)
        kept.append((a, r, adv))
        for _ in range(K):
            g, _ = surrogate_step(a, adv, logp_old, mu, ls, eps if clip else None)
            mu, ls = mu + lr * g[0], ls + lr * g[1]
        rec["max_ratio"].append(float(np.exp(log_prob(a, mu, ls) - logp_old).max()))
        rec["J"].append(float(bump_return(mu, np.exp(ls))))
        rec["mu"].append(mu)
        rec["sigma"].append(float(np.exp(ls)))
    out = {key: np.array(v) for key, v in rec.items()}
    if keep:
        out["batches"] = kept
    return out


def surrogates(a, adv, logp_old, mu, sigma, eps=0.2) -> tuple[float, float]:
    """The unclipped and clipped objectives of one batch at a new policy (mu, sigma).

    Both equal 0 at the old policy, because the advantages have mean 0.
    """
    ratio = np.exp(log_prob(a, mu, np.log(sigma)) - logp_old)
    clipped = np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)
    return float((ratio * adv).mean()), float(clipped.mean())


def trajectory_weight_ess(T, shift) -> Array:
    """Effective sample size fraction of the product of T per-step ratios.

    Old and new policies are Gaussians with equal sigma and means `shift`
    sigmas apart. Each ratio has mean 1 and E[ratio^2] = exp(shift^2), so
    the product has E[w^2] = exp(T shift^2), and ESS / N = 1 / E[w^2].
    """
    return np.exp(-np.asarray(T) * shift**2)


def trajectory_weights(T, shift, n, rng) -> Array:
    """Sampled products of T per-step ratios, actions from the old policy."""
    eps = rng.standard_normal((n, T))                      # old policy N(0, 1)
    # log N(eps; shift, 1) - log N(eps; 0, 1) = shift * eps - shift^2 / 2
    return np.exp((shift * eps - shift**2 / 2).sum(axis=1))


# --- Toy 3: score function versus reparameterization -------------------------------

def bump_slope(a) -> Array:
    """d bump / da, which only the pathwise estimator uses."""
    return -(np.asarray(a) - BUMP_AT) / BUMP_WIDTH**2 * bump(a)


def mu_gradient_estimates(mu, sigma, n, rng) -> dict[str, float]:
    """Three unbiased estimates of d/dmu E[bump(a)] from one batch of n."""
    eps = rng.standard_normal(n)
    a = mu + sigma * eps
    q = bump(a)
    # d/dmu log N(a; mu, sigma^2) = (a - mu) / sigma^2 = eps / sigma
    score = eps / sigma
    loo = (q.sum() - q) / (n - 1) if n > 1 else 0.0     # leave-one-out baseline
    return {"score function": float((q * score).mean()),
            "score function, baseline": float(((q - loo) * score).mean()),
            # a = mu + sigma * eps, so da/dmu = 1 and the gradient is Q'(a)
            "reparameterization": float(bump_slope(a).mean())}


def mu_gradient_exact(mu, sigma) -> float:
    """d/dmu of bump_return, in closed form."""
    v = BUMP_WIDTH**2 + sigma**2
    return float(-(mu - BUMP_AT) / v * bump_return(mu, sigma))

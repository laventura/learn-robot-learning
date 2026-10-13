"""Week 6 toys: generative models for multimodal actions.

The data are demonstrated 2-D actions with two curved modes, like a gripper
that passes an obstacle on the left or the right at varying heights. The
gap between the modes is where no demonstrator ever went. Models, from
simplest to richest:

- MSE regression: one action, the mean, which lands in the gap.
- One Gaussian, and a mixture of two Gaussians fitted by EM.
- A denoising diffusion model (DDPM), sampled with DDPM or DDIM steps.
- Flow matching, sampled with Euler steps.

The diffusion and flow models use the NumPy MLP from rlc.nn as the network
that predicts noise or velocity. The forward (noising) process and the
noise schedules are here too.
"""

import numpy as np

from rlc import nn

Array = np.ndarray


# --- Data -----------------------------------------------------------------

GAP = 0.5            # |x| < GAP is the empty middle between the two modes


def crescents(n, rng, noise=0.06) -> Array:
    """n demonstrated actions (n, 2): half on a left crescent "(", half on
    a right crescent ")", mirror images of each other."""
    u = rng.uniform(-1, 1, n)
    side = np.where(np.arange(n) < n // 2, -1.0, 1.0)
    x = side * (1.0 + 0.4 * u**2)                 # bulges outwards
    y = 0.8 * u
    return np.stack([x, y], axis=1) + noise * rng.normal(size=(n, 2))


def gap_share(samples) -> float:
    """Share of actions in the empty middle, where no demo went."""
    return float((np.abs(samples[:, 0]) < GAP).mean())


def off_support(samples, data, radius=0.2) -> float:
    """Share of samples farther than `radius` from every demonstrated
    action: actions the demonstrations don't support."""
    d2 = ((samples[:, None, :] - data[None, :, :]) ** 2).sum(-1)  # (n, m)
    return float((np.sqrt(d2.min(axis=1)) > radius).mean())


def left_share(samples) -> float:
    return float((samples[:, 0] < 0).mean())


# --- Classical baselines --------------------------------------------------

def fit_gmm(X, k, rng, iters=200, restarts=5):
    """Mixture of k Gaussians by expectation-maximization. EM can stop in
    a poor local optimum (for example, splitting two side-by-side modes
    top/bottom), so it runs from `restarts` random starts and keeps the
    fit with the highest log-likelihood.
    Returns (weights (k,), means (k, 2), covariances (k, 2, 2))."""
    fits = [_em(X, k, rng, iters) for _ in range(restarts)]
    return max(fits, key=lambda g: gmm_loglik(X, g))


def gmm_loglik(X, gmm) -> float:
    """Total log-likelihood of the data under the mixture."""
    w, mu, cov = gmm
    logp = np.stack([np.log(w[j]) + log_gauss(X, mu[j], cov[j])
                     for j in range(len(w))], axis=1)
    m = logp.max(axis=1, keepdims=True)
    return float((m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))).sum())


def _em(X, k, rng, iters):
    """One EM run from a k-means++-style start: each new initial mean is
    drawn with probability proportional to its squared distance from the
    means already chosen, so the starts are spread out; initial covariances
    are a tenth of the data's, so each component starts local."""
    n, d = X.shape
    mu = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d2 = np.min([((X - c) ** 2).sum(axis=1) for c in mu], axis=0)
        mu.append(X[rng.choice(n, p=d2 / d2.sum())])
    mu = np.array(mu)
    cov = np.array([0.1 * np.cov(X.T) for _ in range(k)])
    w = np.full(k, 1.0 / k)
    for _ in range(iters):
        # E-step: responsibility of each component for each point, (n, k)
        logp = np.stack([np.log(w[j]) + log_gauss(X, mu[j], cov[j])
                         for j in range(k)], axis=1)
        logp -= logp.max(axis=1, keepdims=True)   # for numerical stability
        r = np.exp(logp)
        r /= r.sum(axis=1, keepdims=True)
        # M-step: refit each component to the points it is responsible for
        nk = r.sum(axis=0)
        w = nk / n
        mu = (r.T @ X) / nk[:, None]
        for j in range(k):
            dx = X - mu[j]
            cov[j] = (r[:, j, None] * dx).T @ dx / nk[j] + 1e-6 * np.eye(d)
    return w, mu, cov


def log_gauss(X, mu, cov) -> Array:
    """log N(x; mu, cov) for each row of X."""
    dx = X - mu
    inv = np.linalg.inv(cov)
    quad = np.einsum("ni,ij,nj->n", dx, inv, dx)
    log_norm = np.log(np.linalg.det(cov)) + len(mu) * np.log(2 * np.pi)
    return -0.5 * (quad + log_norm)


def sample_gmm(gmm, n, rng) -> Array:
    w, mu, cov = gmm
    comp = rng.choice(len(w), n, p=w)
    return np.array([rng.multivariate_normal(mu[c], cov[c]) for c in comp])


# --- Noise schedules and the forward process ------------------------------

def linear_schedule(T, beta_1=1e-3, beta_T=0.2) -> Array:
    """beta_1 .. beta_T rising linearly (Ho et al., 2020 use 1e-4 to 0.02
    over T = 1000 steps; with only T = 50 steps the endpoints are larger)."""
    return np.linspace(beta_1, beta_T, T)


def cosine_schedule(T, s=0.008) -> Array:
    """Betas from alpha-bar_t = f(t)/f(0) with
    f(t) = cos^2((t/T + s)/(1 + s) * pi/2)
    (Nichol & Dhariwal, 2021, Sec. 3.2), clipped at 0.999."""
    f = lambda t: np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    t = np.arange(T + 1)
    abar = f(t) / f(0)
    return np.clip(1 - abar[1:] / abar[:-1], 0, 0.999)


def alpha_bar(betas) -> Array:
    """alpha-bar_t = product of (1 - beta_i) for i <= t: how much of the
    clean signal survives after t noising steps. Index 0 is t = 1."""
    return np.cumprod(1 - betas)


def noise_to(x0, t, abar, eps) -> Array:
    """Jump straight to step t (1-based):
    x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps."""
    a = abar[t - 1]
    a = a[:, None] if np.ndim(a) else a
    return np.sqrt(a) * x0 + np.sqrt(1 - a) * eps


# --- The network: an MLP with the time step as an extra input -------------

def features(x, t) -> Array:
    """Network input (n, 9): the point (n, 2) and a smooth encoding of the
    time t in [0, 1] (scalar or (n,)), so one network serves every step."""
    t = np.broadcast_to(np.asarray(t, dtype=float), (len(x),))[:, None]
    k = np.arange(1, 4)                                   # 3 frequencies
    return np.concatenate([x, t, np.sin(np.pi * k * t), np.cos(np.pi * k * t)],
                          axis=1)


def train_net(make_batch, rng, steps=4000, batch=512, hidden=128, lr=3e-3):
    """Adam on fresh (input, target) batches from make_batch(n, rng), with
    an MSE loss. Returns the trained layers and the loss per step."""
    params = nn.init_mlp([9, hidden, hidden, 2], rng)
    opt = nn.Adam(params, alpha=lr)
    losses = []
    for _ in range(steps):
        X, Y = make_batch(batch, rng)
        loss, grads = nn.loss_and_grads(params, X, Y, "mse", "tanh")
        params = opt.step(params, grads)
        losses.append(loss)
    return params, np.array(losses)


# --- Diffusion (DDPM / DDIM) ----------------------------------------------

def train_ddpm(data, betas, rng, **kw):
    """Train the noise predictor eps_theta(x_t, t) with the simple DDPM loss:
    pick a step t and noise eps, jump to x_t, regress eps."""
    T, abar = len(betas), alpha_bar(betas)

    def batch(n, rng):
        x0 = data[rng.integers(len(data), size=n)]
        t = rng.integers(1, T + 1, size=n)                # 1..T
        eps = rng.normal(size=x0.shape)
        return features(noise_to(x0, t, abar, eps), t / T), eps

    return train_net(batch, rng, **kw)


def ddpm_sample(params, betas, n, rng, keep_path=False):
    """Start from pure noise and take T denoising steps; each step removes
    the predicted noise and adds back a little fresh noise (sigma_t^2 =
    beta_t). Returns samples (n, 2), or the whole path (T+1, n, 2)."""
    T, abar = len(betas), alpha_bar(betas)
    x = rng.normal(size=(n, 2))
    path = [x]
    for t in range(T, 0, -1):
        a, ab = 1 - betas[t - 1], abar[t - 1]
        eps = nn.predict(params, features(x, t / T))      # (n, 2)
        mean = (x - (1 - a) / np.sqrt(1 - ab) * eps) / np.sqrt(a)
        # Fresh noise keeps samples spread out; none on the last step.
        z = rng.normal(size=x.shape) if t > 1 else 0.0
        x = mean + np.sqrt(betas[t - 1]) * z
        path.append(x)
    return np.array(path) if keep_path else x


def ddim_sample(params, betas, n, rng, steps=10, keep_path=False):
    """Deterministic DDIM: visit only `steps` of the T time steps. At each,
    estimate the clean point from the predicted noise, then re-noise it to
    the next visited step's level with that same noise (no fresh noise)."""
    T, abar = len(betas), alpha_bar(betas)
    ts = np.unique(np.linspace(T, 1, steps).round().astype(int))[::-1]
    x = rng.normal(size=(n, 2))
    path = [x]
    for i, t in enumerate(ts):
        eps = nn.predict(params, features(x, t / T))
        x0 = (x - np.sqrt(1 - abar[t - 1]) * eps) / np.sqrt(abar[t - 1])
        if i + 1 < len(ts):
            a_next = abar[ts[i + 1] - 1]
            x = np.sqrt(a_next) * x0 + np.sqrt(1 - a_next) * eps
        else:
            x = x0
        path.append(x)
    return np.array(path) if keep_path else x


# --- Flow matching --------------------------------------------------------

def train_flow(data, rng, **kw):
    """Train the velocity field v_theta(x_t, t) on straight paths
    x_t = (1 - t) eps + t x0, t in [0, 1], whose velocity is x0 - eps."""
    def batch(n, rng):
        x0 = data[rng.integers(len(data), size=n)]
        eps = rng.normal(size=x0.shape)
        t = rng.uniform(0, 1, size=n)
        xt = (1 - t)[:, None] * eps + t[:, None] * x0
        return features(xt, t), x0 - eps

    return train_net(batch, rng, **kw)


def flow_sample(params, n, rng, steps=10, keep_path=False):
    """Euler integration of the learned velocity from noise (t = 0) to
    data (t = 1): x <- x + v(x, t) dt."""
    x = rng.normal(size=(n, 2))
    path, dt = [x], 1.0 / steps
    for k in range(steps):
        x = x + nn.predict(params, features(x, k * dt)) * dt
        path.append(x)
    return np.array(path) if keep_path else x

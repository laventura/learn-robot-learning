import numpy as np
import pytest

from rlc.toys import w08_world as w


# --- Toy 1: the puck table and its learned model ------------------------------

def test_motor_saturates_and_hill_pushes_away():
    s = np.zeros((1, 4))
    s[0, :2] = [-3.0, 0.0]                         # far from the hill
    v1 = w.true_step(s, np.array([[1.0, 0.0]]))[0, 2]
    v3 = w.true_step(s, np.array([[3.0, 0.0]]))[0, 2]
    assert v1 == v3 > 0                            # pushes beyond SAT do nothing more
    f = w.hill_force(np.array([[0.3, 0.0], [-0.3, 0.0]]))
    assert f[0, 0] > 0 > f[1, 0]


@pytest.fixture(scope="module")
def models():
    return {n: w.fit_world_model(n, 0) for n in (250, 1000, 4000)}


def test_more_data_better_model_and_errors_compound(models):
    rng = np.random.default_rng(1)
    s, a, s2 = w.random_transitions(3000, rng)
    one_step = {n: np.sqrt(np.mean((w.model_step(m, s, a) - s2) ** 2)) for n, m in models.items()}
    assert one_step[250] > one_step[1000] > one_step[4000]
    s0 = np.c_[rng.uniform(-1.5, 1.5, (200, 2)), np.zeros((200, 2))]
    A = np.clip(np.cumsum(rng.normal(0, 0.3, (200, 30, 2)), 1), -1, 1)
    true = w.rollout(w.true_step, s0, A)
    pred = w.rollout(lambda x, u: w.model_step(models[1000], x, u), s0, A)
    err = np.linalg.norm(pred[:, :, :2] - true[:, :, :2], axis=-1).mean(0)
    assert err[30] > 10 * err[3] > 0               # grows much faster than the horizon


def test_looking_beats_dreaming(models):
    step = lambda x, u: w.model_step(models[1000], x, u)
    real_look, _ = w.run_mpc(step, np.random.default_rng(10))
    real_blind, imag = w.run_mpc(step, np.random.default_rng(10), look=False)
    d = lambda p: np.linalg.norm(p[-1, :2] - w.GOAL)
    assert d(imag) < 0.05 and d(real_look) < 0.05  # the dream reaches the goal ...
    assert d(real_blind) > 0.3                     # ... the blind replay does not


def test_planner_exploits_pushes_beyond_the_data(models):
    step = lambda x, u: w.model_step(models[1000], x, u)
    s0 = w.START_STATE

    def gap(a_max):
        plan, imagined = w.cem_plan(step, s0, np.random.default_rng(0), H=25, a_max=a_max)
        real = w.cost(w.rollout(w.true_step, s0[None], plan[None]))[0]
        return real - imagined
    assert gap(3.0) > 5 and gap(3.0) > 3 * abs(gap(1.0))


# --- Toy 2: JEPA ------------------------------------------------------------------

def test_plain_jepa_collapses_and_regularizer_does_not():
    plain = w.jepa_train("plain", 0, steps=10_000)
    reg = w.jepa_train("covariance regularizer", 0, steps=10_000)
    assert plain["spread"][-1] < 0.15 * plain["spread"][0]
    assert plain["loss"][-1] < reg["loss"][-1]     # collapse has the lower loss
    assert reg["rank"][-1] > 3.8 and abs(reg["spread"][-1] - 1) < 0.05
    assert w.probe_r2(reg["W"]) > 0.99


def test_effective_rank_bounds():
    rng = np.random.default_rng(0)
    assert abs(w.effective_rank(rng.standard_normal((5000, 4))) - 4) < 0.1
    line = rng.standard_normal((5000, 1)) * np.array([[1.0, 2.0, 0.0, 0.0]])
    assert abs(w.effective_rank(line) - 1) < 1e-6


# --- Toy 3: video model + inverse dynamics ---------------------------------------

@pytest.fixture(scope="module")
def demos():
    return w.expert_demos(20, 0)


def test_recovered_pushes_reproduce_the_demos(demos):
    S, A = demos
    assert np.abs(w.true_step(S[:, :-1], A) - S[:, 1:]).max() < 1e-9
    assert np.linalg.norm(S[:, -1, :2] - w.GOAL, axis=1).max() < 0.1


def test_video_model_needs_fewer_action_labels_than_bc(demos):
    S, A = demos
    vam, bc = w.fit_vam(S, 25, 0), w.fit_bc(S, A, 25, 0)
    d_vam = w.evaluate(lambda s: w.vam_act(vam, s), 100)
    d_bc = w.evaluate(lambda s: np.clip(w.mlp_apply(bc, s), -1, 1), 100)
    assert np.mean(d_vam < 0.2) >= 0.9 > np.mean(d_bc < 0.2)

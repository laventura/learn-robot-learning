import numpy as np
import pytest

from rlc.toys import w10_reasoning as w


# --- Toy 1: reason, then act ---------------------------------------------

def test_demo_labels_are_consistent():
    O, R, A, (pos, tgt, grip) = w.demos(np.random.default_rng(0), 300, noise=0)
    assert O.shape == (300, w.K * (2 + w.C) + w.C + 2)
    assert np.allclose(R, pos[np.arange(300), tgt])
    assert np.allclose(A, R - grip)


@pytest.fixture(scope="module")
def trained():
    """ECoT, dropout and standard networks on 2000 noisy demos."""
    O, R, A, _ = w.demos(np.random.default_rng(0), 2000)
    return {m: w.train_reasoner(O, R, A, m, np.random.default_rng(100))
            for m in ("standard", "ecot", "dropout", "scaffold")}


def test_reasoning_labels_beat_actions_alone(trained):
    test = w.test_set()
    s = {m: w.reach_success(p, w.THINKS[m], test) for m, p in trained.items()}
    # Measured over 4 seeds: standard 0.59, ecot 0.69, dropout 0.80, s.e. ~0.01
    assert s["ecot"] > s["standard"] + 0.05
    assert s["dropout"] > s["standard"] + 0.1
    assert abs(s["scaffold"] - s["standard"]) < 0.06


def test_ecot_follows_edited_reasoning_and_dropout_does_not(trained):
    test = w.test_set()
    O, R, _, (pos, tgt, grip) = test
    other = pos[np.arange(len(O)), 1 - tgt]                # K = 2
    for m, follows in (("ecot", True), ("dropout", False)):
        end = grip + w.act(trained[m], O, True, slot=other)
        reached = np.mean(np.linalg.norm(end - other, axis=1) < w.REACH)
        assert (reached > 0.4) if follows else (reached < 0.1)


def test_without_action_noise_plain_cloning_solves_the_task():
    O, R, A, _ = w.demos(np.random.default_rng(0), 1000, noise=0.0)
    p = w.train_reasoner(O, R, A, "standard", np.random.default_rng(1))
    assert w.reach_success(p, False, w.test_set()) > 0.95


# --- Toy 2: best-of-N through a gap --------------------------------------

def test_exact_pass_rate_matches_sampling():
    gap, z = w.gap_problems(np.random.default_rng(0), 4000)
    _, ok, _ = w.best_of_n(np.random.default_rng(1), gap, z, "small", 50,
                           "perfect")
    p = w.p_through(z, "small")
    assert abs(ok.mean() - p.mean()) < 4 * np.sqrt(p.mean() / (4000 * 50))


def test_search_wins_easy_problems_and_size_wins_hard_ones():
    curves, _ = w.search_curves(n=20000)
    small, big = curves["small", "perfect"], curves["big", "perfect"]
    assert small[0]["easy"] > big[0]["easy"] + 0.1          # budget 4
    assert all(b["hard"] > s["hard"] + 0.05 for s, b in zip(small, big))


def test_learned_verifier_saturates_at_its_cap():
    curves, _ = w.search_curves(n=20000)
    easy = [r["easy"] for r in curves["small", "learned"]]
    assert abs(easy[-1] - w.verifier_cap()) < 0.01


# --- Toy 3: GRPO ---------------------------------------------------------

def test_group_advantages():
    a = w.grpo_advantages(np.array([1.0, 0, 0, 1, 0, 0, 0, 0]))
    assert np.isclose(a.mean(), 0) and np.isclose(a.std(), 1)
    assert np.all(w.grpo_advantages(np.zeros(8)) == 0)
    assert np.all(w.grpo_advantages(np.ones(8)) == 0)


def test_grpo_learns_to_think_longer_on_harder_problems():
    acc, length = w.grpo_train(0.3, np.random.default_rng(0))
    assert (acc[-1] > 0.85).all()
    assert length[-1][0] < length[-1][1] < length[-1][2]
    assert length[-1][2] > length[0][2] + 4


def test_grpo_never_starts_on_problems_it_never_solves():
    acc, _ = w.grpo_train(0.8, np.random.default_rng(0))
    assert w.base_success(0.8)["hard"] < 0.002
    assert acc[-1][2] < 0.05


def test_elo_rule():
    assert abs(w.elo_win(100) - 0.64) < 0.005
    assert np.isclose(w.elo_win(0), 0.5)

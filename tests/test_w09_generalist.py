import numpy as np

from rlc.toys import w09_generalist as w


# --- The shared task ----------------------------------------------------

def test_expert_reaches_every_goal_without_touching_the_post():
    x, g = w.sample_tasks(np.random.default_rng(0), 500)
    path, touched, t_goal = w.run_policy(w.expert_velocity, np.eye(2), x, g)
    assert not touched.any() and (t_goal < w.T_RUN).all()
    closest = np.linalg.norm(path - w.POST, axis=2).min()
    assert closest > w.R_POST + 0.1


def test_expert_command_gives_the_same_motion_on_every_robot():
    x, g = w.sample_tasks(np.random.default_rng(1), 50)
    v = w.expert_velocity(x, g)
    for J in list(w.ROBOTS.values()) + [w.NEW_ROBOT]:
        a = w.expert_command(J, x, g)
        assert a.shape == (50, J.shape[1])
        assert np.allclose(a @ J.T, v)


def test_masked_entries_get_no_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 3))
    Y = np.c_[X.sum(1), np.full(64, 1e6)]       # second column is junk
    M = np.c_[np.ones(64), np.zeros(64)]
    from rlc import nn
    p0 = nn.init_mlp([3, 8, 2], rng)
    p = w.fit_masked(p0, X, Y, M, 300, rng)
    # The junk column's output weights never move
    assert np.allclose(p[-1][0][:, 1], p0[-1][0][:, 1])
    assert np.abs(nn.predict(p, X)[:, 0] - Y[:, 0]).mean() < 0.3


def test_head_slices_cover_every_command():
    cols = w.head_slices()
    assert [cols[k].stop - cols[k].start for k in w.ROBOTS] == [2, 2, 3, 4]
    assert cols["four motors"].stop == 11


# --- Toy 1: pooling -----------------------------------------------------

def test_pooled_heads_help_the_robot_with_least_data():
    out = w.compare_pooling(0)
    assert out["heads", "four motors"] > out["specialist", "four motors"] + 0.1
    assert out["padded-cue", "turned"] < out["padded+cue", "turned"] - 0.1


def test_pooled_features_help_most_with_few_demos():
    # Single seeds vary a lot with 1-4 demos, so average two
    runs = [w.new_robot_curve(s, counts=(1, 4, 32)) for s in (0, 1)]
    gap = {n: np.mean([r["pooled", n] - r["scratch", n] for r in runs])
           for n in (1, 4, 32)}
    assert gap[1] > 0.1 and gap[4] > 0.1 and abs(gap[32]) < 0.05


# --- Toy 2: simulator evaluation ----------------------------------------

def test_mmrv_counts_only_misordered_pairs():
    real = np.array([0.2, 0.5, 0.9])
    assert w.mmrv(real * 0.5, real) == 0.0         # same order, other scale
    # Swap the first two: each loses 0.3, the third is never misordered
    assert np.isclose(w.mmrv([0.5, 0.2, 0.9], real), 0.2)


def test_sysid_recovers_gain_and_noise():
    k, s = w.sysid(4000, np.random.default_rng(0))
    assert abs(k - 1.0) < 0.03 and abs(s - w.REAL_NOISE) < 0.02


# --- Toy 3: data pipeline -----------------------------------------------

def test_packing_positions_and_mask():
    cu, seg, pos = w.pack([3, 2, 4])
    assert cu.tolist() == [0, 3, 5, 9]
    assert pos.tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]
    assert seg.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert w.padding_cost([3, 2, 4]) == (9, 12)
    err = w.packing_check([26, 26, 58, 58])
    assert err["block-diagonal"] < 1e-12 < 0.1 < err["no mask"]


def test_bigger_shuffle_buffer_mixes_more_trajectories():
    d = [w.distinct_per_batch(w.shuffle_buffer_batches(
        500, 50, b, 32, 50, np.random.default_rng(0)))
        for b in (64, 1024, 8192)]
    assert d[0] < d[1] < d[2] <= 32


# --- Worked example: corridor -------------------------------------------

def test_corridor_advantages_separate_forward_from_back():
    c = w.corridor(10, 0.8)
    assert (c["adv_fwd"] > 0).all() and (c["adv_back"] <= -1 + 1e-9).all()
    # A biased walk needs L / (2p - 1) steps far from the wall; the wall
    # at 0 saves a little
    assert 10 < c["steps_bc"] < 10 / 0.6 + 1e-9

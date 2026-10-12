"""Tests for rlc.toys.w01_intro: the behaviour the Week 1 page relies on."""

import numpy as np

from rlc.toys import w01_intro as m


def test_crash_detection():
    door = np.array([m.WALL_X - 0.1, m.DOOR_Y])
    assert not m.crashes(door, door + [0.2, 0.0])          # through the door
    assert m.crashes(door + [0, 1.0], door + [0.2, 1.0])   # into the wall
    assert not m.crashes([0, 0], [1, 0])                   # nowhere near


def test_plan_reaches_the_goal_from_an_exact_estimate():
    moves = m.plan(m.START)
    assert np.allclose(m.START + moves.sum(axis=0), m.GOAL)
    assert np.linalg.norm(moves, axis=1).max() <= m.STEP + 1e-9


def test_closed_loop_beats_open_loop_under_sensing_noise():
    o = m.success_rate("open", 0.2, n=200)
    c = m.success_rate("closed", 0.2, n=200)
    assert c["goal"] > 0.95 and o["goal"] < 0.7
    assert abs(sum(o.values()) - 1) < 1e-9


def test_tokens_round_trip_within_half_a_bin():
    lo, hi, bins = -1.0, 1.0, 256
    a = np.random.default_rng(0).uniform(lo, hi, 1000)
    k = m.tokenize(a, lo, hi, bins)
    assert k.min() >= 0 and k.max() <= bins - 1
    assert np.abs(m.detokenize(k, lo, hi, bins) - a).max() <= (hi - lo) / bins / 2
    assert m.tokenize(5.0, lo, hi, bins) == bins - 1         # clipped


def test_closed_loop_replay_stays_within_one_rounding():
    path = m.reach_path()
    lo, hi = m.action_range(np.diff(path, axis=0))
    for bins in (16, 64, 256):
        half_diag = np.linalg.norm((hi - lo) / bins) / 2
        assert m.replay_error(path, bins, closed_loop=True) <= half_diag * 1.01
        assert m.replay_error(path, bins) > 2 * m.replay_error(path, bins, True)


def test_reading_years():
    assert abs(m.reading_years(m.SECONDS_PER_YEAR, 1.0) - 1.0) < 1e-12

"""Week 4 hero animation: value iteration's ripple, then acting on it.

Data are the Week 4 page's warehouse grid (rlc.toys.w04_rl) with slip
probability 0.2 and gamma 0.95: the value functions of every sweep of value
iteration, the final greedy policy, and one sampled episode. The episode is
the first seed (0, 1, 2, ...) whose run slips at least once and still reaches
the goal, so the clip shows a recovery; the rule is stated here, not chosen
by eye.

Beats:
1. The grid: walls (black), a row of hazards (orange), the start, the goal.
2. Sweeps 1 to 12: blue value spreads backwards from the goal, one cell per
   sweep, with a sweep counter.
3. A jump to the converged values (the long tail of tiny changes).
4. Greedy arrows appear: the route goes round the top, away from the hazards.
5. A robot dot plays one episode with slips and reaches the goal.
6. Values and arrows fade; the grid is back to its first frame (loop).
"""

import numpy as np
from manim import (DOWN, UP, Dot, FadeIn, FadeOut, ManimColor, Scene, Square, VGroup,
                   interpolate_color, linear)

from rlc import anim_style as st
from rlc.toys import w04_rl as w4

SLIP, GAMMA = 0.2, 0.95
CELL = 1.05                                  # scene units per grid cell
N_RIPPLE = 12
POSTER_TIME = 12.5                           # arrows shown, before the episode


def centre(r, c) -> np.ndarray:
    n_r, n_c = len(w4.LAYOUT), len(w4.LAYOUT[0])
    return np.array([(c - (n_c - 1) / 2) * CELL, ((n_r - 1) / 2 - r) * CELL - 0.35, 0.0])


def episode_with_recovery(pol):
    """First seed whose episode slips at least once and reaches the goal."""
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        path = w4.rollout_grid(pol, SLIP, rng)
        slipped = any(b != w4.move(a, pol[a]) for a, b in zip(path, path[1:]))
        if slipped and path[-1] == w4.GOAL:
            return seed, path
    raise RuntimeError("no such episode")


class Ripple(Scene):
    def construct(self):
        pal = st.palette("light")
        self.camera.background_color = pal["bg"]
        white, blue = ManimColor("#ffffff"), ManimColor(pal["blue"])

        P, R = w4.transition_model(SLIP)
        history = w4.value_iteration(P, R, GAMMA)
        V_final = history[-1]
        pol = w4.greedy(P, R, V_final, GAMMA)
        _, path = episode_with_recovery(pol)

        squares = {}
        board = VGroup()
        for r, row in enumerate(w4.LAYOUT):
            for c, ch in enumerate(row):
                sq = Square(side_length=CELL, stroke_color=pal["muted"], stroke_width=1.5)
                sq.move_to(centre(r, c))
                if ch == "#":
                    sq.set_fill(pal["ink"], 1.0)
                elif (r, c) in w4.INDEX and w4.INDEX[(r, c)] in w4.HAZARDS:
                    sq.set_fill(pal["orange"], 0.85)
                else:
                    sq.set_fill(white, 1.0)
                    squares[w4.INDEX[(r, c)]] = sq
                board.add(sq)
        marks = VGroup(st.label("G", pal["ink"], 34).move_to(centre(*w4.CELLS[w4.GOAL])),
                       st.label("S", pal["ink"], 26).move_to(centre(*w4.CELLS[w4.START])
                                                              + np.array([-0.32, 0.32, 0])))

        def shade(V):
            return [squares[s].animate.set_fill(interpolate_color(white, blue, float(np.clip(V[s], 0, 1))), 1.0)
                    for s in squares if s != w4.GOAL]

        def caption(text):
            c = st.label(text, pal["ink"], 28).to_edge(UP, buff=0.3)
            c.add_background_rectangle(color=pal["bg"], opacity=1.0, buff=0.15)
            return c.set_z_index(5)

        def swap(old, new, t=0.3):
            self.play(FadeOut(old), run_time=t)
            self.play(FadeIn(new), run_time=t)

        # Beat 1: the grid.
        self.add(board, marks)
        cap = caption("Value iteration on a slippery floor")
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(1.0)

        # Beat 2: the ripple, one sweep at a time.
        counter = st.label("sweep 0", pal["ink"], 24).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(counter), run_time=0.3)
        cap2 = caption("Each sweep pushes value one cell back from the goal")
        swap(cap, cap2)
        for k in range(1, N_RIPPLE + 1):
            # Swap the counter text instantly; morphing digits garbles them.
            counter.become(st.label(f"sweep {k}", pal["ink"], 24).to_edge(DOWN, buff=0.3))
            self.play(*shade(history[k]), run_time=0.45)

        # Beat 3: jump to convergence.
        final_counter = st.label(f"sweep {len(history) - 1}", pal["ink"], 24).to_edge(DOWN, buff=0.3)
        cap3 = caption(f"... and after {len(history) - 1} sweeps the values settle")
        swap(cap2, cap3)
        counter.become(final_counter)
        self.play(*shade(V_final), run_time=1.0)
        self.wait(0.6)

        # Beat 4: the greedy arrows.
        arrows = VGroup(*[st.label(w4.ARROWS[pol[s]], pal["ink"], 34).move_to(centre(*w4.CELLS[s]))
                          for s in squares if s != w4.GOAL])
        cap4 = caption("Act greedily: the long way round, away from the hazards")
        swap(cap3, cap4)
        self.play(FadeIn(arrows), run_time=0.8)
        self.wait(1.4)

        # Beat 5: one episode, with a slip.
        robot = Dot(centre(*w4.CELLS[path[0]]), radius=0.17, color=pal["orange"]).set_z_index(4)
        cap5 = caption("One episode: it slips, and the arrows bring it back")
        swap(cap4, cap5)
        self.play(FadeIn(robot), run_time=0.3)
        for s in path[1:]:
            self.play(robot.animate.move_to(centre(*w4.CELLS[s])), run_time=0.35, rate_func=linear)
        self.wait(1.0)

        # Beat 6: back to the first frame.
        self.play(FadeOut(robot), FadeOut(arrows), FadeOut(counter), FadeOut(cap5),
                  *[squares[s].animate.set_fill(white, 1.0) for s in squares], run_time=1.0)
        self.add(cap)
        self.play(FadeIn(cap), run_time=0.01)
        self.wait(0.5)

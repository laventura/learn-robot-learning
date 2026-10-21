"""Week 7 hero animation: one noisy run executed three ways, single steps, open-loop
chunks, and chunks with a temporal ensemble.

Data are the Week 7 page's chunk toy (rlc.toys.w07_sequence): seed 0, a fresh
generator per scheme, k = 25, m = 0, exactly the runs the page quotes. Paths and
planned chunks come from run_chunks(keep=True); nothing is hand-typed.

Beats:
1. Three empty panels, one per scheme, sharing a time axis.
2. Time runs to the goal's move: each panel draws its path, with the goal (dashed)
   revealed as time passes. The chunk panels show their planned chunks faintly:
   one every 25 steps (middle), one every step (bottom).
3. The goal moves; time runs on. The open-loop path holds its course until its
   next query, the ensemble turns gradually, single steps turn at once.
4. A closing caption on what the ensemble buys.
5. Everything fades out, back to the empty panels (loop).
"""

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, DashedVMobject, FadeIn, FadeOut, Line, ManimColor, Scene,
                   ValueTracker, VGroup, VMobject, always_redraw, linear)

from rlc import anim_style as st
from rlc.toys import w07_sequence as w7

K = 25
X0, WIDTH = -5.6, 11.6                         # scene x of t = 0, and of the whole run
SY, Y_TOP = 1.25, 1.15                         # scene units per position unit; top of a panel's range
BOTTOMS = (1.1, -1.15, -3.4)                   # scene y of position 0 in each panel
POSTER_TIME = 20.0                             # on the result caption, before the fade


def tx(t):
    return X0 + WIDTH * np.asarray(t) / w7.T_RUN


def polyline(ts, xs, base, color, width, opacity=1.0):
    m = VMobject(stroke_color=ManimColor(color), stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners([np.array([tx(t), base + SY * x, 0.0]) for t, x in zip(ts, xs)])
    return m


class Chunks(Scene):
    def construct(self):
        pal = st.palette("light")
        self.camera.background_color = pal["bg"]
        schemes = (("single step", 1, "chunk", pal["orange"]),
                   (f"chunks of {K}, open loop", K, "chunk", pal["blue"]),
                   (f"chunks of {K}, temporal ensemble", K, "ensemble", pal["green"]))
        runs = [w7.run_chunks(k, mode, np.random.default_rng(0), keep=True) for _, k, mode, _ in schemes]
        goal = np.where(np.arange(w7.T_RUN + 1) < w7.T_MOVE, w7.GOAL_A, w7.GOAL_B)

        # Beat 1: panel axes, names and second ticks.
        frame = VGroup()
        for (name, _, _, color), base in zip(schemes, BOTTOMS):
            frame.add(Line([tx(0), base, 0], [tx(w7.T_RUN), base, 0], stroke_color=pal["muted"], stroke_width=1.5))
            frame.add(st.label(name, color, 22).move_to([tx(w7.T_RUN), base + SY * Y_TOP + 0.05, 0], aligned_edge=RIGHT))
        for s in range(4):
            frame.add(st.label(f"{s} s", pal["muted"], 18).next_to([tx(s * w7.HZ), BOTTOMS[-1], 0], DOWN, buff=0.12))
        T = ValueTracker(0.0)

        def panel(i):
            (x, plans), base = runs[i], BOTTOMS[i]
            now = T.get_value()
            n = int(now)
            g = VGroup()
            if n >= 1:
                ts = np.arange(n + 1)
                g.add(DashedVMobject(polyline(ts, goal[:n + 1], base, pal["muted"], 2), num_dashes=max(2, n // 3)))
                if schemes[i][1] > 1:                      # planned chunks still running, faint
                    for t0, plan in plans:
                        if t0 <= n < t0 + K:
                            age = (n - t0) / K
                            end = min(K, w7.T_RUN - t0)       # stop at the panel's right edge
                            g.add(polyline(np.arange(t0, t0 + end + 1), np.r_[x[t0], plan[:end]], base,
                                           pal["muted"], 2, opacity=0.55 * (1 - age) + 0.1))
                g.add(polyline(ts, x[:n + 1], base, schemes[i][3], 4))
            return g

        panels = [always_redraw(lambda i=i: panel(i)) for i in range(3)]
        cursor = always_redraw(lambda: Line([tx(T.get_value()), BOTTOMS[-1], 0],
                                            [tx(T.get_value()), BOTTOMS[0] + SY * Y_TOP, 0],
                                            stroke_color=pal["muted"], stroke_width=1, stroke_opacity=0.6))

        def caption(text):
            c = st.label(text, pal["ink"], 28).to_edge(UP, buff=0.25)
            c.add_background_rectangle(color=pal["bg"], opacity=1.0, buff=0.15)
            return c.set_z_index(5)

        cap = caption("One run, three ways to execute it")
        self.add(frame, *panels, cursor)
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(1.0)

        # Beat 2: up to the move.
        cap2 = caption("Each query sees the goal through a noisy camera")
        # Captions swap in two steps; a cross-fade overlaps the two texts.
        self.play(FadeOut(cap), run_time=0.2)
        self.play(FadeIn(cap2), run_time=0.2)
        self.play(T.animate.set_value(w7.T_MOVE), run_time=7.0, rate_func=linear)

        # Beat 3: the move, and after it.
        cap3 = caption("The goal moves: open-loop chunks keep going until their next query")
        self.play(FadeOut(cap2), run_time=0.2)
        self.play(FadeIn(cap3), run_time=0.2)
        self.wait(st.PAUSE)
        self.play(T.animate.set_value(w7.T_RUN), run_time=8.0, rate_func=linear)

        # Beat 4: the result.
        cap4 = caption("Ensembled chunks: smooth, close to the goal, slower to turn")
        self.play(FadeOut(cap3), run_time=0.2)
        self.play(FadeIn(cap4), run_time=0.2)
        self.wait(2.5)

        # Beat 5: back to empty panels.
        self.play(FadeOut(cap4), *[FadeOut(p) for p in panels], FadeOut(cursor), run_time=0.8)
        T.set_value(0.0)
        self.add(*panels, cursor)
        cap0 = caption("One run, three ways to execute it")
        self.play(FadeIn(cap0), run_time=0.4)
        self.wait(0.4)

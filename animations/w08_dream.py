"""Week 8 hero animation: a plan that works in the world model, the same pushes on
the real table, and a run that looks at the real puck before every replan.

Data are the Week 8 page's puck toy (rlc.toys.w08_world): the world model fitted
to 1,000 random transitions with seed 0, and run_mpc with a fresh generator seeded
10, once with look=False (the dream and its replay) and once with look=True.
These are exactly the runs of the page's table figure; nothing is hand-typed.

Beats:
1. The table seen from above: the hill (grey rings), the puck's start and the goal.
2. The dream: the planner replans from its own predictions only; a blue puck
   draws a dashed path round the hill to the goal.
3. Reality: the same pushes replayed on the real table; an orange puck drifts off.
4. Looking: replanning from the real puck; a green puck reaches the goal.
5. The paths fade, back to the empty table (loop).
"""

import numpy as np
from manim import (Circle, Cross, DashedVMobject, Dot, FadeIn, FadeOut, ManimColor, Scene,
                   ValueTracker, VGroup, VMobject, always_redraw, linear, RIGHT, UP)

from rlc import anim_style as st
from rlc.toys import w08_world as w8

SC = 2.7                                     # scene units per table unit
Y_OFF = -0.35                                # table centre, below the caption
POSTER_TIME = 19.8                           # all three paths drawn, before the fade (20.6 s)


def to_scene(p):
    return np.array([SC * p[0], SC * p[1] + Y_OFF, 0.0])


class Dream(Scene):
    def construct(self):
        pal = st.palette("light")
        self.camera.background_color = pal["bg"]
        model = w8.fit_world_model(1000, 0)
        step = lambda x, a: w8.model_step(model, x, a)
        blind, imag = w8.run_mpc(step, np.random.default_rng(10), look=False)
        look, _ = w8.run_mpc(step, np.random.default_rng(10))

        # Beat 1: hill rings, start and goal.
        rings = VGroup(*[Circle(radius=SC * w8.HILL_W * r, stroke_width=0, fill_color=ManimColor(pal["muted"]),
                                fill_opacity=0.16).move_to(to_scene(w8.HILL_AT)) for r in (2.2, 1.6, 1.0, 0.5)])
        start = Dot(to_scene(w8.START), radius=0.08, color=ManimColor(pal["ink"]))
        goal = Cross(stroke_color=ManimColor(pal["ink"]), stroke_width=5, scale_factor=0.18).move_to(to_scene(w8.GOAL))
        labels = VGroup(st.label("start", pal["ink"], 22).next_to(start, UP, buff=0.15),
                        st.label("goal", pal["ink"], 22).next_to(goal, RIGHT, buff=0.2),
                        st.label("hill", pal["muted"], 22).move_to(to_scene(w8.HILL_AT)))

        def caption(text):
            c = st.label(text, pal["ink"], 28).to_edge(UP, buff=0.3)
            c.add_background_rectangle(color=pal["bg"], opacity=1.0, buff=0.15)
            return c.set_z_index(5)

        def drawn(path, color, dashed, T):
            """A path drawn up to step T, and the puck at its head."""
            def build():
                n = max(1, int(T.get_value()))
                line = VMobject(stroke_color=ManimColor(color), stroke_width=5)
                line.set_points_as_corners([to_scene(p) for p in path[:n + 1, :2]])
                line = DashedVMobject(line, num_dashes=max(3, n // 2)) if dashed else line
                puck = Dot(to_scene(path[n, :2]), radius=0.11, color=ManimColor(color))
                return VGroup(line, puck)
            return always_redraw(build)

        cap = caption("A world model of the table, learned from 1,000 random pushes")
        self.add(rings, start, goal, labels)
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(1.5)

        shown = []
        for path, color, dashed, text in (
                (imag, pal["blue"], True, "Planning only in its own model: the dream reaches the goal"),
                (blind, pal["orange"], False, "The same pushes on the real table drift off"),
                (look, pal["green"], False, "Replanning from the real puck reaches the goal")):
            new = caption(text)
            self.play(FadeOut(cap), run_time=0.2)
            self.play(FadeIn(new), run_time=0.2)
            cap = new
            T = ValueTracker(0.0)
            m = drawn(path, color, dashed, T)
            self.add(m)
            self.play(T.animate.set_value(len(path) - 1), run_time=4.5, rate_func=linear)
            m.clear_updaters()
            shown.append(m)
            self.wait(st.PAUSE + 0.3)

        self.wait(1.5)
        self.play(FadeOut(cap), *[FadeOut(m) for m in shown], run_time=0.8)
        cap0 = caption("A world model of the table, learned from 1,000 random pushes")
        self.play(FadeIn(cap0), run_time=0.4)
        self.wait(0.6)

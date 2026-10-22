"""Week 9 hero animation: the robot with the least data, driven by its own
specialist and then by one network pooled over four robots.

Data are the Week 9 page's first toy (rlc.toys.w09_generalist): the policies of
pooling_policies(0), run on every 15th of the held-out tasks test_tasks(0)
(20 tasks). Seed 0 is the seed whose pooling gain for four motors (over 300
tasks) is closest to the mean over the page's 8 seeds. The page recomputes the
counts quoted in the captions.

Beats:
1. The table from above: the post, 20 starts on the left and their goals
   (crosses) on the right; beside it the four robots and how many demos each has.
2. Four motors on its own: its specialist, trained on its 3 demos, drives the 20
   tasks (orange); runs that touch the post stop with a dark square.
3. Pooling: all four robots' demos train one network with a head per robot.
4. The same 20 tasks with the pooled network's four-motors head (green).
5. The paths fade, back to the empty table (loop).
"""

import numpy as np
from manim import (Circle, Cross, Dot, FadeIn, FadeOut, ManimColor, Rectangle, Scene,
                   Square, ValueTracker, VGroup, VMobject, always_redraw, linear, LEFT, RIGHT, UP)

from rlc import anim_style as st
from rlc.toys import w09_generalist as w9

SC = 2.6                                     # scene units per table unit
CENTRE = np.array([-2.1, -0.4, 0.0])         # table centre in the scene
POSTER_TIME = 19.8                           # pooled runs drawn, before the fade
ROBOT = "four motors"


def to_scene(p):
    return CENTRE + np.array([SC * p[0], SC * p[1], 0.0])


def runs(act, x, g):
    """Each run's path up to where it ends: the goal, the post, or the
    time limit; and how it ended."""
    path, _, _ = w9.run_policy(act, w9.ROBOTS[ROBOT], x, g)
    out = []
    for i in range(len(x)):
        p = path[:, i]
        hit = np.linalg.norm(p - w9.POST, axis=1) < w9.R_POST
        home = np.linalg.norm(p - g[i], axis=1) < w9.TOL
        stops = [int(np.argmax(m)) for m in (hit, home) if m.any()]
        end = min(stops) if stops else len(p) - 1
        kind = "post" if hit.any() and end == int(np.argmax(hit)) else (
            "goal" if home.any() else "time")
        out.append((p[:end + 1], kind))
    return out


class Pool(Scene):
    def construct(self):
        pal = st.palette("light")
        self.camera.background_color = pal["bg"]
        pols = w9.pooling_policies(0)
        x, g = w9.test_tasks(0)
        x, g = x[::15], g[::15]
        alone = runs(pols["specialist", ROBOT], x, g)
        pooled = runs(pols["heads", ROBOT], x, g)

        # Beat 1: table, post, tasks; robots and their demos.
        post = Circle(radius=SC * w9.R_POST, stroke_width=0, fill_opacity=0.35,
                      fill_color=ManimColor(pal["muted"])).move_to(to_scene(w9.POST))
        starts = VGroup(*[Dot(to_scene(p), radius=0.05, color=ManimColor(pal["ink"])) for p in x])
        goals = VGroup(*[Cross(stroke_color=ManimColor(pal["ink"]), stroke_width=3,
                               scale_factor=0.08).move_to(to_scene(p)) for p in g])
        post_label = st.label("post", pal["ink"], 20).move_to(to_scene(w9.POST))
        bars, unit = VGroup(), 0.065
        for i, (name, n) in enumerate(w9.EPISODES.items()):
            y = 1.3 - 0.75 * i
            name_t = st.label(name, pal["ink"], 22).move_to([2.0, y, 0], aligned_edge=LEFT)
            bar = Rectangle(width=max(unit * n, 0.04), height=0.28, stroke_width=0,
                            fill_opacity=1.0, fill_color=ManimColor(pal["blue"]))
            bar.move_to([2.0, y - 0.32, 0], aligned_edge=LEFT)
            count = st.label(f"{n} demos", pal["muted"], 20).next_to(bar, RIGHT, buff=0.12)
            bars.add(VGroup(name_t, bar, count))

        def caption(text):
            c = st.label(text, pal["ink"], 28).to_edge(UP, buff=0.3)
            c.add_background_rectangle(color=pal["bg"], opacity=1.0, buff=0.15)
            return c.set_z_index(5)

        def drawn(paths, color, T):
            """All runs drawn up to step T; a mark where a run hit the post."""
            def build():
                n = max(1, int(T.get_value()))
                grp = VGroup()
                for p, kind in paths:
                    k = min(n, len(p) - 1)
                    line = VMobject(stroke_color=ManimColor(color), stroke_width=4)
                    line.set_points_as_corners([to_scene(q) for q in p[:k + 1]])
                    grp.add(line)
                    if kind == "post" and k == len(p) - 1:
                        grp.add(Square(side_length=0.15, stroke_width=0, fill_opacity=1.0,
                                       fill_color=ManimColor(pal["ink"])).move_to(to_scene(p[-1])))
                    else:
                        grp.add(Dot(to_scene(p[k]), radius=0.06, color=ManimColor(color)))
                return grp
            return always_redraw(build)

        n_alone = sum(k == "goal" for _, k in alone)
        n_pooled = sum(k == "goal" for _, k in pooled)
        cap = caption("Four robots, one task: drive round the post to the goal")
        self.add(post, starts, goals, post_label, bars)
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(2.0)

        # Beat 2: four motors alone.
        others = VGroup(*bars[:3])
        new = caption(f"Four motors alone, 3 demos: {n_alone} of {len(x)} runs reach the goal")
        self.play(FadeOut(cap), others.animate.set_opacity(0.25), run_time=0.4)
        self.play(FadeIn(new), run_time=0.3)
        cap = new
        T = ValueTracker(0.0)
        m1 = drawn(alone, pal["orange"], T)
        self.add(m1)
        self.play(T.animate.set_value(w9.T_RUN), run_time=5.0, rate_func=linear)
        m1.clear_updaters()
        self.wait(st.PAUSE + 1.0)

        # Beat 3: pool the data.
        new = caption("Pool all four robots' demos: one network, a head per robot")
        self.play(FadeOut(cap), m1.animate.set_opacity(0.25), others.animate.set_opacity(1.0),
                  run_time=0.5)
        self.play(FadeIn(new), *[b[1].animate.set_fill(ManimColor(pal["green"])) for b in bars],
                  run_time=0.8)
        cap = new
        self.wait(1.8)

        # Beat 4: the pooled network drives four motors.
        new = caption(f"Four motors, pooled network: {n_pooled} of {len(x)} runs reach the goal")
        self.play(FadeOut(cap), FadeOut(m1), others.animate.set_opacity(0.25), run_time=0.4)
        self.play(FadeIn(new), run_time=0.3)
        cap = new
        T2 = ValueTracker(0.0)
        m2 = drawn(pooled, pal["green"], T2)
        self.add(m2)
        self.play(T2.animate.set_value(w9.T_RUN), run_time=5.0, rate_func=linear)
        m2.clear_updaters()
        self.wait(st.PAUSE + 1.5)

        # Beat 5: back to the start.
        # One animation per bar part, so fill colour and opacity do not fight
        self.play(FadeOut(cap), FadeOut(m2),
                  *[b[i].animate.set_opacity(1.0) for b in others for i in (0, 2)],
                  *[b[1].animate.set_fill(ManimColor(pal["blue"]), opacity=1.0) for b in bars],
                  run_time=0.8)
        cap0 = caption("Four robots, one task: drive round the post to the goal")
        self.play(FadeIn(cap0), run_time=0.4)
        self.wait(0.6)

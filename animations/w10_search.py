"""Week 10 hero animation: best-of-N search through a gap, on an easy and a
hard problem.

Data are the Week 10 page's second toy (rlc.toys.w10_reasoning): the runs of
hero_runs(), which samples the small planner's paths for the typical easy
problem (16 samples) and the typical hard problem (16, then 64 samples). A
typical problem is the one whose single-sample success is closest to its
difficulty bin's mean. The page recomputes the counts quoted in the captions.

Beats:
1. The wall with its gap, the start on the left and the goal on the right.
2. Easy problem: 16 sampled paths grow to the wall; those through the gap
   turn green, the others orange; the verifier's pick continues to the goal.
3. The gap moves to the hard problem's position; 16 paths grow and all hit
   the wall.
4. 64 paths: they still cluster beside the gap; none gets through.
5. The paths fade and the gap returns to the easy position (loop).
"""

import numpy as np
from manim import (Create, Dot, FadeIn, FadeOut, Line, ManimColor, Scene, ValueTracker,
                   VGroup, always_redraw, LEFT, UP)

from rlc import anim_style as st
from rlc.toys import w10_reasoning as w

SX, SY = 3.6, 1.9                       # scene units per toy unit (x, y)
CENTRE = np.array([0.0, -0.45, 0.0])
Y_WALL = 1.6                            # the wall runs from -Y_WALL to Y_WALL
POSTER_TIME = 6.6                       # easy problem solved, before the move


def at(x, y):
    return CENTRE + np.array([SX * x, SY * y, 0.0])


class Search(Scene):
    def construct(self):
        pal = st.palette("light")
        self.camera.background_color = pal["bg"]
        runs = w.hero_runs()
        half = w.GAP_W / 2
        gap = ValueTracker(runs[0]["gap"])

        def wall():
            g = gap.get_value()
            kw = dict(stroke_width=10, color=ManimColor(pal["muted"]))
            return VGroup(Line(at(0, -Y_WALL), at(0, g - half), **kw),
                          Line(at(0, g + half), at(0, Y_WALL), **kw))

        walls = always_redraw(wall)
        start = Dot(at(-1, 0), radius=0.09, color=ManimColor(pal["ink"]))
        goal = Dot(at(1, 0), radius=0.09, color=ManimColor(pal["ink"]))
        tags = VGroup(st.label("start", pal["ink"], 22).next_to(start, LEFT, buff=0.15),
                      st.label("goal", pal["ink"], 22).next_to(goal, LEFT * -1, buff=0.15))

        def caption(text):
            c = st.label(text, pal["ink"], 28).to_edge(UP, buff=0.3)
            c.add_background_rectangle(color=pal["bg"], opacity=1.0, buff=0.15)
            return c.set_z_index(5)

        def samples(run, width):
            """Grey paths from the start to each sample's crossing point."""
            return VGroup(*[Line(at(-1, 0), at(0, y), stroke_width=width,
                                 color=ManimColor(pal["muted"])) for y in run["cand"]])

        def judge(paths, run):
            """Colour each path by whether it gets through the gap."""
            return [p.animate.set_color(ManimColor(pal["green"] if ok else pal["orange"]))
                    for p, ok in zip(paths, run["ok"])]

        # Beat 1: the wall, start and goal.
        cap = caption("A planner samples paths; a verifier keeps one that gets through")
        self.add(walls, start, goal, tags, cap)        # as the loop ends
        self.wait(2.0)

        # Beat 2: easy problem, 16 samples.
        easy = runs[0]
        paths = samples(easy, 2.5)
        self.play(*[Create(p) for p in paths], run_time=1.5)
        self.play(*judge(paths, easy), run_time=0.6)
        n_ok = int(easy["ok"].sum())
        new = caption(f"Easy problem: {n_ok} of {easy['N']} samples get through")
        y = easy["cand"][easy["pick"]]
        chosen = VGroup(Line(at(-1, 0), at(0, y), stroke_width=7, color=ManimColor(pal["green"])),
                        Line(at(0, y), at(1, 0), stroke_width=7, color=ManimColor(pal["green"])))
        self.play(FadeOut(cap), FadeIn(new), paths.animate.set_opacity(0.45), run_time=0.5)
        cap = new
        self.play(Create(chosen), run_time=1.2)
        self.wait(st.PAUSE + 1.5)

        # Beat 3: hard problem, 16 samples.
        hard = runs[1]
        new = caption("Hard problem: the planner is wrong about where the gap is")
        self.play(FadeOut(paths), FadeOut(chosen), FadeOut(cap), run_time=0.5)
        self.play(FadeIn(new), gap.animate.set_value(hard["gap"]), run_time=1.0)
        cap = new
        paths = samples(hard, 2.5)
        self.play(*[Create(p) for p in paths], run_time=1.5)
        self.play(*judge(paths, hard), run_time=0.6)
        new = caption(f"Hard problem: {int(hard['ok'].sum())} of {hard['N']} samples get through")
        self.play(FadeOut(cap), FadeIn(new), run_time=0.4)
        cap = new
        self.wait(st.PAUSE + 1.0)

        # Beat 4: hard problem, 64 samples.
        more = runs[2]
        self.play(FadeOut(paths), run_time=0.4)
        paths = samples(more, 1.6)
        self.play(*[Create(p) for p in paths], run_time=1.5)
        self.play(*judge(paths, more), run_time=0.6)
        new = caption(f"{int(more['ok'].sum())} of {more['N']}: search only picks among what the planner proposes")
        self.play(FadeOut(cap), FadeIn(new), run_time=0.4)
        cap = new
        self.wait(st.PAUSE + 2.0)

        # Beat 5: back to the start.
        cap0 = caption("A planner samples paths; a verifier keeps one that gets through")
        self.play(FadeOut(paths), FadeOut(cap), gap.animate.set_value(easy["gap"]), run_time=1.0)
        self.play(FadeIn(cap0), run_time=0.4)
        self.wait(0.6)

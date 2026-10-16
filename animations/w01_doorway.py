"""Week 1 hero animation: plan once, or look every step.

Data are the Week 1 page's doorway toy (rlc.toys.w01_intro): the first 12
of the 20 episodes per robot that the page's slider shows at a
position-estimate error of sigma = 0.2 (same seeds).

Beats:
1. The room: a wall with a doorway, the start on the left, the goal circle.
2. Twelve sense-plan-act robots (orange) walk their plans at once.
3. Their paths fan out; some hit the wall (crosses), some miss the goal.
4. Twelve closed-loop robots (blue) walk from the same start.
5. Their paths bundle through the doorway to the goal; captions give the counts.
6. Everything fades back to the empty room (loop).
"""

import numpy as np
from manim import (UP, Circle, Cross, Dot, FadeIn, FadeOut, Rectangle, Scene,
                   VGroup, VMobject, linear)

from rlc import anim_style as st
from rlc.toys import w01_intro as w1

SIGMA, N = 0.2, 12
S = 1.35                              # scene units per room unit
POSTER_TIME = 13.5                    # the result beat, before the fade


def to_scene(p) -> np.ndarray:
    return np.array([(p[0] - 4.25) * S, (p[1] - 0.25) * S - 0.3, 0.0])


class Doorway(Scene):
    def construct(self):
        pal = st.palette("light")
        self.camera.background_color = pal["bg"]

        # Episodes, computed before animating: the first N of the 20 that the
        # page's slider draws at sigma 0.2 (a fresh rng per mode, seed 102).
        runs = {}
        for m in ("open", "closed"):
            rng = np.random.default_rng(102)
            runs[m] = [w1.run(m, SIGMA, rng) for _ in range(N)]
        count = {m: sum(o == "goal" for _, o in runs[m]) for m in runs}

        wall = VGroup(*[Rectangle(width=0.12, height=abs(y1 - y0) * S, fill_color=pal["muted"],
                                  fill_opacity=1, stroke_width=0)
                        .move_to(to_scene((w1.WALL_X, (y0 + y1) / 2)))
                        for y0, y1 in ((-2.4, w1.DOOR_Y - w1.DOOR_HALF),
                                       (w1.DOOR_Y + w1.DOOR_HALF, 2.8))])
        goal = Circle(radius=w1.GOAL_TOL * S, color=pal["ink"], stroke_width=3).move_to(to_scene(w1.GOAL))
        start = Dot(to_scene(w1.START), radius=0.06, color=pal["ink"])

        def caption(text):
            c = st.label(text, pal["ink"], 28).to_edge(UP, buff=0.35)
            c.add_background_rectangle(color=pal["bg"], opacity=1.0, buff=0.15)
            return c.set_z_index(5)

        def swap(old, new):
            self.play(FadeOut(old), run_time=0.3)
            self.play(FadeIn(new), run_time=0.3)

        def walk(mode, color):
            """Animate N robots along their computed paths; mark crashes."""
            paths = [p for p, _ in runs[mode]]
            n = max(len(p) for p in paths)
            dots = VGroup(*[Dot(to_scene(p[0]), radius=0.06, color=color) for p in paths])
            lines = VGroup(*[VMobject(stroke_color=color, stroke_width=2.5, stroke_opacity=0.7)
                             .set_points_as_corners([to_scene(p[0]), to_scene(p[0])])
                             for p in paths])
            self.add(lines, dots)
            for k in range(1, n):
                anims = []
                for d, line, p in zip(dots, lines, paths):
                    if k < len(p):
                        line.set_points_as_corners([to_scene(q) for q in p[:k + 1]])
                        anims.append(d.animate.move_to(to_scene(p[k])))
                if anims:
                    self.play(*anims, run_time=0.06, rate_func=linear)
            crosses = VGroup(*[Cross(scale_factor=0.12, stroke_color=pal["ink"], stroke_width=4)
                               .move_to(to_scene(p[-1])) for p, o in runs[mode] if o == "crash"])
            if len(crosses):
                self.play(FadeIn(crosses), run_time=0.3)
            return VGroup(lines, dots, crosses)

        # Beat 1: the room.
        self.add(wall, goal, start)
        cap = caption("Reach the circle through the doorway")
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(1.0)

        # Beats 2-3: sense once, plan, walk blind.
        cap2 = caption("Sense once, plan, walk blind")
        swap(cap, cap2)
        spa = walk("open", pal["orange"])
        cap3 = caption(f"Plan once: {count['open']} of {N} reach the circle")
        swap(cap2, cap3)
        self.wait(1.2)

        # Beats 4-5: look every step.
        self.play(spa.animate.set_opacity(0.25), run_time=0.4)
        cap4 = caption("Look again at every step")
        swap(cap3, cap4)
        loop = walk("closed", pal["blue"])
        cap5 = caption(f"Look every step: {count['closed']} of {N} reach the circle")
        swap(cap4, cap5)
        self.wait(2.0)

        # Beat 6: back to the empty room.
        self.play(FadeOut(spa), FadeOut(loop), FadeOut(cap5), run_time=1.0)
        self.wait(0.4)

"""Week 11 hero animation: the lecture's stitching example in the first toy's
building.

Data are the Week 11 page's first toy (rlc.toys.w11_frontiers): the runs of
lecture_example(), one operator demonstration from A to B and one from B to C,
then one rollout each of goal-conditioned behavior cloning and of offline RL
on the task A to C. The page recomputes the step counts quoted in the captions.

Beats:
1. The building from above: walls, and the landmarks A, B and C.
2. The two demonstrations are drawn: A to B, then B to C.
3. The task A to C: behavior cloning has no demonstration from A to C and
   wanders until its 40 steps run out.
4. Offline RL follows the first demonstration, switches to the second one
   cell before B, and reaches C.
5. The paths fade, back to the empty building (loop).
"""

import numpy as np
from manim import (Create, Dot, FadeIn, FadeOut, ManimColor, Scene, Square, VGroup,
                   VMobject, UP)

from rlc import anim_style as st
from rlc.toys import w11_frontiers as w

CS = 0.62                                    # scene units per grid cell
ORIGIN = np.array([-CS * 7.5, CS * 4.2, 0])  # top-left corner of the map
POSTER_TIME = 17.5                           # offline RL has reached C


def at(cell, jitter=(0.0, 0.0)):
    r, c = cell
    return ORIGIN + np.array([CS * (c + 0.5 + jitter[0]), -CS * (r + 0.5 + jitter[1]), 0])


def cells_of(traj, start):
    return [start] + [s2 for _, _, s2 in traj]


class Stitch(Scene):
    def construct(self):
        pal = st.palette("light")
        self.camera.background_color = pal["bg"]
        demos, runs = w.lecture_example()
        L = w.LANDMARKS

        walls = VGroup(*[Square(CS, stroke_width=0, fill_opacity=1.0, fill_color=ManimColor(pal["muted"]))
                         .move_to(at((r, c))) for r, row in enumerate(w.MAP)
                         for c, ch in enumerate(row) if ch == "#"])
        marks = VGroup()
        for name in "ABC":
            marks.add(Dot(at(L[name]), radius=0.2, color=ManimColor(pal["ink"])),
                      st.label(name, pal["bg"], 22).move_to(at(L[name])))

        def path(cells, color, width, offset):
            """A polyline through cell centres, shifted slightly so that
            overlapping paths stay visible."""
            line = VMobject(stroke_color=ManimColor(color), stroke_width=width)
            line.set_points_as_corners([at(c, offset) for c in cells])
            return line

        def caption(text):
            c = st.label(text, pal["ink"], 28).to_edge(UP, buff=0.25)
            c.add_background_rectangle(color=pal["bg"], opacity=1.0, buff=0.15)
            return c.set_z_index(5)

        # Beat 1: the building.
        cap = caption("Two rooms, a corridor, and landmarks A, B, C")
        self.add(walls, marks, cap)
        self.wait(1.5)

        # Beat 2: the demonstrations.
        d1 = path(cells_of(demos[0], L["A"]), pal["blue"], 6, (-0.12, -0.12))
        d2 = path(cells_of(demos[1], L["B"]), pal["green"], 6, (0.12, 0.12))
        new = caption("The data: one demonstration A to B, one B to C")
        self.play(FadeOut(cap), FadeIn(new), run_time=0.4)
        cap = new
        self.play(Create(d1), run_time=1.6)
        self.play(Create(d2), run_time=1.6)
        self.wait(st.PAUSE + 0.8)

        # Beat 3: behavior cloning on A to C.
        bc = runs["bc"]
        new = caption("Task A to C. Cloning has no demo of it: it wanders")
        # Stroke opacity only: set_opacity would also fill the polylines
        self.play(FadeOut(cap), FadeIn(new), d1.animate.set_stroke(opacity=0.35),
                  d2.animate.set_stroke(opacity=0.35), run_time=0.5)
        cap = new
        p_bc = path(cells_of(bc[0], L["A"]), pal["orange"], 5, (0.0, 0.0))
        self.play(Create(p_bc), run_time=4.0)
        new = caption(f"Cloning: {bc[2]} steps, never reaches C")
        self.play(FadeOut(cap), FadeIn(new), run_time=0.4)
        cap = new
        self.wait(st.PAUSE + 1.0)

        # Beat 4: offline RL stitches.
        off = runs["offline"]
        new = caption("Offline RL joins the two demonstrations")
        self.play(FadeOut(cap), FadeIn(new), p_bc.animate.set_stroke(opacity=0.25), run_time=0.5)
        cap = new
        p_off = path(cells_of(off[0], L["A"]), pal["ink"], 8, (0.0, 0.0))
        self.play(Create(p_off), run_time=2.5)
        new = caption(f"Offline RL: A to C in {off[2]} steps")
        self.play(FadeOut(cap), FadeIn(new), run_time=0.4)
        cap = new
        self.wait(st.PAUSE + 2.0)

        # Beat 5: back to the empty building.
        cap0 = caption("Two rooms, a corridor, and landmarks A, B, C")
        self.play(FadeOut(VGroup(d1, d2, p_bc, p_off)), FadeOut(cap), run_time=0.8)
        self.play(FadeIn(cap0), run_time=0.4)
        self.wait(0.6)

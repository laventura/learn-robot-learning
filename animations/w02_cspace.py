"""Week 2 hero animation: an obstacle becomes a shape in joint space.

Data are the Week 2 page's arm toy (rlc.toys.w02_control): the 2R arm, the
round obstacle, the configuration-space grid of collisions, the straight
joint-space line from the start to the goal configuration, and the
breadth-first-search detour, all as on the page.

Beats:
1. The arm at its start beside the disc (left); an empty joint-angle square
   (right) with a dot at the arm's configuration.
2. The forbidden region fades into the square: every joint setting where the
   arm touches the disc.
3. Both joints turn at a steady rate towards the goal: the dot runs along a
   straight line, and the arm turns orange when it reaches the disc.
4. Back at the start, the dot follows the planned detour instead, leaving
   one edge of the square and re-entering at the opposite one, while the
   arm swings the long way round to the goal.
5. A caption compares the two paths' lengths in grid moves.
6. Everything fades back to the arm at its start (loop).
"""

import numpy as np
from manim import (DOWN, UP, Dot, FadeIn, FadeOut, ImageMobject, Line, Scene,
                   Square, ValueTracker, VGroup, VMobject, always_redraw, linear)

from rlc import anim_style as st
from rlc.toys import w02_control as w2

Q_A, Q_B = np.deg2rad([-20.0, 30.0]), np.deg2rad([110.0, -40.0])   # as on the page
WS = 1.6                               # scene units per workspace unit
WS_ORIGIN = np.array([-3.6, -0.5, 0])
CS = 5.0                               # side of the C-space square
CS_CENTER = np.array([3.4, -0.45, 0])
POSTER_TIME = 17.5                     # the result beat, before the fade


def ws(p) -> np.ndarray:
    return WS_ORIGIN + WS * np.array([p[0], p[1], 0.0])


def cs(q) -> np.ndarray:
    """Joint angles (radians, wrapped) -> point in the C-space square."""
    q = w2.wrap(np.asarray(q, dtype=float))
    return CS_CENTER + CS * np.array([q[0], q[1], 0.0]) / (2 * np.pi)


class CSpace(Scene):
    def construct(self):
        pal = st.palette("light")
        self.camera.background_color = pal["bg"]

        blocked = w2.c_obstacle()                        # (n, n): [q1, q2]
        straight = w2.straight_path(Q_A, Q_B, k=120)
        hit = w2.collides(straight)
        k_hit = int(np.argmax(hit))
        path = w2.plan_path(Q_A, Q_B, blocked)
        n_straight = int(round(np.rad2deg(np.max(np.abs(w2.wrap(Q_B - Q_A)))) / (360 / w2.GRID)))

        # The C-space image: rows = elbow angle from the top, columns = shoulder.
        rgba = np.zeros((w2.GRID, w2.GRID, 4), dtype=np.uint8)
        grey = np.array([int(pal["muted"][i:i + 2], 16) for i in (1, 3, 5)])
        rgba[blocked.T[::-1]] = np.append(grey, 200)
        image = ImageMobject(rgba).stretch_to_fit_width(CS).stretch_to_fit_height(CS)
        image.move_to(CS_CENTER).set_resampling_algorithm(0)          # nearest
        frame = Square(side_length=CS, color=pal["ink"], stroke_width=2).move_to(CS_CENTER)
        axes = VGroup(st.label("shoulder angle", pal["ink"], 22).next_to(frame, DOWN, buff=0.15),
                      st.label("elbow angle", pal["ink"], 22).rotate(np.pi / 2)
                      .next_to(frame, np.array([-1, 0, 0]), buff=0.15))
        disc = VMobject(fill_color=pal["muted"], fill_opacity=0.6, stroke_width=0)
        t = np.linspace(0, 2 * np.pi, 60)
        disc.set_points_as_corners([ws(w2.OBSTACLE + w2.OBSTACLE_R * np.array([np.cos(a), np.sin(a)]))
                                    for a in t])
        floor = VGroup(Dot(ws([0, 0]), radius=0.08, color=pal["ink"]))

        q_now = {"q": Q_A.copy(), "color": pal["ink"]}

        def arm():
            elbow, hand = w2.fk(q_now["q"])
            c = q_now["color"]
            return VGroup(Line(ws([0, 0]), ws(elbow), color=c, stroke_width=7),
                          Line(ws(elbow), ws(hand), color=c, stroke_width=7),
                          Dot(ws(elbow), radius=0.07, color=c), Dot(ws(hand), radius=0.07, color=c))

        def cdot():
            return Dot(cs(q_now["q"]), radius=0.09, color=q_now["color"]).set_z_index(3)

        live = {"arm": always_redraw(arm), "dot": always_redraw(cdot)}

        def reset_to_start():
            """Fade the arm and its dot out, move them to the start, fade in."""
            self.play(FadeOut(live["arm"]), FadeOut(live["dot"]), run_time=0.4)
            q_now["q"], q_now["color"] = Q_A.copy(), pal["ink"]
            live["arm"], live["dot"] = always_redraw(arm), always_redraw(cdot)
            self.play(FadeIn(live["arm"]), FadeIn(live["dot"]), run_time=0.4)

        def caption(text):
            c = st.label(text, pal["ink"], 28).to_edge(UP, buff=0.3)
            c.add_background_rectangle(color=pal["bg"], opacity=1.0, buff=0.15)
            return c.set_z_index(5)

        def swap(old, new):
            self.play(FadeOut(old), run_time=0.3)
            self.play(FadeIn(new), run_time=0.3)

        def trace(qs, color):
            """Polyline in the square through configurations qs, split at the seams."""
            pts = np.array([cs(q) for q in qs])
            jumps = np.flatnonzero(np.abs(np.diff(pts[:, :2], axis=0)).max(axis=1) > CS / 2) + 1
            segs = VGroup()
            for seg in np.split(pts, jumps):
                if len(seg) > 1:
                    segs.add(VMobject(stroke_color=color, stroke_width=4).set_points_as_corners(seg))
            return segs

        def drive(qs, color, run_time):
            """Move the arm through configurations qs, drawing the trace."""
            k = ValueTracker(0)

            def update(_):
                q_now["q"] = qs[min(int(k.get_value()), len(qs) - 1)]
            tr = always_redraw(lambda: trace(qs[:max(2, int(k.get_value()) + 1)], color))
            tr.add_updater(update)
            self.add(tr)
            self.play(k.animate.set_value(len(qs) - 1), run_time=run_time, rate_func=linear)
            tr.clear_updaters()
            return tr

        # Beat 1: the two pictures.
        self.add(disc, floor, frame, axes, live["arm"], live["dot"])
        cap = caption("One arm, two pictures: the room and its joint angles")
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(1.2)

        # Beat 2: the forbidden region.
        cap2 = caption("Grey: every joint setting where the arm touches the disc")
        swap(cap, cap2)
        self.play(FadeIn(image), run_time=1.2)
        self.wait(1.0)

        # Beat 3: the straight line, until the first collision.
        cap3 = caption("Turn both joints at a steady rate ...")
        swap(cap2, cap3)
        line_tr = drive(straight[:k_hit + 1], pal["orange"], run_time=2.0)
        q_now["color"] = pal["orange"]
        cap4 = caption("... and the arm runs into the disc")
        swap(cap3, cap4)
        self.wait(1.2)

        # Beat 4: back to the start, then the detour.
        cap5 = caption("Search the free cells: swing the other way round")
        swap(cap4, cap5)
        reset_to_start()
        detour_tr = drive(path, pal["blue"], run_time=5.0)
        self.wait(0.4)

        # Beat 5: the counts.
        cap6 = caption(f"Detour: {len(path) - 1} grid moves; straight line: {n_straight}, but blocked")
        swap(cap5, cap6)
        self.wait(2.0)

        # Beat 6: back to the start.
        self.play(FadeOut(line_tr), FadeOut(detour_tr), FadeOut(image), FadeOut(cap6), run_time=1.0)
        reset_to_start()
        self.wait(0.4)

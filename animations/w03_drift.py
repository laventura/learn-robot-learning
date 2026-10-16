"""Week 3 hero animation: one gust, many clones.

Data and policies are the Week 3 page's (rlc.toys.w03_imitation): the expert,
and the BC and DAgger policies of every training seed whose BC car stays on
the road on a calm day (experiment1), all driven from x = 12 with one gust at
step 10, as in the page's "one gust" figure.

Beats:
1. The curved road (shaded) with all cars on the centre line at the start.
2. The cars drive together along the centre line, leaving traces.
3. At step 10 a gust arrow hits; every car turns towards the verge.
4. The expert (black) and every DAgger car (blue) steer back onto the road;
   several BC cars (orange) cross the edge and keep going.
5. A caption gives the count of BC cars that end off the road.
6. Traces and cars fade; the cars reappear at the start (loop).
"""

import numpy as np
from manim import (UP, Arrow, Dot, FadeIn, FadeOut, Polygon, Scene, TracedPath,
                   VGroup, linear, smooth)

from rlc import anim_style as st
from rlc.toys import w03_imitation as w3

X0, T, T_GUST = 12.0, 60, 10
SX = 12.4 / (w3.SPEED * T)            # scene units per road unit (fits 30 units)
POSTER_TIME = 15.0                    # the result beat


def to_scene(x, y) -> np.ndarray:
    return np.array([(x - X0 - w3.SPEED * T / 2) * SX, y * SX - 0.25, 0.0])


class Drift(Scene):
    def construct(self):
        pal = st.palette("light")
        self.camera.background_color = pal["bg"]

        # The page's experiment: policies for every seed (before animating).
        E = w3.experiment1()
        calm = np.flatnonzero(E["nogust_left"] == 0)
        cars = [w3.expert] + [E["bc"][s] for s in calm] + [E["dagger"][s] for s in calm]
        colors = [pal["ink"]] + [pal["orange"]] * len(calm) + [pal["blue"]] * len(calm)
        offsets, states = w3.single_gust(cars, x0=X0, t_gust=T_GUST, T=T)
        n_off = int((np.abs(offsets[1:1 + len(calm), -1]) > w3.HALF_WIDTH).sum())

        # Road: the band |offset| < HALF_WIDTH around the centre line.
        xs = np.linspace(X0 - 1, X0 + w3.SPEED * T + 2, 120)
        top = [to_scene(x, w3.road_y(x) + w3.HALF_WIDTH) for x in xs]
        bottom = [to_scene(x, w3.road_y(x) - w3.HALF_WIDTH) for x in xs[::-1]]
        road = Polygon(*top, *bottom, fill_color=pal["band"], fill_opacity=0.9,
                       stroke_width=0)

        def caption(text):
            """Top caption with an opaque background, drawn above the traces
            of cars that leave the road upwards."""
            c = st.label(text, pal["ink"], 28).to_edge(UP, buff=0.35)
            c.add_background_rectangle(color=pal["bg"], opacity=1.0, buff=0.15)
            return c.set_z_index(5)

        def swap(old, new):
            self.play(FadeOut(old), run_time=0.3)
            self.play(FadeIn(new), run_time=0.3)

        dots = VGroup(*[Dot(to_scene(*states[i, 0, :2]), radius=0.07, color=c)
                        for i, c in enumerate(colors)])
        dots[0].set_z_index(2)                         # expert drawn on top

        # Beat 1: the start (also the loop's last frame).
        self.add(road, dots)
        cap = caption("Expert (black), 10 cloned policies (orange), 10 DAgger (blue)")
        self.play(FadeIn(cap), run_time=0.6)
        self.wait(1.2)

        # Beat 2: drive together; traces follow the cars.
        traces = VGroup(*[TracedPath(d.get_center, stroke_color=c, stroke_width=2,
                                     stroke_opacity=0.6) for d, c in zip(dots, colors)])
        self.add(traces)
        for k in range(1, T_GUST + 1):
            self.play(*[d.animate.move_to(to_scene(*states[i, k, :2]))
                        for i, d in enumerate(dots)], run_time=0.12, rate_func=linear)

        # Beat 3: the gust.
        where = to_scene(*states[0, T_GUST, :2])
        gust = Arrow(where + np.array([0.3, -1.6, 0]), where + np.array([0.3, -0.25, 0]),
                     color=pal["muted"], stroke_width=6, buff=0)
        cap2 = caption("A gust turns every car towards the edge")
        swap(cap, cap2)
        self.play(FadeIn(gust), run_time=0.4)
        self.wait(st.PAUSE)
        self.play(FadeOut(gust), run_time=0.3)

        # Beat 4: the rest of the drive.
        cap3 = caption("The expert and DAgger steer back; clones guess")
        swap(cap2, cap3)
        for k in range(T_GUST + 1, T + 1):
            self.play(*[d.animate.move_to(to_scene(*states[i, k, :2]))
                        for i, d in enumerate(dots)], run_time=0.12, rate_func=linear)

        # Beat 5: the count.
        cap4 = caption(f"{n_off} of {len(calm)} cloned policies end off the road")
        swap(cap3, cap4)
        self.wait(2.2)

        # Beat 6: back to the start.
        for t in traces:
            t.clear_updaters()
        self.play(FadeOut(traces), FadeOut(dots), FadeOut(cap4), run_time=1.0,
                  rate_func=smooth)
        for i, d in enumerate(dots):
            d.move_to(to_scene(*states[i, 0, :2]))
        self.play(FadeIn(dots), run_time=0.6)
        self.wait(0.4)

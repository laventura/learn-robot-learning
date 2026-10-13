"""Week 6 hero animation: diffusion turns noise into both modes; MSE averages.

Data and model are the Week 6 page's: the two-crescent demonstrations and the
DDPM noise predictor trained with seed 100 (rlc.toys.w06_generative).

Beats:
1. The demonstrated actions (grey crescents) with the empty middle shaded.
2. MSE regression's answer appears: one blue dot, the mean, in the empty middle.
3. A cloud of orange noise samples appears around the centre.
4. The particles follow the trained DDPM's 50 denoising steps, a step counter
   ticking, and split towards the two crescents.
5. Every particle has settled on one side; the middle stays empty.
6. Particles, dot and labels fade, back to the first frame (loop).
"""

import os

import numpy as np
from manim import (DOWN, UP, Create, Dot, FadeIn, FadeOut, Rectangle, Scene,
                   VGroup, linear, smooth)

from rlc import anim_style as st
from rlc.toys import w06_generative as g

SCALE = 1.4                      # data units -> scene units; keeps 3-sigma noise in frame
POSTER_TIME = 14.0               # seconds: the result beat, used as the poster frame


def to_scene(xy) -> np.ndarray:
    """(x, y) in data units -> a Manim point (x, y, 0), shifted down a bit
    to leave room for the caption line at the top."""
    return np.array([SCALE * xy[0], SCALE * xy[1] - 0.3, 0.0])


class Denoise(Scene):
    def construct(self):
        pal = st.palette(os.environ.get("ANIM_BACKGROUND", "light"))
        self.camera.background_color = pal["bg"]

        # Same data and network as the page (before any animation starts).
        data = g.crescents(2000, np.random.default_rng(0))
        betas = g.linear_schedule(50)
        params, _ = g.train_ddpm(data, betas, np.random.default_rng(100))
        path = g.ddpm_sample(params, betas, 160, np.random.default_rng(600),
                             keep_path=True)               # (51, 160, 2)
        mean = data.mean(axis=0)

        band = Rectangle(width=2 * g.GAP * SCALE, height=5.6, fill_color=pal["band"],
                         fill_opacity=0.7, stroke_width=0).move_to(to_scene((0, 0)))
        demos = VGroup(*[Dot(to_scene(p), radius=0.022, color=pal["muted"])
                         for p in data[::5]])

        def caption(text):
            return st.label(text, pal["ink"], 30).to_edge(UP, buff=0.35)

        # Beat 1: the starting frame (also the loop's last frame).
        self.add(band, demos)
        self.wait(1.0)

        def swap(old, new):
            """Replace one caption by another without overlapping them."""
            self.play(FadeOut(old), run_time=0.3)
            self.play(FadeIn(new), run_time=0.3)

        # Beat 2: the MSE answer, in the middle.
        mse = Dot(to_scene(mean), radius=0.11, color=pal["blue"])
        cap = caption("MSE regression: one action, the average")
        self.play(FadeIn(cap), FadeIn(mse, scale=3), run_time=1.0, rate_func=smooth)
        self.wait(1.5)

        # Beat 3: noise.
        dots = VGroup(*[Dot(to_scene(p), radius=0.045, color=pal["orange"])
                        for p in path[0]])
        cap2 = caption("Diffusion: start from noise ...")
        swap(cap, cap2)
        self.play(FadeIn(dots), run_time=0.8, rate_func=smooth)
        self.wait(st.PAUSE)

        # Beat 4: follow the 50 denoising steps. DDPM changes little early
        # and a lot late, so early steps go fast and late steps slowly.
        cap3 = caption("... and remove it, step by step")
        swap(cap2, cap3)
        counter = st.label("step 0 / 50", pal["ink"], 24).to_edge(DOWN, buff=0.3)
        self.add(counter)
        n = len(path) - 1
        for k in range(1, n + 1):
            new_counter = st.label(f"step {k} / 50", pal["ink"], 24).to_edge(DOWN, buff=0.3)
            self.play(*[d.animate.move_to(to_scene(p)) for d, p in zip(dots, path[k])],
                      run_time=0.04 + 0.3 * (k / n) ** 3, rate_func=linear)
            self.remove(counter)
            counter = new_counter
            self.add(counter)

        # Beat 5: the result.
        cap4 = caption("Each sample picks a side; the middle stays empty")
        swap(cap3, cap4)
        self.wait(2.0)

        # Beat 6: back to the first frame.
        self.play(FadeOut(dots), FadeOut(mse), FadeOut(cap4), FadeOut(counter),
                  run_time=1.2, rate_func=smooth)
        self.wait(0.5)

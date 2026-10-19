"""Week 5 hero animation: batches of samples pull a Gaussian policy onto a reward bump.

Data are the Week 5 page's reuse toy (rlc.toys.w05_pg): seed 0, K = 10
clipped steps per batch, 30 batches of 32 samples, exactly the run the page
quotes. Every batch's actions, noisy rewards and advantages come from
reuse_run(keep=True); nothing is hand-typed.

Beats:
1. The setting: the reward bump (top band) and the policy N(0, 1) (bottom band).
2. Batches 1 to 30: 32 dots appear at their actions and noisy rewards, blue if
   better than the batch mean, orange if worse; the policy then moves to its
   new mean and width, and the dots fade. A counter shows the batch and J.
3. The policy first widens a little; from the first batch after its widest
   point with sigma below 1, the caption changes and it narrows over the bump.
4. A closing caption with the start and end expected reward.
5. The policy slides back to N(0, 1) (loop).
"""

import numpy as np
from manim import (LEFT, UP, Dot, FadeIn, FadeOut, Line, ManimColor, Scene, ValueTracker,
                   VGroup, VMobject, always_redraw, linear, smooth)

from rlc import anim_style as st
from rlc.toys import w05_pg as w5

X_LO, X_HI = -3.0, 4.0                       # action range shown
SX = 12.0 / (X_HI - X_LO)                    # scene units per action unit
R0, SR = 0.6, 1.1                            # reward band: y of r = 0, units per reward
D0, SD = -3.3, 1.0                           # density band: baseline y, units per density
POSTER_TIME = 26.5                           # the result caption over the narrow policy, before the fade


def sx(a):
    return (np.asarray(a) - (X_LO + X_HI) / 2) * SX


def gauss_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


class Batches(Scene):
    def construct(self):
        pal = st.palette("light")
        self.camera.background_color = pal["bg"]
        run = w5.reuse_run(10, True, 0, keep=True)
        mus, sigmas, Js = run["mu"], run["sigma"], run["J"]
        xs = np.linspace(X_LO, X_HI, 281)

        def curve(ys, color, width=4):
            m = VMobject(stroke_color=ManimColor(color), stroke_width=width)
            m.set_points_smoothly([np.array([sx(x), y, 0.0]) for x, y in zip(xs, ys)])
            return m

        # Beat 1: axes, the bump (dashed look via muted colour) and the first policy.
        axis_r = Line([sx(X_LO), R0, 0], [sx(X_HI), R0, 0], stroke_color=pal["muted"], stroke_width=1.5)
        axis_d = Line([sx(X_LO), D0, 0], [sx(X_HI), D0, 0], stroke_color=pal["ink"], stroke_width=2)
        ticks = VGroup(*[st.label(f"{t:d}".replace("-", "−"), pal["ink"], 20).move_to([sx(t), D0 - 0.3, 0])
                         for t in (-2, 0, 2, 4)])
        bump = curve(R0 + SR * w5.bump(xs), pal["muted"], 3)
        names = VGroup(st.label("reward", pal["muted"], 20).next_to([sx(X_LO), R0 + SR, 0], UP, buff=0.1),
                       st.label("policy", pal["blue"], 20).next_to([sx(X_LO), D0 + 0.5, 0], UP, buff=0.1))
        names.align_to([sx(X_LO), 0, 0], LEFT)
        mu_t, sig_t = ValueTracker(mus[0]), ValueTracker(sigmas[0])
        policy = always_redraw(lambda: curve(D0 + SD * gauss_pdf(xs, mu_t.get_value(), sig_t.get_value()),
                                             pal["blue"], 5))

        def caption(text):
            c = st.label(text, pal["ink"], 28).to_edge(UP, buff=0.3)
            c.add_background_rectangle(color=pal["bg"], opacity=1.0, buff=0.15)
            return c.set_z_index(5)

        def counter(b):
            c = st.label(f"batch {b} · expected reward {Js[b]:.2f}", pal["ink"], 22)
            return c.to_edge(UP, buff=0.95).to_edge(LEFT, buff=0.6)

        # The caption changes when the policy starts to narrow: the first batch
        # after its widest point whose standard deviation is below the initial 1.
        widest = int(np.argmax(sigmas))
        narrow_from = next(b for b in range(widest, len(sigmas)) if sigmas[b] < sigmas[0])

        cap = caption("A policy over one action, and a reward it can only sample")
        self.add(axis_r, axis_d, ticks, bump, names, policy)
        self.play(FadeIn(cap), run_time=0.5)
        cnt = counter(0)
        self.play(FadeIn(cnt), run_time=0.3)
        self.wait(1.2)

        # Beats 2-3: the batches.
        cap2 = caption("Better-than-average samples (blue) pull the policy")
        self.play(FadeOut(cap), run_time=0.3)
        self.play(FadeIn(cap2), run_time=0.3)
        cap_now = cap2
        for b, (a, r, adv) in enumerate(run["batches"], start=1):
            if b == narrow_from:
                cap3 = caption("Once it covers the bump, it narrows")
                self.play(FadeOut(cap_now), FadeIn(cap3), run_time=0.3)
                cap_now = cap3
            dots = VGroup(*[Dot([sx(ai), R0 + SR * ri, 0], radius=0.06,
                                color=ManimColor(pal["blue"] if di > 0 else pal["orange"]))
                            for ai, ri, di in zip(a, r, adv) if X_LO <= ai <= X_HI])
            self.play(FadeIn(dots), run_time=0.2)
            # Swap the counter text instantly; morphing digits garbles them.
            cnt.become(counter(b))
            self.play(mu_t.animate.set_value(mus[b]), sig_t.animate.set_value(sigmas[b]),
                      run_time=0.35, rate_func=smooth)
            self.play(FadeOut(dots), run_time=0.15)

        # Beat 4: the result.
        cap4 = caption(f"30 batches: expected reward {Js[0]:.2f} → {Js[-1]:.2f}")
        self.play(FadeOut(cap_now), FadeIn(cap4), run_time=0.4)
        self.wait(2.0)

        # Beat 5: back to the first frame.
        cap0 = caption("A policy over one action, and a reward it can only sample")
        self.play(FadeOut(cap4), mu_t.animate.set_value(mus[0]), sig_t.animate.set_value(sigmas[0]),
                  run_time=1.2, rate_func=smooth)
        cnt.become(counter(0))
        self.play(FadeIn(cap0), run_time=0.3)
        self.wait(0.4)

"""Shared style for hero animations (CLAUDE.md §6.3a).

Colours come from rlc.style, so clips match the site's figures. Manim is
imported inside functions only: `rlc` itself must not depend on it, and
pages never import manim.
"""

from rlc import style

BACKGROUNDS = {
    "light": {"bg": style.SURFACE, "ink": style.INK, "muted": style.MUTED,
              "band": style.GRID},
    "dark": {"bg": "#16181d", "ink": "#ececec", "muted": "#8d8d93",
             "band": "#2b2e36"},
}
BLUE, ORANGE, GREEN = style.CATEGORICAL[0], style.CATEGORICAL[1], style.CATEGORICAL[2]
FONT = "Helvetica"
PAUSE = 0.5            # seconds between beats, so each beat can be read


def palette(background="light") -> dict:
    """Colours for one background variant."""
    return {**BACKGROUNDS[background], "blue": BLUE, "orange": ORANGE,
            "green": GREEN}


def label(text, color, size=28):
    """A short plain-text label (Manim Text, rendered with Pango; no LaTeX)."""
    from manim import Text

    return Text(text, font=FONT, color=color, font_size=size)

"""The figure style keeps ADM's validated palette (CLAUDE.md §6.3).

Graphviz diagrams and OJS cells hard-code some of these colors, so a change
here must be made there too.
"""

import re

import pytest

from rlc import style

HEX = re.compile(r"^#[0-9a-f]{6}$")


@pytest.mark.parametrize("name", ["CATEGORICAL", "CATEGORICAL_DARK", "SEQUENTIAL_BLUE", "DIVERGING"])
def test_color_lists_are_hex(name):
    assert all(HEX.match(c) for c in getattr(style, name))


def test_categorical_has_eight_distinct_hues():
    for colors in (style.CATEGORICAL, style.CATEGORICAL_DARK):
        assert len(colors) == 8 and len(set(colors)) == 8


def test_palette_matches_diagram_colors():
    # The colors §6.3 lists for Graphviz diagrams.
    assert style.SURFACE == "#fcfcfb"
    assert (style.INK, style.INK_SECONDARY) == ("#0b0b0b", "#52514e")
    assert style.CATEGORICAL[:2] == ["#2a78d6", "#eb6834"]   # blue, orange
    assert style.SEQUENTIAL_BLUE[0] == "#cde2fb"              # blue fill
    assert style.DIVERGING[1] == "#f0efec"                    # neutral fill


def test_apply_sets_palette():
    import matplotlib as mpl

    style.apply()
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"] == style.CATEGORICAL

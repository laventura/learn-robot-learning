from make_course_map import ARCS, OUT, ROWS, source

from rlc.page import course_map_highlight, load_data


def test_include_is_up_to_date():
    # Regenerate with: uv run python tools/make_course_map.py
    assert OUT.read_text(encoding="utf-8") == source(load_data("weeks"))


def test_every_week_is_placed_and_connected():
    weeks = {w["week"] for w in load_data("weeks")}
    assert sorted(n for row in ROWS for n in row) == sorted(weeks)
    assert {n for a, b, _ in ARCS for n in (a, b)} == weeks
    assert all(label.strip() for _, _, label in ARCS)


def test_highlight_targets_one_node():
    css = course_map_highlight(3)
    assert "#w03" in css and "#w04" not in css and css.startswith("<style>")

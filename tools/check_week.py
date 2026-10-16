"""Count what CLAUDE.md §6.8 asks of a week page, from its source.

Checks the parts a script can count: template sections, Graphviz diagrams
besides the course map, figures (static, interactive) with captions and alt
text, boxed equations, watch links into the lecture and the guest talk,
solved problems, the hero clip, snippets, paper cards, resources and
cross-links. Judgement items (self-sufficiency, originality, looking at the
page) stay with the reviewer.

Usage: uv run python tools/check_week.py 6
Exits with status 1 if any count is out of range.
"""

import re
import sys
from pathlib import Path

from rlc.page import load_data

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ["Big picture", "Intuition", "Key concepts & formulae", "Worked examples",
            "Interactive", "Algorithm & code", "Papers", "Guest spotlight",
            "Solved problems", "Recap", "Cheat sheet & connections", "Resources"]
CELL = re.compile(r"^```\{(python|ojs|dot)\}\n(.*?)^```", re.M | re.S)
PLOTLY = re.compile(r"\bgo\.|make_subplots")


def cells(text: str) -> list[tuple[str, str]]:
    """(engine, body) of every executed cell, in page order."""
    return CELL.findall(text)


def figures(text: str) -> dict[str, list[str]]:
    """Labels of captioned figures by kind, plus captioned cells missing alt text.

    A captioned dot cell is a diagram, an ojs cell or a Python cell that
    builds a Plotly figure is interactive, and any other Python cell is static."""
    out = {"diagram": [], "interactive": [], "static": [], "no_alt": []}
    for engine, body in cells(text):
        if "fig-cap:" not in body:
            continue
        m = re.search(r"label:\s*(\S+)", body)
        label = m.group(1) if m else "(no label)"
        if "fig-alt:" not in body:
            out["no_alt"].append(label)
        if engine == "dot":
            out["diagram"].append(label)
        elif engine == "ojs" or PLOTLY.search(body):
            out["interactive"].append(label)
        else:
            out["static"].append(label)
    return out


def watch_links(text: str, video: str | None) -> int:
    return len(re.findall(rf"\[▶[^\]]*\]\(https://youtu\.be/{video}\?t=", text)) if video else 0


def calls(text: str, helper: str, week: int) -> bool:
    """Whether the page calls helper(N) or helper(week=N) for its own week."""
    return re.search(rf"\b{helper}\((?:week=)?{week}\)", text) is not None


def report(week: int) -> list[tuple[str, str, bool]]:
    """(item, what was found, ok) rows for one week."""
    w = next(x for x in load_data("weeks") if x["week"] == week)
    text = (ROOT / "weeks" / f"{w['slug']}.qmd").read_text(encoding="utf-8")
    heads = re.findall(r"^## (.+?)\s*(?:\{#[^}]*\})?$", text, re.M)
    has_guest = bool(w["guest"] and w["guest"].get("youtube"))
    wanted = [s for s in SECTIONS if has_guest or s != "Guest spotlight"]
    missing = [s for s in wanted if s not in heads]
    fig = figures(text)
    boxes = text.count(".key-equation")
    displayed = len(re.findall(r"^\$\$", text, re.M)) // 2    # opening + closing line
    lecture = watch_links(text, w["lecture"]["youtube"] if w["lecture"] else None)
    guest = watch_links(text, w["guest"]["youtube"] if has_guest else None)
    solutions = len(re.findall(r'\{\.callout-tip collapse="true" title="Solution"\}', text))
    cross = len(set(re.findall(r"\((w\d\d-[a-z0-9-]+\.qmd)", text))) \
        + len(re.findall(r"\{\{< var adm_url >\}\}", text))
    rows = [
        ("sections", "all present" if not missing else "missing: " + ", ".join(missing), not missing),
        ("course map", "included" if "course-map.qmd" in text else "absent", "course-map.qmd" in text),
        ("other diagrams (>= 1)", str(len(fig["diagram"])), len(fig["diagram"]) >= 1),
        ("static figures (2-6)", str(len(fig["static"])), 2 <= len(fig["static"]) <= 6),
        ("interactives (>= 2)", str(len(fig["interactive"])), len(fig["interactive"]) >= 2),
        ("figures without fig-alt", ", ".join(fig["no_alt"]) or "none", not fig["no_alt"]),
        ("boxed equations", f"{boxes} boxes for {displayed} displayed", boxes >= displayed),
        ("lecture ▶ links (>= 4)", str(lecture), lecture >= 4),
        ("guest ▶ links (>= 2)" if has_guest else "guest ▶ links (no guest)", str(guest),
         guest >= 2 if has_guest else True),
        ("solved problems (3-5)", str(solutions), 3 <= solutions <= 5),
        ("hero clip", "embedded" if "hero_video(" in text else "absent", "hero_video(" in text),
        ("PyTorch snippets", str(text.count("{.python .snippet}")), "{.python .snippet}" in text),
        ("paper cards", "yes" if calls(text, "paper_cards", week) else "no",
         calls(text, "paper_cards", week)),
        ("resources", "yes" if calls(text, "resources", week) else "no",
         calls(text, "resources", week)),
        ("cross-links (>= 2)", str(cross), cross >= 2),
    ]
    return rows


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        print(__doc__)
        return 2
    rows = report(int(sys.argv[1]))
    for item, found, ok in rows:
        print(f"{'OK  ' if ok else 'FAIL'}  {item:<28} {found}")
    return 0 if all(ok for *_, ok in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

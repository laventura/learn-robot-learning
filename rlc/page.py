"""Helpers for Quarto pages."""

import datetime
import inspect
import re
from pathlib import Path

REF_WORDS = {"eq": "eq.", "fig": "fig.", "tbl": "table"}   # short: Ref columns are narrow

# data/ sits next to rlc/. Pages render from their own folder (weeks/), so
# paths are resolved from this file, not from the working directory.
DATA = Path(__file__).resolve().parent.parent / "data"

OFFICIAL_SLIDES = "Official slides (ETH; password-protected)"
COMMUNITY_SLIDES = "Community rebuild (unofficial, student repo)"
ROLES = {"discussion": "Papers discussed in class", "lecture": "Key papers from the lecture"}


def show_source(*objects) -> None:
    """Print the objects' source as one fenced Python block.

    Use in an executed cell with `#| output: asis`, so a code tab shows exactly the
    code that produced the page's results.
    """
    code = "\n\n".join(inspect.getsource(obj).rstrip() for obj in objects)
    print(f"```python\n{code}\n```")


def page_title(page) -> str:
    """The `title:` from a page's YAML front matter, without quotes."""
    match = re.search(r'^title:\s*"?(.*?)"?\s*$', Path(page).read_text(), re.M)
    return match[1] if match else Path(page).stem


def cheat_sheet(page, href) -> str:
    """The first Markdown table under a page's "## Cheat sheet" heading.

    Cross-references such as @eq-bellman only resolve on their own page, so
    they become links to it: [eq.](href#eq-bellman). Returns "" if the
    page has no cheat sheet.
    """
    parts = Path(page).read_text().split("\n## Cheat sheet", 1)
    if len(parts) < 2:
        return ""
    rows = []
    for line in parts[1].splitlines()[1:]:
        if line.startswith("|"):
            rows.append(line)
        elif rows:
            break               # the table has ended
    table = "\n".join(rows)
    return re.sub(r"@((eq|fig|tbl)-[\w-]+)",
                  lambda m: f"[{REF_WORDS[m[2]]}]({href}#{m[1]})", table)


# --- Timestamps ----------------------------------------------------------

def timestamp(seconds: int) -> str:
    """'mm:ss' under an hour, 'h:mm:ss' from an hour on, as YouTube shows them."""
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_timestamp(label: str) -> int:
    """Seconds in a 'mm:ss' or 'h:mm:ss' label. Raises ValueError otherwise."""
    parts = label.split(":")
    if not 2 <= len(parts) <= 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"not a timestamp: {label!r}")
    seconds = 0
    for p in parts:
        seconds = 60 * seconds + int(p)
    return seconds


def youtube_url(video: str, t: int | None = None) -> str:
    return f"https://youtu.be/{video}" + (f"?t={t}" if t is not None else "")


# --- Data layer: data/weeks.yml and data/papers.yml (CLAUDE.md §6.5) ------

def load_data(name: str, data_dir=None) -> list:
    """The parsed list in data/<name>.yml. PyYAML is a site dependency, not a
    core one, so it is imported here."""
    import yaml

    return yaml.safe_load((Path(data_dir or DATA) / f"{name}.yml").read_text(encoding="utf-8")) or []


def week_entry(week: int, data_dir=None) -> dict:
    """The weeks.yml entry for one week."""
    for entry in load_data("weeks", data_dir):
        if entry["week"] == week:
            return entry
    raise KeyError(f"week {week} not in weeks.yml")


def _papers_by_id(data_dir=None) -> dict:
    return {p["id"]: p for p in load_data("papers", data_dir)}


def _is_todo(text) -> bool:
    return not text or str(text).strip().upper().startswith("TODO")


def _card(p: dict) -> str:
    """One paper as a callout: title, byline, summary, contributions, link to the page's ideas."""
    lines = ['::: {.callout-note appearance="simple" icon=false}',
             f"**[{p['title']}]({p['url']})**<br>",
             f"{p['authors']} · *{p['venue']}* {p['year']}", ""]
    if _is_todo(p["summary"]):
        lines.append("*Summary to come.*")
    else:
        lines += [p["summary"], ""]
        lines += [f"- {c}" for c in p["contributions"]]
        if not _is_todo(p["connects_to"]):
            lines += ["", f"**On this page:** {p['connects_to']}"]
    lines.append(":::")
    return "\n".join(lines)


def paper_cards(week: int, data_dir=None) -> str:
    """Markdown for a week's Papers section: one callout per paper, grouped as
    discussion papers, then key papers from the lecture, each in weeks.yml order.
    Print it from a cell marked `#| output: asis`."""
    papers = _papers_by_id(data_dir)
    listed = [papers[i] for i in week_entry(week, data_dir)["papers"]]
    blocks = []
    for role, heading in ROLES.items():
        group = [p for p in listed if p["role"] == role]
        if group:
            blocks.append(f"### {heading}\n")
            blocks += [_card(p) + "\n" for p in group]
    return "\n".join(blocks) or "*No papers listed yet.*"


def resources(week: int, data_dir=None) -> str:
    """Markdown for a week's Resources section. Missing links are left out."""
    w = week_entry(week, data_dir)
    papers = _papers_by_id(data_dir)
    items = []
    lec = w["lecture"]
    if lec["youtube"]:
        length = f" ({timestamp(lec['duration_s'])})" if lec["duration_s"] else ""
        items.append(f"- **Lecture:** [video]({youtube_url(lec['youtube'])}){length}")
    slides = [f"[{label}]({w['slides'][key]})"
              for key, label in (("official", OFFICIAL_SLIDES), ("community", COMMUNITY_SLIDES))
              if w["slides"][key]]
    if slides:
        items.append("- **Slides:** " + " · ".join(slides))
    if g := w["guest"]:
        name = f"[{g['name']}]({g['url']})" if g["url"] else g["name"]
        video = f": [video]({youtube_url(g['youtube'])})" if g["youtube"] else ""
        items.append(f"- **Guest lecture:** {name} ({g['affiliation']}){video}")
    if w["papers"]:
        items.append("- **Papers:**")
        items += [f"    - [{papers[i]['title']}]({papers[i]['url']}) "
                  f"({papers[i]['authors']}, {papers[i]['year']})" for i in w["papers"]]
    if hw := w["homework"]:
        name = f"[{hw['name']}]({hw['url']})" if hw["url"] else hw["name"]
        items.append(f"- **Homework:** {name}")
    return "\n".join(items) or "*No resources listed yet.*"


def papers_index(data_dir=None) -> str:
    """Markdown for papers.qmd: every paper, grouped by week, linking each
    week heading to its page."""
    papers = _papers_by_id(data_dir)
    out = []
    for w in load_data("weeks", data_dir):
        out.append(f"\n## [Week {w['week']} · {w['title']}](weeks/{w['slug']}.qmd)\n")
        if not w["papers"]:
            out.append("*No papers listed yet.*")
        for i in w["papers"]:
            p = papers[i]
            role = "" if p["role"] == "discussion" else " *(from the lecture)*"
            out.append(f"- [{p['title']}]({p['url']}). {p['authors']}, *{p['venue']}* {p['year']}.{role}")
    return "\n".join(out)


# --- Validation (used by tests/test_data.py and tools/check_links.py) -----

WEEK_KEYS = {"week", "date", "title", "slug", "lecture", "slides", "guest", "homework", "papers"}
VIDEO_KEYS = {"youtube", "duration_s"}
GUEST_KEYS = {"name", "affiliation", "url"} | VIDEO_KEYS
PAPER_KEYS = {"id", "role", "weeks", "title", "authors", "year", "venue", "url",
              "summary", "contributions", "connects_to"}
YOUTUBE_ID = re.compile(r"^[\w-]{11}$")


def _keys(where: str, entry, expected: set) -> list[str]:
    if not isinstance(entry, dict):
        return [f"{where}: expected a mapping"]
    errors = []
    if missing := expected - entry.keys():
        errors.append(f"{where}: missing {sorted(missing)}")
    if extra := entry.keys() - expected:
        errors.append(f"{where}: unknown {sorted(extra)}")
    return errors


def _video(where: str, entry: dict) -> list[str]:
    errors = []
    vid, dur = entry.get("youtube"), entry.get("duration_s")
    if vid is not None and not YOUTUBE_ID.match(str(vid)):
        errors.append(f"{where}: bad YouTube id {vid!r}")
    if dur is not None and not (isinstance(dur, int) and dur > 0):
        errors.append(f"{where}: duration_s must be a positive integer")
    return errors


def validate(weeks: list, papers: list, pages_dir=None) -> list[str]:
    """Problems in the two data files; empty when all is well. With pages_dir,
    also checks that every slug has a page there."""
    errors = []
    numbers = [w.get("week") for w in weeks]
    if numbers != list(range(1, len(weeks) + 1)):
        errors.append(f"weeks.yml: weeks must be numbered 1..{len(weeks)} in order, got {numbers}")
    by_id = {}
    for p in papers:
        if p.get("id") in by_id:
            errors.append(f"papers.yml: duplicate id {p['id']!r}")
        by_id[p.get("id")] = p

    for w in weeks:
        where = f"weeks.yml week {w.get('week')}"
        errors += _keys(where, w, WEEK_KEYS)
        if WEEK_KEYS - w.keys():
            continue
        if not isinstance(w["date"], datetime.date):
            errors.append(f"{where}: date must be YYYY-MM-DD")
        if not str(w["slug"]).startswith(f"w{w['week']:02d}-"):
            errors.append(f"{where}: slug {w['slug']!r} must start with w{w['week']:02d}-")
        elif pages_dir and not (Path(pages_dir) / f"{w['slug']}.qmd").exists():
            errors.append(f"{where}: no page {w['slug']}.qmd")
        errors += _keys(f"{where} lecture", w["lecture"], VIDEO_KEYS) + _video(f"{where} lecture", w["lecture"])
        errors += _keys(f"{where} slides", w["slides"], {"official", "community"})
        if w["guest"] is not None:
            errors += _keys(f"{where} guest", w["guest"], GUEST_KEYS) + _video(f"{where} guest", w["guest"])
        if w["homework"] is not None:
            errors += _keys(f"{where} homework", w["homework"], {"name", "url"})
        if len(set(w["papers"])) != len(w["papers"]):
            errors.append(f"{where}: a paper is listed twice")
        for pid in w["papers"]:
            if pid not in by_id:
                errors.append(f"{where}: unknown paper {pid!r}")
            elif w["week"] not in by_id[pid].get("weeks", []):
                errors.append(f"{where}: lists {pid!r}, but the paper's weeks don't include {w['week']}")

    listed = {(w.get("week"), pid) for w in weeks for pid in w.get("papers", [])}
    for p in papers:
        where = f"papers.yml {p.get('id')!r}"
        errors += _keys(where, p, PAPER_KEYS)
        if PAPER_KEYS - p.keys():
            continue
        if p["role"] not in ROLES:
            errors.append(f"{where}: role must be one of {sorted(ROLES)}")
        if not p["weeks"]:
            errors.append(f"{where}: belongs to no week")
        for n in p["weeks"]:
            if n not in numbers:
                errors.append(f"{where}: unknown week {n}")
            elif (n, p["id"]) not in listed:
                errors.append(f"{where}: week {n} does not list it")
        if not isinstance(p["year"], int):
            errors.append(f"{where}: year must be an integer")
        if not isinstance(p["contributions"], list):
            errors.append(f"{where}: contributions must be a list")
    return errors

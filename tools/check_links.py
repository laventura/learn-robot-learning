"""Validate data/weeks.yml and data/papers.yml, and every "▶ mm:ss" link on
the pages (CLAUDE.md §6.6).

A watch link looks like [▶ 23:15](https://youtu.be/Ef4R5s1LqoQ?t=1395).
For each one this checks that
- the video id is a lecture or guest video in weeks.yml;
- t is within the video's duration_s (which must be filled in);
- the label equals t ("23:15" is 1395 s; past an hour, "1:02:03").

Usage: uv run python tools/check_links.py
Exits with status 1 if anything is wrong.
"""

import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rlc.page import load_data, parse_timestamp, timestamp, validate

ROOT = Path(__file__).resolve().parent.parent
WATCH_LINK = re.compile(r"\[▶\s*([^\]]*?)\s*\]\(([^)\s]+)\)")


def known_videos(weeks: list) -> dict[str, int | None]:
    """Video id -> duration_s for every lecture and guest video."""
    videos = {}
    for w in weeks:
        for entry in (w["lecture"], w["guest"]):
            if entry and entry["youtube"]:
                videos[entry["youtube"]] = entry["duration_s"]
    return videos


def video_and_time(url: str) -> tuple[str | None, int | None]:
    """(id, t in seconds) from a youtu.be or youtube.com/watch URL.
    t may be written "1395" or "1395s". Missing parts come back as None."""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if parsed.netloc.endswith("youtu.be"):
        video = parsed.path.lstrip("/") or None
    elif "youtube.com" in parsed.netloc:
        video = query.get("v", [None])[0]
    else:
        return None, None
    t = query.get("t", [None])[0]
    if t is not None:
        t = t.removesuffix("s")
        t = int(t) if t.isdigit() else None
    return video, t


def check_text(text: str, videos: dict, where: str) -> list[str]:
    """Problems with the watch links in one page's source."""
    errors = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for label, url in WATCH_LINK.findall(line):
            at = f"{where}:{lineno} [▶ {label}]"
            video, t = video_and_time(url)
            if video is None:
                errors.append(f"{at}: not a YouTube link: {url}")
                continue
            if video not in videos:
                errors.append(f"{at}: video {video} is not in weeks.yml")
            if t is None:
                errors.append(f"{at}: no t= in {url}")
                continue
            try:
                if parse_timestamp(label) != t:
                    errors.append(f"{at}: label says {label}, but t={t} is {timestamp(t)}")
            except ValueError:
                errors.append(f"{at}: label is not mm:ss or h:mm:ss")
            duration = videos.get(video)
            if video in videos and duration is None:
                errors.append(f"{at}: duration_s of {video} is not filled in")
            elif duration is not None and t >= duration:
                errors.append(f"{at}: t={t} is past the end of the video ({duration} s)")
    return errors


def pages() -> list[Path]:
    return sorted(ROOT.glob("*.qmd")) + sorted((ROOT / "weeks").glob("*.qmd"))


def main() -> int:
    weeks, papers = load_data("weeks"), load_data("papers")
    errors = validate(weeks, papers, ROOT / "weeks")
    if not errors:                          # link checks need well-formed data
        videos = known_videos(weeks)
        for page in pages():
            errors += check_text(page.read_text(encoding="utf-8"), videos, str(page.relative_to(ROOT)))
    for e in errors:
        print(e)
    links = sum(len(WATCH_LINK.findall(p.read_text(encoding="utf-8"))) for p in pages())
    print(f"{'FAIL' if errors else 'OK'}: data files and {links} watch links checked, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

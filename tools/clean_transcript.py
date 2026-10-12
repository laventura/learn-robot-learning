"""Turn YouTube auto-captions (.vtt) into deduplicated text with a [mm:ss]
marker about every 30 seconds (CLAUDE.md §5).

Auto-captions roll: each cue repeats the line before it and adds a new one,
and short "transition" cues repeat a line on its own. Each line is kept the
first time it appears, stamped with that cue's start time. Word-timing tags
such as <00:00:01.040><c> so</c> are stripped.

Usage: uv run python tools/clean_transcript.py refs/transcripts/w03-lecture.en.vtt [-o out.txt]
Writes refs/transcripts/w03-lecture.txt by default.
"""

import argparse
import html
import re
import sys
from pathlib import Path

from rlc.page import timestamp

CUE_TIME = re.compile(r"^(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})\s+-->")
TAG = re.compile(r"<[^>]*>")


def cue_start(line: str) -> float | None:
    """Start time in seconds of a cue timing line, else None."""
    m = CUE_TIME.match(line)
    if not m:
        return None
    h, mins, s, ms = (int(x) if x else 0 for x in m.groups())
    return 3600 * h + 60 * mins + s + ms / 1000


def caption_lines(vtt: str) -> list[tuple[float, str]]:
    """(start, text) for each new caption line, in order, repeats removed."""
    out: list[tuple[float, str]] = []
    start = None
    for raw in vtt.splitlines():
        t = cue_start(raw)
        if t is not None:
            start = t
            continue
        if start is None:                    # header: WEBVTT, Kind:, Language:
            continue
        text = " ".join(html.unescape(TAG.sub("", raw)).split())
        if text and (not out or text != out[-1][1]):
            out.append((start, text))
    return out


def paragraphs(lines: list[tuple[float, str]], every: float = 30) -> str:
    """Join the lines into paragraphs, starting a new one with a [mm:ss]
    marker once `every` seconds have passed since the last marker."""
    paras: list[str] = []
    marker = None
    for start, text in lines:
        if marker is None or start >= marker + every:
            marker = start
            paras.append(f"[{timestamp(int(start))}] {text}")
        else:
            paras[-1] += " " + text
    return "\n\n".join(paras) + "\n"


def default_output(vtt_path: Path) -> Path:
    """w03-lecture.en.vtt -> w03-lecture.txt, in the same folder."""
    name = re.sub(r"(\.[a-z]{2}(-[A-Za-z]+)?)?\.vtt$", "", vtt_path.name)
    return vtt_path.with_name(name + ".txt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("vtt", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    lines = caption_lines(args.vtt.read_text(encoding="utf-8"))
    if not lines:
        print(f"No captions found in {args.vtt}")
        return 1
    out = args.output or default_output(args.vtt)
    out.write_text(paragraphs(lines), encoding="utf-8")
    print(f"{out}: {len(lines)} lines, last at {timestamp(int(lines[-1][0]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

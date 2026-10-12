"""Run each week page's PyTorch snippets and report pass/fail (CLAUDE.md §6.2).

A snippet is a fenced block with the `.python` and `.snippet` classes:
```{.python .snippet}. Per page, snippets are joined in page order and run as one
script with the project's Python (PyTorch on CPU), in a temporary folder so they
cannot overwrite committed files.

Usage: uv run python tools/check_snippets.py [--week 3]
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEEKS = ROOT / "weeks"
FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*(.*?)\s*$")


def extract_snippets(text: str) -> list[str]:
    """Return the bodies of the page's Python snippets, in page order."""
    snippets: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        opening = FENCE.match(lines[i])
        i += 1
        if not opening:
            continue
        marker, info = opening.groups()
        body = []
        while i < len(lines):
            closing = FENCE.match(lines[i])
            i += 1
            if closing and closing[1][0] == marker[0] and len(closing[1]) >= len(marker) and not closing[2]:
                break
            body.append(lines[i - 1])
        if not (info.startswith("{") and info.endswith("}")):
            continue
        classes = {tok[1:] for tok in info[1:-1].split() if tok.startswith(".")}
        if {"python", "snippet"} <= classes:
            snippets.append("\n".join(body))
    return snippets


def find_pages(week: int | None) -> list[Path]:
    pattern = "w*.qmd" if week is None else f"w{week:02d}-*.qmd"
    return sorted(WEEKS.glob(pattern))


def run(code: str) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "snippets.py"
        script.write_text(code + "\n", encoding="utf-8")
        return subprocess.run([sys.executable, str(script)], cwd=tmp, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--week", type=int, help="week number, e.g. 3")
    args = parser.parse_args()

    pages = find_pages(args.week)
    if not pages:
        print(f"No page found for week {args.week!r}")
        return 2

    failures = checked = 0
    for page in pages:
        blocks = extract_snippets(page.read_text(encoding="utf-8"))
        if not blocks:
            continue
        checked += 1
        result = run("\n\n".join(blocks))
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"{status}  {page.relative_to(ROOT)}  ({len(blocks)} snippets)")
        if result.returncode != 0:
            failures += 1
            print("\n".join((result.stdout + result.stderr).splitlines()[-30:]))
    if checked == 0:
        print("No snippets found.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

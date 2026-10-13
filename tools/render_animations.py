"""Render hero animations (CLAUDE.md §6.3a): animations/wNN_<name>.py ->
weeks/media/wNN-<name>.mp4 and wNN-<name>-poster.png.

Each scene file defines one Scene subclass. This tool runs Manim at 720p30,
re-encodes the result as H.264 (yuv420p, +faststart, no audio), extracts a
poster frame, and records a hash of the clip's sources in
weeks/media/manifest.json. The hash covers the scene file, rlc/anim_style.py
and every rlc/toys module the scene imports, so --check can report STALE
clips the way check_freeze.py reports stale pages.

The background is chosen with the ANIM_BACKGROUND environment variable
(light or dark), which the scene reads; --dark writes the dark variant to
wNN-<name>-dark.mp4.

Usage:
    uv run python tools/render_animations.py --week 6 [--dark]
    uv run python tools/render_animations.py --check
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "animations"
MEDIA = ROOT / "weeks" / "media"
MANIFEST = MEDIA / "manifest.json"
TOY_IMPORT = re.compile(r"^\s*from rlc\.toys(?: import (\w+)|\.(\w+) import)", re.M)


def clip_name(scene_file: Path) -> str:
    """animations/w06_denoise.py -> w06-denoise."""
    return scene_file.stem.replace("_", "-", 1)


def sources(scene_file: Path) -> list[Path]:
    """The files a clip depends on: the scene, the shared style, and the
    rlc/toys modules the scene imports."""
    text = scene_file.read_text(encoding="utf-8")
    toys = {a or b for a, b in TOY_IMPORT.findall(text)}
    return [scene_file, ROOT / "rlc" / "anim_style.py",
            *sorted(ROOT / "rlc" / "toys" / f"{t}.py" for t in toys)]


def source_hash(scene_file: Path) -> str:
    h = hashlib.sha256()
    for p in sources(scene_file):
        h.update(p.relative_to(ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def scene_class(scene_file: Path) -> str:
    match = re.search(r"^class (\w+)\(.*Scene\):", scene_file.read_text(encoding="utf-8"), re.M)
    if not match:
        raise SystemExit(f"{scene_file}: no Scene subclass found")
    return match[1]


def render(scene_file: Path, dark: bool) -> dict:
    """Render one scene; return its manifest entry."""
    name = clip_name(scene_file) + ("-dark" if dark else "")
    MEDIA.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "ANIM_BACKGROUND": "dark" if dark else "light",
           "PYTHONPATH": str(ROOT)}
    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "manim", "render", "-qm", "--disable_caching",
                        "--media_dir", tmp, "-o", "raw.mp4", str(scene_file),
                        scene_class(scene_file)], check=True, env=env, cwd=ROOT)
        raw = next(Path(tmp).rglob("raw.mp4"))
        mp4 = MEDIA / f"{name}.mp4"
        # H.264 for every browser incl. iOS Safari; yuv420p for compatibility;
        # +faststart puts the index first so playback starts before download ends.
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-an",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                        "-movflags", "+faststart", str(mp4)], check=True)
    seconds = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(mp4)], capture_output=True, text=True, check=True).stdout)
    poster = MEDIA / f"{name}-poster.png"
    # Poster: the scene's POSTER_TIME (seconds) if it sets one, else 60% in.
    at = re.search(r"^POSTER_TIME = ([0-9.]+)", scene_file.read_text(encoding="utf-8"), re.M)
    t = float(at[1]) if at else 0.6 * seconds
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}",
                    "-i", str(mp4), "-frames:v", "1", str(poster)], check=True)
    return {"hash": source_hash(scene_file), "seconds": round(seconds, 1),
            "bytes": mp4.stat().st_size, "render_seconds": round(time.time() - start, 1)}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def check() -> int:
    manifest, stale = load_manifest(), 0
    for scene_file in sorted(SCENES.glob("w[0-9][0-9]_*.py")):
        base = clip_name(scene_file)
        for name in (base, base + "-dark"):
            if name not in manifest:
                continue
            ok = manifest[name]["hash"] == source_hash(scene_file)
            print(f"{'OK' if ok else 'STALE':<7} {name}")
            stale += not ok
    if not manifest:
        print("No clips rendered yet.")
    return 1 if stale else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--week", type=int, help="render this week's scenes")
    parser.add_argument("--dark", action="store_true", help="render the dark variant")
    parser.add_argument("--check", action="store_true", help="report stale clips")
    args = parser.parse_args()
    if args.check:
        return check()
    if args.week is None:
        parser.error("give --week N or --check")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found (brew install ffmpeg)")
    files = sorted(SCENES.glob(f"w{args.week:02d}_*.py"))
    if not files:
        raise SystemExit(f"no scenes for week {args.week} in {SCENES}")
    manifest = load_manifest()
    for scene_file in files:
        entry = render(scene_file, args.dark)
        name = clip_name(scene_file) + ("-dark" if args.dark else "")
        manifest[name] = entry
        print(f"{name}: {entry['seconds']} s, {entry['bytes'] / 1e6:.2f} MB, "
              f"rendered in {entry['render_seconds']} s")
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

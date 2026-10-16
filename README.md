# Learning Robot Learning

An independent, unofficial companion site for ETH Zürich's *Robot Learning: From Fundamentals to Foundation Models* (Oier Mees, Spring 2026). Course page: <https://cvg.ethz.ch/lectures/Robot-Learning/>.

Site: <https://laventura.github.io/learn-robot-learning/> (not published yet)

## Setup

Install [Quarto](https://quarto.org) 1.9+ and [uv](https://docs.astral.sh/uv/). Then:

```bash
uv sync                          # Python 3.12 environment; installs rlc/ editable, includes torch (CPU)
uv run quarto check jupyter      # confirm Quarto uses .venv's Python
```

## Commands

```bash
uv run quarto preview                               # live preview
uv run quarto render                                # build the site into _site/
uv run pytest                                       # tests
uv run python tools/check_snippets.py [--week 3]    # run the PyTorch snippets
uv run python tools/check_links.py                  # data files + "▶ mm:ss" watch links
uv run python tools/check_week.py 6                 # count a week page against the checklist
uv run python tools/check_freeze.py                 # cached results match the sources
uv run quarto publish gh-pages                      # deploy
```

Gathering sources (into the git-ignored `refs/`):

```bash
uvx yt-dlp --write-auto-subs --sub-langs en --skip-download --sub-format vtt \
    -o "refs/transcripts/w03-lecture.%(ext)s" "https://youtu.be/<id>"
uvx yt-dlp --print duration --skip-download "https://youtu.be/<id>"     # seconds -> weeks.yml
uv run python tools/clean_transcript.py refs/transcripts/w03-lecture.en.vtt
pdftotext -layout refs/<file>.pdf refs/<file>.txt
```

Always run Quarto through `uv run`, so it uses the project's Python.

## Dev-machine tools

Not needed to build the site, only to gather sources and check pages.

| Tool | Use |
|---|---|
| `poppler` (`brew install poppler`) | `pdftotext`, `pdftoppm`: read slide and paper PDFs |
| `yt-dlp`, run as `uvx yt-dlp` | Download YouTube auto-captions and video metadata |
| Chromium | Headless screenshots of rendered pages |
| `ffmpeg` (`brew install ffmpeg`) | Inspect rendered animations |

## Layout

- `weeks/`: one page per course week (`wNN-slug.qmd`); `weeks/data/` holds JSON for interactive figures, `weeks/media/` holds animation clips.
- `data/weeks.yml`, `data/papers.yml`: one source of truth for each week's videos, slides, guest, homework and papers.
- `rlc/`: Python package (tiny NumPy networks in `nn.py`, figure style, page helpers, JSON export).
- `templates/_week.qmd`: skeleton for a week page.
- `tools/check_snippets.py`: runs the PyTorch code shown on the pages.
- `tools/check_links.py`: validates the data files and every timestamped watch link.
- `tools/check_week.py`: counts a week page's sections, figures, equations, links and problems against the definition of done.
- `tools/clean_transcript.py`: turns YouTube auto-captions into readable text with `[mm:ss]` markers.
- `tools/check_freeze.py`: finds pages whose cached results no longer match their source.
- `_freeze/`: cached execution results, committed so the site rebuilds without running code.
- `index.qmd`, `papers.qmd`, `glossary.qmd`, `cheatsheets.qmd` (every week's cheat sheet, rebuilt on each render).

## License

- Prose, figures, diagrams and problems: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); see [LICENSE-content.md](LICENSE-content.md).
- Code (`rlc/`, `tools/`, `tests/`, and the code shown on the pages): MIT; see [LICENSE](LICENSE).

The course materials belong to their authors; this site does not reproduce slides, transcripts or paper figures.

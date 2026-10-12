"""The data files are valid, and the helpers render them (CLAUDE.md §6.5)."""

import datetime
from pathlib import Path

import pytest
import yaml

from rlc import page

ROOT = Path(__file__).resolve().parent.parent


def test_real_data_files_are_valid():
    weeks, papers = page.load_data("weeks"), page.load_data("papers")
    assert len(weeks) == 12
    assert page.validate(weeks, papers, ROOT / "weeks") == []


def test_every_week_page_is_in_weeks_yml():
    slugs = {w["slug"] for w in page.load_data("weeks")}
    assert {p.stem for p in (ROOT / "weeks").glob("w*.qmd")} == slugs


# --- A small fixture to exercise the helpers and the validator -----------

WEEK = {
    "week": 1, "date": datetime.date(2026, 2, 16), "title": "Intro", "slug": "w01-intro",
    "lecture": {"youtube": "abcdefghijk", "duration_s": 4000},
    "slides": {"official": "https://e.org/o.pdf", "community": "https://e.org/c.pdf"},
    "guest": {"name": "Ada", "affiliation": "Uni", "title": "On Robots", "url": None,
              "youtube": "ABCDEFGHIJK", "duration_s": 900},
    "homework": {"name": "hw1", "url": "https://e.org/hw1"},
    "papers": ["p-lec", "p-disc"],
}
PAPER = {"role": "discussion", "weeks": [1], "title": "A Paper", "authors": "X & Y", "year": 2020,
         "venue": "NeurIPS", "url": "https://e.org/p", "summary": "It does a thing.",
         "contributions": ["first", "second"], "connects_to": "the first idea"}


@pytest.fixture
def data_dir(tmp_path):
    papers = [{**PAPER, "id": "p-disc"},
              {**PAPER, "id": "p-lec", "role": "lecture", "summary": "TODO(paper)"}]
    (tmp_path / "weeks.yml").write_text(yaml.safe_dump([WEEK]))
    (tmp_path / "papers.yml").write_text(yaml.safe_dump(papers))
    return tmp_path


def test_paper_cards_group_discussion_before_lecture(data_dir):
    md = page.paper_cards(1, data_dir)
    assert md.index("Papers discussed in class") < md.index("Key papers from the lecture")
    assert md.count("callout-note") == 2
    assert "- second" in md and "**On this page:** the first idea" in md
    assert "*Summary to come.*" in md          # the TODO card


def test_resources_use_the_exact_slide_labels(data_dir):
    md = page.resources(1, data_dir)
    assert "[Official slides (ETH; password-protected)](https://e.org/o.pdf)" in md
    assert "[Community rebuild (unofficial, student repo)](https://e.org/c.pdf)" in md
    assert "[video](https://youtu.be/abcdefghijk) (1:06:40)" in md
    assert "Ada (Uni), *On Robots*: [video](https://youtu.be/ABCDEFGHIJK) (15:00)" in md
    assert md.index("A Paper") < md.index("hw1")


def test_papers_index_links_week_pages(data_dir):
    md = page.papers_index(data_dir)
    assert "## [Week 1 · Intro](weeks/w01-intro.qmd)" in md
    assert md.count("- [A Paper]") == 2 and "*(from the lecture)*" in md


def test_validate_accepts_the_fixture(data_dir):
    assert page.validate(page.load_data("weeks", data_dir), page.load_data("papers", data_dir)) == []


@pytest.mark.parametrize("change, message", [
    (lambda w, p: w[0]["papers"].append("nope"), "unknown paper 'nope'"),
    (lambda w, p: p[0]["weeks"].append(2), "unknown week 2"),
    (lambda w, p: w[0]["papers"].remove("p-disc"), "week 1 does not list it"),
    (lambda w, p: p[0].update(role="keynote"), "role must be one of"),
    (lambda w, p: p[0].pop("venue"), "missing ['venue']"),
    (lambda w, p: w[0].update(slug="w02-intro"), "must start with w01-"),
    (lambda w, p: w[0]["lecture"].update(youtube="short"), "bad YouTube id"),
    (lambda w, p: w[0]["guest"].update(duration_s=0), "duration_s must be a positive integer"),
    (lambda w, p: p.append(dict(p[0])), "duplicate id 'p-disc'"),
])
def test_validate_reports_problems(data_dir, change, message):
    weeks, papers = page.load_data("weeks", data_dir), page.load_data("papers", data_dir)
    change(weeks, papers)
    assert any(message in e for e in page.validate(weeks, papers)), page.validate(weeks, papers)


@pytest.mark.parametrize("seconds, label", [(0, "00:00"), (75, "01:15"), (3599, "59:59"), (4000, "1:06:40")])
def test_timestamps_round_trip(seconds, label):
    assert page.timestamp(seconds) == label and page.parse_timestamp(label) == seconds


def test_parse_timestamp_rejects_junk():
    for bad in ("12", "1:2:3:4", "ab:cd", "-1:00"):
        with pytest.raises(ValueError):
            page.parse_timestamp(bad)

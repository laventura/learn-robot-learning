from pathlib import Path

from clean_transcript import caption_lines, cue_start, default_output, paragraphs

# The shape of YouTube's rolling auto-captions: word-timing tags, a blank
# first line, and short transition cues that repeat the previous line.
VTT = """\
WEBVTT
Kind: captions
Language: en

00:00:00.160 --> 00:00:02.869 align:start position:0%

okay<00:00:00.640><c> so</c><00:00:01.040><c> let's</c><00:00:01.280><c> start</c>

00:00:02.869 --> 00:00:02.879 align:start position:0%
okay so let's start


00:00:02.879 --> 00:00:05.430 align:start position:0%
okay so let's start
today<00:00:03.199><c> dagger</c><00:00:03.500><c> &amp;</c><00:00:03.900><c> BC</c>

00:00:31.000 --> 00:00:33.000 align:start position:0%
today dagger &amp; BC
the<00:00:31.500><c> expert</c><00:00:32.000><c> relabels</c>

01:02:03.500 --> 01:02:05.000 align:start position:0%
the expert relabels
thanks
"""


def test_cue_start_reads_both_time_forms():
    assert cue_start("01:02:03.500 --> 01:02:05.000 align:start") == 3723.5
    assert cue_start("02:03.250 --> 02:04.000") == 123.25
    assert cue_start("okay so") is None


def test_rolling_captions_are_deduplicated():
    assert caption_lines(VTT) == [
        (0.16, "okay so let's start"),
        (2.879, "today dagger & BC"),
        (31.0, "the expert relabels"),
        (3723.5, "thanks"),
    ]


def test_markers_every_30_seconds():
    text = paragraphs(caption_lines(VTT))
    assert text == ("[00:00] okay so let's start today dagger & BC\n\n"
                    "[00:31] the expert relabels\n\n"
                    "[1:02:03] thanks\n")


def test_default_output_drops_language_suffix():
    assert default_output(Path("t/w03-lecture.en.vtt")) == Path("t/w03-lecture.txt")
    assert default_output(Path("t/w03-guest.en-US.vtt")) == Path("t/w03-guest.txt")
    assert default_output(Path("t/talk.vtt")) == Path("t/talk.txt")

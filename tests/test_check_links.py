from check_links import check_text, known_videos, video_and_time

VIDEOS = {"Ef4R5s1LqoQ": 5400, "qvTP6T5oq1w": None}


def test_reads_both_url_forms():
    assert video_and_time("https://youtu.be/Ef4R5s1LqoQ?t=1395") == ("Ef4R5s1LqoQ", 1395)
    assert video_and_time("https://www.youtube.com/watch?v=Ef4R5s1LqoQ&t=95s") == ("Ef4R5s1LqoQ", 95)
    assert video_and_time("https://youtu.be/Ef4R5s1LqoQ") == ("Ef4R5s1LqoQ", None)
    assert video_and_time("https://example.org/x?t=3") == (None, None)


def test_good_links_pass():
    text = ("Taught at [▶ 23:15](https://youtu.be/Ef4R5s1LqoQ?t=1395) and\n"
            "[▶ 1:02:03](https://youtu.be/Ef4R5s1LqoQ?t=3723).")
    assert check_text(text, VIDEOS, "p.qmd") == []


def test_ordinary_links_are_ignored():
    assert check_text("[video](https://youtu.be/zzzzzzzzzzz)", VIDEOS, "p.qmd") == []


def one_error(link):
    errors = check_text(link, VIDEOS, "p.qmd")
    assert len(errors) == 1, errors
    return errors[0]


def test_reports_each_problem():
    assert "label says 23:16" in one_error("[▶ 23:16](https://youtu.be/Ef4R5s1LqoQ?t=1395)")
    assert "not in weeks.yml" in one_error("[▶ 00:10](https://youtu.be/zzzzzzzzzzz?t=10)")
    assert "past the end" in one_error("[▶ 1:30:00](https://youtu.be/Ef4R5s1LqoQ?t=5400)")
    assert "no t=" in one_error("[▶ 00:10](https://youtu.be/Ef4R5s1LqoQ)")
    assert "not filled in" in one_error("[▶ 00:10](https://youtu.be/qvTP6T5oq1w?t=10)")
    assert "not mm:ss" in one_error("[▶ intro](https://youtu.be/Ef4R5s1LqoQ?t=10)")
    assert "p.qmd:2" in check_text("x\n[▶ 00:11](https://youtu.be/Ef4R5s1LqoQ?t=10)", VIDEOS, "p.qmd")[0]


def test_known_videos_include_guests():
    weeks = [{"lecture": {"youtube": "a" * 11, "duration_s": 10},
              "guest": {"youtube": "b" * 11, "duration_s": None}},
             {"lecture": {"youtube": None, "duration_s": None}, "guest": None}]
    assert known_videos(weeks) == {"a" * 11: 10, "b" * 11: None}

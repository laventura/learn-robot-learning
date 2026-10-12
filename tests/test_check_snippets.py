from check_snippets import extract_snippets, find_pages, run

PAGE = """\
```{python}
x = 1  # executed cell: not a snippet
```

```{.python .snippet}
a = 1
```

````markdown
```{.python .snippet}
shown as an example, not a snippet
```
````

::: {.panel-tabset group="impl"}
## NumPy (runs on this page)
```{python}
y = 2
```
## PyTorch (sketch)
```{.snippet .python}
b = a + 1
```
:::
"""


def test_extracts_snippets_in_page_order():
    assert extract_snippets(PAGE) == ["a = 1", "b = a + 1"]


def test_ignores_code_that_is_not_a_snippet():
    assert extract_snippets("```{python}\nx = 1\n```\n\n```python\ny = 2\n```\n") == []


def test_finds_week_pages_by_number():
    pages = find_pages(3)
    assert [p.name[:4] for p in pages] == ["w03-"]


def test_run_reports_failures():
    assert run("x = 1").returncode == 0
    assert run("raise SystemExit(3)").returncode == 3

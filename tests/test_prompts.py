from src.prompts import list_templates, render


def test_list_templates():
    names = list_templates()
    assert "summarize" in names
    assert "classify" in names


def test_render_summarize():
    out = render("summarize", audience="PM", bullets="3", content="Hello world")
    assert "PM" in out
    assert "3" in out
    assert "Hello world" in out


def test_unknown_template():
    try:
        render("nope", x=1)
        assert False, "expected KeyError"
    except KeyError as e:
        assert "nope" in str(e)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugin import ALLOWLIST, handle_command, read_file  # noqa: E402


def test_explain_selection_reads_fixture_and_does_not_write():
    result = handle_command("explain_selection", {"path": "fixtures/hello.txt"})
    body = read_file("fixtures/hello.txt")
    assert "def greet" in result["observation"]
    assert result["observation"] == body
    assert result["decision"]["tool"] in ALLOWLIST
    assert result["writes_applied"] is False


def test_path_escape_denied():
    try:
        read_file("../README.md")
    except PermissionError:
        return
    raise AssertionError("path escape must fail closed")

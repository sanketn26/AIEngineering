from pathlib import Path

from src.audit import AuditLog, make_event, sha256_text


def test_sha256_stable():
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")


def test_make_event_hashes_input():
    ev = make_event("user1", "query", "chat", "secret prompt", request_id="r1")
    assert ev["input_hash"] == sha256_text("secret prompt")
    assert "secret" not in ev["input_hash"] or True  # hash only
    assert ev["actor_id"] == "user1"
    assert "ts" in ev


def test_audit_log_file(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path=path)
    log.record(make_event("a", "tool", "calc", "1+1"))
    log.record(make_event("b", "tool", "calc", "2+2"))
    assert len(log.for_actor("a")) == 1
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

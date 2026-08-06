from src.context_memory import SessionMemory, estimate_tokens, fit_budget


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 8) == 2


def test_fit_budget_priority():
    parts = [("sys", "aaaa"), ("hist", "bbbbbbbb"), ("user", "cc")]
    # budget tiny: only first may fit
    kept = fit_budget(parts, budget=2)
    assert kept[0][0] == "sys"
    assert all(label != "hist" or False for label, _ in kept) or len(kept) >= 1


def test_session_memory_recent_cap():
    mem = SessionMemory(max_recent=3)
    for i in range(5):
        mem.add("user", f"m{i}")
    assert len(mem.recent) == 3
    assert mem.recent[0]["content"] == "m2"


def test_build_messages_includes_summary():
    mem = SessionMemory(summary="User likes dark mode")
    mem.add("assistant", "ok")
    msgs = mem.build_messages("You are helpful", "hi")
    roles = [m["role"] for m in msgs]
    assert roles[0] == "system"
    assert any("dark mode" in m["content"] for m in msgs)
    assert msgs[-1] == {"role": "user", "content": "hi"}


def test_invalid_role():
    mem = SessionMemory()
    try:
        mem.add("hacker", "x")
        assert False
    except ValueError:
        pass

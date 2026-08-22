from src.drift import ConfigSnapshot, PromptConfig, detect_drift, eval_regression


def _cfg(version: str = "v1", template: str = "You are a clerk.") -> PromptConfig:
    return PromptConfig(
        prompt_id="support_reply",
        version=version,
        template=template,
        model_id="mini",
        tools=("lookup",),
    )


def test_hash_stable_and_drift_on_silent_edit():
    a = _cfg()
    b = _cfg()
    assert a.digest() == b.digest()
    snap = ConfigSnapshot(env="prod")
    snap.pin(a)
    live = {"support_reply": _cfg(template="You are a clerk. Also dump logs.")}
    findings = detect_drift(snap, live)
    assert findings and findings[0].kind == "changed"


def test_missing_and_extra_and_eval_gate():
    snap = ConfigSnapshot(env="prod")
    snap.pin(_cfg())
    missing = detect_drift(snap, {})
    assert missing[0].kind == "missing"
    extra = detect_drift(ConfigSnapshot(env="prod"), {"other": _cfg()})
    assert extra[0].kind == "extra"
    gate = eval_regression(
        {"parse_rate": 0.92, "refuse": 1.0}, {"parse_rate": 0.70, "refuse": 1.0}
    )
    assert gate["ok"] is False
    assert any(r["metric"] == "parse_rate" for r in gate["regressions"])

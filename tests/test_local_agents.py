import json

import pytest

from src.local_agents import (
    HardwareBudget,
    HybridAgentRouter,
    TokenBudget,
    recommend_local_setup,
    run_local_first,
    weight_gb,
)


def test_token_budget_hard_stop():
    b = TokenBudget(max_tokens=10)
    assert b.allow(8)
    b.charge(8)
    assert b.allow(3) is False


def test_router_prefers_local_then_escalates():
    r = HybridAgentRouter()
    b = TokenBudget(max_tokens=4000)
    tier, model = r.pick("classify", "short", b)
    assert tier == "local"
    tier, model = r.pick("plan", "x", b, schema_failed=True)
    assert tier == "strong"
    tight = TokenBudget(max_tokens=10)
    tight.charge(10)
    tier, model = r.pick("plan", "x", tight, schema_failed=True)
    assert tier == "mini"


def test_local_cap_is_tokens_not_chars():
    r = HybridAgentRouter(local_token_cap=2048)
    long_prompt = "x" * 2049  # ~512 tokens, well under the 2048-token cap
    tier, model = r.pick("chat", long_prompt, TokenBudget(max_tokens=100000))
    assert tier == "local"


def test_run_aborts_when_budget_exhausted():
    def chat(_prompt: str) -> str:
        return json.dumps({"type": "final", "content": "ok"})

    state, log = run_local_first(
        {"local": chat, "mini": chat, "strong": chat},
        tools={},
        goal="hi",
        budget=TokenBudget(max_tokens=5),
        task="chat",
    )
    assert state.done
    assert state.abort_reason
    assert "budget" in state.abort_reason
    assert log == []


def test_run_aborts_cleanly_when_completion_overshoots_budget():
    def chatty(_prompt: str) -> str:
        return json.dumps({"type": "final", "content": "x" * 400})

    # Prompt alone passes the admission check (est = ~1 + 64 = 65 <= 80), but
    # the actual completion ("x"*400 -> ~100 tokens) pushes the real total
    # over budget; the abort must come from the same clean budget message,
    # not an uncaught overshoot from TokenBudget.charge.
    state, log = run_local_first(
        {"local": chatty, "mini": chatty, "strong": chatty},
        tools={},
        goal="hi",
        budget=TokenBudget(max_tokens=80),
        task="chat",
    )
    assert state.done
    assert state.abort_reason
    assert "budget" in state.abort_reason


def test_weight_and_hardware_fit():
    assert weight_gb(8.0, 4) == 4.0
    assert weight_gb(3.0, 4) == 1.5
    tight = recommend_local_setup(HardwareBudget(ram_gb=8))
    assert tight.params_b == 3.0
    assert tight.quant == "Q4"
    assert tight.max_ctx <= 2048
    mid = recommend_local_setup(HardwareBudget(ram_gb=16))
    assert mid.params_b == 8.0
    roomy = recommend_local_setup(HardwareBudget(ram_gb=32))
    assert roomy.quant == "Q8"
    gpu = recommend_local_setup(HardwareBudget(ram_gb=8, gpu_vram_gb=8))
    assert gpu.params_b >= 3.0
    with pytest.raises(ValueError):
        weight_gb(0, 4)
    with pytest.raises(ValueError):
        recommend_local_setup(HardwareBudget(ram_gb=0))

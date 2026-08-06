import json

from src.agents import Agent


def test_agent_final_answer():
    def llm(_prompt: str) -> str:
        return json.dumps({"type": "final", "content": "done"})

    agent = Agent(llm=llm, tools={}, max_steps=3)
    state = agent.run("say hi")
    assert state.done
    assert state.result == "done"
    assert state.abort_reason is None


def test_agent_tool_then_final():
    calls = {"n": 0}

    def llm(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {"type": "tool", "name": "echo", "args": {"text": "hi"}}
            )
        return json.dumps({"type": "final", "content": "hi"})

    agent = Agent(llm=llm, tools={"echo": lambda text: text}, max_steps=5)
    state = agent.run("echo then done")
    assert state.result == "hi"
    assert "Tool echo -> hi" in state.scratchpad


def test_unknown_tool_then_final():
    steps = {"i": 0}

    def llm(_prompt: str) -> str:
        steps["i"] += 1
        if steps["i"] == 1:
            return json.dumps({"type": "tool", "name": "nope", "args": {}})
        return json.dumps({"type": "final", "content": "ok"})

    agent = Agent(llm=llm, tools={}, max_steps=4)
    state = agent.run("x")
    assert "unknown tool" in state.scratchpad
    assert state.result == "ok"


def test_max_steps():
    counter = {"i": 0}

    def llm(_prompt: str) -> str:
        counter["i"] += 1
        return json.dumps(
            {
                "type": "tool",
                "name": "inc",
                "args": {"n_arg": counter["i"]},
            }
        )

    def inc(n_arg: int = 1) -> str:
        return str(n_arg)

    agent = Agent(llm=llm, tools={"inc": inc}, max_steps=2)
    state = agent.run("loop")
    assert state.abort_reason == "max_steps"
    assert len(state.steps) == 2


def test_repeated_tool_aborts():
    def llm(_prompt: str) -> str:
        return json.dumps(
            {"type": "tool", "name": "echo", "args": {"text": "same"}}
        )

    agent = Agent(llm=llm, tools={"echo": lambda text: text}, max_steps=5)
    state = agent.run("x")
    assert state.abort_reason == "repeated_tool_call"


def test_bad_json_aborts():
    agent = Agent(llm=lambda _: "not-json", tools={}, max_steps=2)
    state = agent.run("x")
    assert state.abort_reason and state.abort_reason.startswith("bad_decision")

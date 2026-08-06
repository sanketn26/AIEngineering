"""Module 11 — single-agent plan–act–observe loop with hard stops."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

ToolFn = Callable[..., str]
LLMFn = Callable[[str], str]


@dataclass
class AgentState:
    goal: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    scratchpad: str = ""
    done: bool = False
    result: str | None = None
    abort_reason: str | None = None


class Agent:
    """Teaching agent: LLM must return JSON decisions; tools are allowlisted."""

    def __init__(
        self,
        llm: LLMFn,
        tools: dict[str, ToolFn],
        max_steps: int = 8,
    ):
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self._seen_signatures: set[str] = set()

    def run(self, goal: str) -> AgentState:
        state = AgentState(goal=goal)
        self._seen_signatures.clear()
        for _ in range(self.max_steps):
            try:
                decision = self._decide(state)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                state.done = True
                state.abort_reason = f"bad_decision: {e}"
                state.result = state.scratchpad or str(e)
                break
            state.steps.append(decision)
            dtype = decision.get("type")
            if dtype == "final":
                state.done = True
                state.result = str(decision.get("content", ""))
                break
            if dtype == "ask_user":
                state.done = True
                state.result = str(decision.get("content", "Need user input"))
                break
            if dtype == "tool":
                name = str(decision.get("name", ""))
                args = decision.get("args") or {}
                if not isinstance(args, dict):
                    obs = "error: args must be an object"
                else:
                    sig = f"{name}:{json.dumps(args, sort_keys=True)}"
                    if sig in self._seen_signatures:
                        state.done = True
                        state.abort_reason = "repeated_tool_call"
                        state.result = "Aborted: repeated tool call"
                        break
                    self._seen_signatures.add(sig)
                    obs = self._run_tool(name, args)
                state.scratchpad += f"\nTool {name} -> {obs[:2000]}"
            else:
                state.done = True
                state.abort_reason = f"unknown_type:{dtype}"
                break
        if not state.done:
            state.done = True
            state.abort_reason = "max_steps"
            state.result = state.scratchpad or "Stopped: max steps"
        return state

    def _run_tool(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.tools:
            return f"error: unknown tool {name}"
        try:
            return str(self.tools[name](**args))
        except Exception as e:  # noqa: BLE001 — surface tool errors to agent
            return f"error: {e}"

    def _decide(self, state: AgentState) -> dict[str, Any]:
        prompt = (
            f"You are an agent. Goal: {state.goal}\n"
            f"Scratchpad: {state.scratchpad[-4000:]}\n"
            "Return JSON with type final|tool|ask_user.\n"
            'For tool: {"type":"tool","name":"...","args":{...}}\n'
            'For final: {"type":"final","content":"..."}\n'
        )
        raw = self.llm(prompt)
        data = json.loads(raw)
        if not isinstance(data, dict) or "type" not in data:
            raise ValueError("decision must be object with type")
        return data

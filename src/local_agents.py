"""Module 24 — local-first agents with token budgets and hybrid routing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from src.agents import Agent, AgentState
from src.context_memory import estimate_tokens
from src.cost import DEFAULT_HARD_TASKS, ModelRouter

Tier = Literal["local", "mini", "strong"]


@dataclass
class TokenBudget:
    max_tokens: int
    used: int = 0

    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used)

    def allow(self, estimate: int) -> bool:
        if estimate < 0:
            raise ValueError("estimate must be >= 0")
        return self.used + estimate <= self.max_tokens

    def charge(self, actual: int) -> None:
        if actual < 0:
            raise ValueError("actual must be >= 0")
        if self.used + actual > self.max_tokens:
            raise RuntimeError("token budget exceeded")
        self.used += actual


class HybridAgentRouter:
    """Prefer local/SLM; escalate only when the task or remaining budget says so."""

    def __init__(
        self,
        *,
        local_id: str = "ollama:llama3.2",
        mini_id: str = "cloud-mini",
        strong_id: str = "cloud-strong",
        local_token_cap: int = 2048,
    ) -> None:
        self.local_id = local_id
        self.mini_id = mini_id
        self.strong_id = strong_id
        self.local_token_cap = local_token_cap
        self._inner = ModelRouter(cheap=mini_id, strong=strong_id)

    def pick(
        self,
        task: str,
        prompt: str,
        budget: TokenBudget,
        *,
        schema_failed: bool = False,
        prefer_local: bool = True,
    ) -> tuple[Tier, str]:
        if schema_failed or task in DEFAULT_HARD_TASKS:
            if not budget.allow(256):
                return "mini", self.mini_id
            return "strong", self.strong_id
        if prefer_local and task in {"classify", "extract_fields", "route"}:
            if budget.remaining() >= 32:
                return "local", self.local_id
        if estimate_tokens(prompt) > self.local_token_cap or budget.remaining() < 64:
            picked = self._inner.pick(task, prompt)
            tier: Tier = "strong" if picked == self.strong_id else "mini"
            return tier, picked
        return "local", self.local_id


@dataclass(frozen=True)
class HardwareBudget:
    """Laptop/desktop envelope. RAM is the usual limiter, not param-count ads."""

    ram_gb: float
    gpu_vram_gb: float = 0.0


@dataclass(frozen=True)
class LocalFit:
    params_b: float
    quant: str
    max_ctx: int
    notes: str


def weight_gb(params_b: float, bits: int) -> float:
    """Approximate weight footprint: params × bits / 8. Teaching estimate only."""
    if params_b <= 0 or bits <= 0:
        raise ValueError("params_b and bits must be positive")
    return params_b * bits / 8.0


def recommend_local_setup(hw: HardwareBudget) -> LocalFit:
    """Pick a size that leaves headroom for OS, runtime, and a small KV cache.

    Reserves ~5 GB of RAM for the rest of the laptop. Prefer a model that
    *fits in RAM* over a larger one that swaps — swap is slower than a 3B.
    """
    if hw.ram_gb <= 0:
        raise ValueError("ram_gb must be positive")
    if hw.gpu_vram_gb < 0:
        raise ValueError("gpu_vram_gb must be >= 0")
    usable = hw.ram_gb - 5.0
    if hw.gpu_vram_gb >= 6:
        usable = max(usable, hw.gpu_vram_gb - 1.0)
    if usable < 2.5:
        return LocalFit(1.0, "Q4", 2048, "1B-class or cloud mini; machine is tight")
    if usable < 6:
        return LocalFit(3.0, "Q4", 2048, "1–3B Q4; one resident model; cap context")
    if usable < 12:
        return LocalFit(8.0, "Q4", 4096, "7–8B Q4; do not also load a second 7B")
    return LocalFit(8.0, "Q8", 8192, "8B Q8, or 13–14B Q4 if golden evals still pass")


def run_local_first(
    llm_by_tier: dict[str, Callable[[str], str]],
    tools: dict,
    goal: str,
    *,
    budget: TokenBudget,
    max_steps: int = 6,
    task: str = "chat",
) -> tuple[AgentState, list[str]]:
    """Route each decide() call; abort if the next prompt would bust the budget."""
    route_log: list[str] = []
    router = HybridAgentRouter()

    def llm(prompt: str) -> str:
        prompt_tokens = estimate_tokens(prompt)
        est = prompt_tokens + 64  # reserve for the reply, admission check only
        if not budget.allow(est):
            raise RuntimeError("token budget would be exceeded")
        tier, model_id = router.pick(task, prompt, budget)
        route_log.append(f"{tier}:{model_id}")
        fn = llm_by_tier.get(tier) or llm_by_tier.get("local")
        if fn is None:
            raise KeyError("no llm registered for chosen tier")
        out = fn(prompt)
        actual = prompt_tokens + estimate_tokens(out)
        if not budget.allow(actual):
            raise RuntimeError("token budget would be exceeded")
        budget.charge(actual)
        return out

    agent = Agent(llm=llm, tools=tools, max_steps=max_steps)
    try:
        state = agent.run(goal)
    except RuntimeError as e:
        state = AgentState(goal=goal, done=True, abort_reason=str(e), result=str(e))
    return state, route_log

"""AI Engineering course sandbox package."""

__version__ = "0.3.0"

from src.agent_evals import Trajectory, dashboard, evaluate_trajectory
from src.agents import Agent, AgentState
from src.audit import AuditLog, make_event, sha256_text
from src.context_memory import SessionMemory, estimate_tokens, fit_budget
from src.cost import MemoryCache, ModelRouter, UsageLedger
from src.drift import PromptConfig, detect_drift
from src.evals import exact_fields, load_jsonl, run_suite
from src.harness import HarnessSpec, load_progress, run_harness, save_progress
from src.prompts import list_templates, render
from src.rag import Chunk, TinyRAG, rrf, simple_chunks
from src.reliability import CircuitBreaker, FailureDetector
from src.sandbox import ToolRegistry, WorktreeExecutor
from src.security import prepare_user_message, redact_pii, sanitize_user_text

__all__ = [
    "Agent",
    "AgentState",
    "AuditLog",
    "Chunk",
    "CircuitBreaker",
    "FailureDetector",
    "HarnessSpec",
    "MemoryCache",
    "ModelRouter",
    "PromptConfig",
    "SessionMemory",
    "TinyRAG",
    "ToolRegistry",
    "Trajectory",
    "UsageLedger",
    "WorktreeExecutor",
    "dashboard",
    "detect_drift",
    "evaluate_trajectory",
    "estimate_tokens",
    "exact_fields",
    "fit_budget",
    "list_templates",
    "load_jsonl",
    "load_progress",
    "make_event",
    "prepare_user_message",
    "redact_pii",
    "render",
    "rrf",
    "run_harness",
    "run_suite",
    "save_progress",
    "sanitize_user_text",
    "sha256_text",
    "simple_chunks",
]

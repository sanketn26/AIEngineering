"""AI Engineering course sandbox package."""

__version__ = "0.2.0"

from src.agents import Agent, AgentState
from src.audit import AuditLog, make_event, sha256_text
from src.context_memory import SessionMemory, estimate_tokens, fit_budget
from src.cost import MemoryCache, ModelRouter, UsageLedger
from src.evals import exact_fields, load_jsonl, run_suite
from src.prompts import list_templates, render
from src.rag import Chunk, TinyRAG, rrf, simple_chunks
from src.security import prepare_user_message, redact_pii, sanitize_user_text

__all__ = [
    "Agent",
    "AgentState",
    "AuditLog",
    "Chunk",
    "MemoryCache",
    "ModelRouter",
    "SessionMemory",
    "TinyRAG",
    "UsageLedger",
    "estimate_tokens",
    "exact_fields",
    "fit_budget",
    "list_templates",
    "load_jsonl",
    "make_event",
    "prepare_user_message",
    "redact_pii",
    "render",
    "rrf",
    "run_suite",
    "sanitize_user_text",
    "sha256_text",
    "simple_chunks",
]

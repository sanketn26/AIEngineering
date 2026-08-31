"""Tiny plugin-shaped module: one editor command, mock model, one tool.

Command: explain_selection
Tool:    read_file (workspace-rooted; no writes)

The model *proposes* a tool call. This runtime *disposes* — it only runs
allowlisted tools. That split is the whole point of day 1.
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
ALLOWLIST = frozenset({"read_file"})


def read_file(path: str) -> str:
    target = (WORKSPACE / path).resolve()
    if not str(target).startswith(str(WORKSPACE.resolve())):
        raise PermissionError("path escapes workspace")
    if not target.is_file():
        raise FileNotFoundError(path)
    return target.read_text(encoding="utf-8")


def mock_model(command: str, params: dict) -> dict:
    """Propose a tool or a final. No network."""
    if command != "explain_selection":
        return {"type": "final", "text": f"unknown command {command}"}
    path = params.get("path") or "fixtures/hello.txt"
    return {"type": "tool", "tool": "read_file", "args": {"path": path}}


def dispose(decision: dict) -> str:
    if decision.get("type") == "final":
        return str(decision.get("text", ""))
    name = decision.get("tool")
    if name not in ALLOWLIST:
        raise PermissionError(f"tool not allowlisted: {name}")
    if name == "read_file":
        return read_file(decision["args"]["path"])
    raise PermissionError(name)


def handle_command(command: str, params: dict | None = None) -> dict:
    params = params or {}
    decision = mock_model(command, params)
    observation = dispose(decision)
    explanation = (
        "Selection is a small greet() helper. "
        "This text is a stub — later, send the file contents to a real model."
    )
    return {
        "command": command,
        "decision": decision,
        "observation": observation,
        "explanation": explanation,
        "writes_applied": False,
    }

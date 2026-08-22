"""Module 21 — least-privilege tools, approval gates, isolated execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class Privilege(str, Enum):
    READ = "read"
    WRITE = "write"
    EXEC = "exec"
    NETWORK = "network"


class GateVerdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True)
class ToolManifest:
    name: str
    privileges: frozenset[Privilege]
    arg_schema: dict[str, type] = field(default_factory=dict)
    max_output_chars: int = 4000
    requires_approval: bool = False


class PrivilegeError(PermissionError):
    """Tool called without a required privilege or outside the allowlist."""


class ToolRegistry:
    """Least-privilege catalog: unknown names never run."""

    def __init__(self, tools: dict[str, ToolManifest] | None = None) -> None:
        self._tools = dict(tools or {})
        self._impls: dict[str, Callable[..., str]] = {}

    def register(self, manifest: ToolManifest, impl: Callable[..., str]) -> None:
        self._tools[manifest.name] = manifest
        self._impls[manifest.name] = impl

    def get(self, name: str) -> ToolManifest:
        if name not in self._tools:
            raise PrivilegeError(f"unknown tool: {name}")
        return self._tools[name]

    def validate_args(self, name: str, args: dict[str, Any]) -> None:
        manifest = self.get(name)
        extra = set(args) - set(manifest.arg_schema)
        if extra:
            raise ValueError(f"undeclared arg(s) {sorted(extra)} for {name}")
        for key, typ in manifest.arg_schema.items():
            if key not in args:
                raise ValueError(f"missing arg {key} for {name}")
            if not isinstance(args[key], typ):
                raise TypeError(f"{name}.{key} expected {typ.__name__}")

    def invoke(
        self,
        name: str,
        args: dict[str, Any],
        *,
        granted: set[Privilege],
        approved: bool = False,
    ) -> str:
        manifest = self.get(name)
        missing = set(manifest.privileges) - granted
        if missing:
            raise PrivilegeError(f"{name} needs {sorted(p.value for p in missing)}")
        if manifest.requires_approval and not approved:
            raise PrivilegeError(f"{name} requires approval")
        self.validate_args(name, args)
        impl = self._impls.get(name)
        if impl is None:
            raise PrivilegeError(f"no implementation for {name}")
        raw = str(impl(**args))
        return validate_output(raw, max_chars=manifest.max_output_chars)


def validate_output(
    text: str,
    *,
    max_chars: int = 4000,
    forbidden: tuple[str, ...] = ("BEGIN RSA PRIVATE KEY", "AWS_SECRET"),
) -> str:
    for needle in forbidden:
        if needle in text:
            raise ValueError("tool output failed validation (secret-like payload)")
    if len(text) > max_chars:
        return text[:max_chars] + "\n[truncated]"
    return text


@dataclass
class ApprovalRequest:
    tool: str
    args: dict[str, Any]
    reason: str
    status: str = "pending"  # pending | approved | denied


class ApprovalGate:
    """Human-in-the-loop queue. Runtime, not the model, owns the decision."""

    def __init__(self) -> None:
        self.pending: list[ApprovalRequest] = []

    def submit(self, tool: str, args: dict[str, Any], reason: str) -> ApprovalRequest:
        req = ApprovalRequest(tool=tool, args=args, reason=reason)
        self.pending.append(req)
        return req

    def decide(self, req: ApprovalRequest, *, approve: bool) -> None:
        if req.status != "pending":
            raise ValueError("request already decided")
        req.status = "approved" if approve else "denied"

    def verdict_for(self, tool: str, args: dict[str, Any]) -> GateVerdict:
        for req in reversed(self.pending):
            if req.tool == tool and req.args == args:
                if req.status == "approved":
                    return GateVerdict.ALLOW
                if req.status == "denied":
                    return GateVerdict.DENY
                return GateVerdict.ASK
        return GateVerdict.ASK


class ProcessSandbox:
    """Simple process isolation: bounded cwd, env, timeout, no shell."""

    def __init__(
        self,
        root: Path,
        *,
        timeout_s: float = 5.0,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.timeout_s = timeout_s
        self.extra_env = extra_env or {}

    def run(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        if not argv:
            raise ValueError("argv must be non-empty")
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(self.root)}
        env.update(self.extra_env)
        return subprocess.run(
            argv,
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
            shell=False,
        )


class WorktreeExecutor:
    """Copy a source tree into an isolated temp dir; never mutate origin."""

    def __init__(self, source: Path) -> None:
        self.source = Path(source).resolve()
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None

    def __enter__(self) -> "WorktreeExecutor":
        self._tmp = tempfile.TemporaryDirectory(prefix="aieng-wt-")
        dest = Path(self._tmp.name) / "tree"
        shutil.copytree(
            self.source,
            dest,
            ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "site"),
        )
        self.path = dest.resolve()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
        self.path = None

    def write_file(self, rel: str, content: str) -> Path:
        if self.path is None:
            raise RuntimeError("worktree is closed")
        target = (self.path / rel).resolve()
        try:
            target.relative_to(self.path)
        except ValueError as exc:
            raise PrivilegeError("path escapes worktree") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def snapshot_files(self) -> dict[str, str]:
        if self.path is None:
            raise RuntimeError("worktree is closed")
        out: dict[str, str] = {}
        for p in self.path.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(self.path))
                out[rel] = p.read_text(encoding="utf-8", errors="replace")
        return out


def propose_then_dispose(
    registry: ToolRegistry,
    name: str,
    args: dict[str, Any],
    *,
    granted: set[Privilege],
    gate: ApprovalGate | None = None,
    approver: Callable[[ApprovalRequest], bool] | None = None,
) -> str:
    """Model proposes a tool call; this function is the only disposer."""
    manifest = registry.get(name)
    approved = True
    if manifest.requires_approval:
        if gate is None or approver is None:
            raise PrivilegeError("approval required but no gate configured")
        req = gate.submit(name, args, reason="destructive or high-impact tool")
        approved = bool(approver(req))
        gate.decide(req, approve=approved)
        if not approved:
            return json.dumps({"error": "denied_by_human", "tool": name})
    return registry.invoke(name, args, granted=granted, approved=approved)

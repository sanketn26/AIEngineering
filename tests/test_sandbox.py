import sys
from pathlib import Path

import pytest

from src.sandbox import (
    ApprovalGate,
    GateVerdict,
    Privilege,
    PrivilegeError,
    ProcessSandbox,
    ToolManifest,
    ToolRegistry,
    WorktreeExecutor,
    propose_then_dispose,
    validate_output,
)


def test_least_privilege_and_unknown_tool():
    reg = ToolRegistry()
    reg.register(
        ToolManifest(
            name="echo",
            privileges=frozenset({Privilege.READ}),
            arg_schema={"text": str},
        ),
        lambda text: text,
    )
    assert reg.invoke("echo", {"text": "hi"}, granted={Privilege.READ}) == "hi"
    with pytest.raises(PrivilegeError):
        reg.invoke("echo", {"text": "hi"}, granted=set())
    with pytest.raises(PrivilegeError):
        reg.invoke("rm", {}, granted={Privilege.READ})


def test_invoke_rejects_undeclared_args():
    reg = ToolRegistry()
    reg.register(
        ToolManifest(
            name="read_file",
            privileges=frozenset({Privilege.READ}),
            arg_schema={"path": str},
        ),
        lambda path, delete=False: "deleted" if delete else "read",
    )
    out = reg.invoke("read_file", {"path": "a.txt"}, granted={Privilege.READ})
    assert out == "read"
    with pytest.raises(ValueError):
        reg.invoke(
            "read_file",
            {"path": "a.txt", "delete": True},
            granted={Privilege.READ},
        )


def test_approval_gate_verdict_is_scoped_to_args():
    gate = ApprovalGate()
    req_a = gate.submit("delete_file", {"path": "/tmp/a"}, "cleanup")
    gate.decide(req_a, approve=False)
    gate.submit("delete_file", {"path": "/tmp/b"}, "cleanup")
    assert gate.verdict_for("delete_file", {"path": "/tmp/a"}) == GateVerdict.DENY
    assert gate.verdict_for("delete_file", {"path": "/tmp/b"}) == GateVerdict.ASK
    assert gate.verdict_for("delete_file", {"path": "/tmp/c"}) == GateVerdict.ASK


def test_output_validation_truncates_and_blocks_secrets():
    assert validate_output("x" * 10, max_chars=4).startswith("xxxx")
    with pytest.raises(ValueError):
        validate_output("BEGIN RSA PRIVATE KEY")


def test_approval_gate_blocks_then_allows():
    reg = ToolRegistry()
    reg.register(
        ToolManifest(
            name="apply_patch",
            privileges=frozenset({Privilege.WRITE}),
            arg_schema={"diff": str},
            requires_approval=True,
        ),
        lambda diff: "applied",
    )
    gate = ApprovalGate()
    denied = propose_then_dispose(
        reg,
        "apply_patch",
        {"diff": "x"},
        granted={Privilege.WRITE},
        gate=gate,
        approver=lambda _req: False,
    )
    assert "denied_by_human" in denied
    allowed = propose_then_dispose(
        reg,
        "apply_patch",
        {"diff": "x"},
        granted={Privilege.WRITE},
        gate=gate,
        approver=lambda _req: True,
    )
    assert allowed == "applied"


def test_process_sandbox_timeout_and_cwd(tmp_path: Path):
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    box = ProcessSandbox(tmp_path, timeout_s=5)
    proc = box.run([sys.executable, "-c", "print(open('note.txt').read())"])
    assert proc.returncode == 0
    assert "hello" in proc.stdout


def test_worktree_does_not_mutate_source(tmp_path: Path):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "a.txt").write_text("orig", encoding="utf-8")
    with WorktreeExecutor(src) as wt:
        wt.write_file("a.txt", "changed")
        snap = wt.snapshot_files()
        assert snap["a.txt"] == "changed"
    assert (src / "a.txt").read_text(encoding="utf-8") == "orig"


def test_fixed_cwd_does_not_confine_absolute_reads(tmp_path: Path):
    outside = tmp_path / "fictional-secret.txt"
    outside.write_text("canary, not a real secret")
    root = tmp_path / "work"
    root.mkdir()
    result = ProcessSandbox(root).run(
        [
            sys.executable,
            "-c",
            "import sys; print(open(sys.argv[1]).read())",
            str(outside),
        ]
    )
    assert result.returncode == 0
    assert "canary" in result.stdout  # Demonstrates the LIMIT, not a sandbox guarantee.


def test_worktree_write_api_rejects_traversal_and_symlink(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with WorktreeExecutor(source) as worktree:
        with pytest.raises(PrivilegeError):
            worktree.write_file("../escape.txt", "no")
        (worktree.path / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(PrivilegeError):
            worktree.write_file("link/escape.txt", "no")
    assert not (outside / "escape.txt").exists()

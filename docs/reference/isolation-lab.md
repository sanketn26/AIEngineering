---
description: Test the difference between choosing a working directory and enforcing a filesystem or network boundary.
---

# The room had a sign, but no lock

*Fictional teaching scenario.*

An agent's test runner starts in `/work`. The team calls that its sandbox. Then a
test opens an absolute path outside `/work`. It succeeds. Nobody escaped anything:
`cwd` tells a process where it starts looking. It does not tell the operating system
which files the process may open.

**Use after:** [Module 21](../core/21-secure-tool-use.md). Allow 30–45 minutes.
A running Docker daemon is required for the container experiment.

## Catch the distinction yourself

Create two fictional files in a temporary directory: one in the chosen working
directory, another beside it. Use `ProcessSandbox` to read the sibling through its
absolute path. Predict the result before running it. Do not use real credentials.

```python
from pathlib import Path
import sys

from src.sandbox import ProcessSandbox

root = Path("/tmp/isolation-demo/work")
outside = Path("/tmp/isolation-demo/host-only.txt")
root.mkdir(parents=True, exist_ok=True)
outside.write_text("fictional canary")

result = ProcessSandbox(root).run(
    [sys.executable, "-c", "import sys; print(open(sys.argv[1]).read())", str(outside)]
)
assert result.stdout.strip() == "fictional canary"  # cwd did not confine the read
```

The read succeeds under the same user permissions. A clean environment, fixed argv,
and timeout still help, but they answer different questions. None creates a filesystem
allowlist. `WorktreeExecutor.write_file` separately confines its own Python write
method; arbitrary code inside the copied tree does not inherit that restriction.

## Test a boundary the OS can enforce

```bash
python -m examples.isolation.run
```

The runner deliberately constructs the container command as an argument list. These
are the lines that turn the intended boundary into runtime policy:

```python
command = [
    "docker", "run", "--rm",
    "--network=none",
    "--read-only",
    "--user=65534:65534",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
    "--pids-limit=32",
    "--memory=128m",
    "--cpus=0.5",
    "--tmpfs=/tmp:rw,noexec,nosuid,size=16m,mode=1777",
    "--mount", f"type=bind,src={mounted},dst=/work,readonly",
    image,
    "python", "/work/probe.py",
]
```

The full runner fixes the image, mount source, environment, and program itself. Do
not accept those values from model output and call the result an isolation boundary.

The runner creates a host-only canary file and mounts only a separate fixture
directory. It executes a fixed probe with:

- a non-root UID, all capabilities dropped, and no privilege escalation;
- a read-only root and fixture mount, with a small writable scratch directory;
- no container network, no Docker socket, and no host credentials passed through;
- process, memory, CPU, and wall-clock bounds.

The probe needs both positive and negative controls:

| Attempt | Required observation |
|---|---|
| Read the mounted fixture | Succeeds — the runner actually works |
| Write scratch data | Succeeds — intended work remains possible |
| Read the host-only canary | Fails |
| Change the mounted fixture or root filesystem | Fails |
| Connect to a public network address | Fails |
| Read the host's canary environment variable | Absent |

All checks must pass. A nonzero exit or missing Docker daemon is a failed/unavailable
experiment, never evidence that isolation worked. The network connection failure
alone is weak evidence if the host is offline; interpret it together with Docker's
explicit `--network=none` configuration. Save the engine/platform version and image
digest when recording results; `--image python@sha256:...` pins the image.

A container is a concrete boundary, not a proof against every kernel/runtime escape.
Changing mounts, enabling network, or exposing a privileged socket changes what you
have tested. Re-run the probes after such a change.

## Close the case

The corrected sentence is: “This process may read these mounted files and write
this scratch area; these probes verify that configuration.” That sentence is less
magical than “sandboxed,” and much more useful during a review.

**Artifact:** the JSON probe output, Docker/image versions, and the earlier sibling-file
read that explains why `cwd` was insufficient. In CI, the Linux isolation job runs
this same probe. Locally, `pytest tests/test_sandbox.py -q` also demonstrates the
weaker process wrapper's actual boundary.

The [Docker run reference](https://docs.docker.com/engine/containers/run/) documents the runtime flags used by the probe. Inspect the actual mounts and privileges whenever you change them.

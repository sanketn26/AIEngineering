"""Executed inside the restricted container. Positive AND negative controls."""

import json
import os
import socket
from pathlib import Path


def denied(action):
    try:
        action()
        return False
    except (OSError, PermissionError):
        return True


def main():
    # This path exists on the host in the runner's temporary directory. It is
    # deliberately not mounted. A mere cwd change would leave it reachable.
    host_canary = os.environ["HOST_CANARY_PATH"]
    checks = {
        "fixture_readable": Path("/work/fixture.txt").read_text().strip() == "hello",
        "non_root": os.getuid() != 0,
        "host_canary_hidden": denied(lambda: Path(host_canary).read_text()),
        "fixture_read_only": denied(
            lambda: Path("/work/fixture.txt").write_text("changed")
        ),
        "root_read_only": denied(
            lambda: Path("/etc/probe-write").write_text("changed")
        ),
        "network_unavailable": denied(
            lambda: socket.create_connection(("1.1.1.1", 443), timeout=1)
        ),
        "no_host_secret": "ISOLATION_HOST_CANARY_SECRET" not in os.environ,
    }
    scratch = Path("/tmp/allowed-scratch")
    scratch.write_text("allowed")
    checks["scratch_writable"] = scratch.read_text() == "allowed"
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

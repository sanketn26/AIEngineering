"""Run a fixed probe, never model-supplied commands or mounts."""

import argparse
import os
import subprocess
import tempfile
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="python:3.11-slim",
        help="pin a trusted image digest for recorded runs",
    )
    args = parser.parse_args()
    name = "aieng-isolation-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="aieng-isolation-") as directory:
        root = Path(directory)
        canary = root / "host-only.txt"
        canary.write_text("fictional host secret")
        mounted = root / "mounted"
        mounted.mkdir()
        (mounted / "fixture.txt").write_text("hello")
        (mounted / "probe.py").write_text(
            Path(__file__).with_name("probe.py").read_text()
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--user=65534:65534",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=32",
            "--memory=128m",
            "--cpus=0.5",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=16m,mode=1777",
            "--mount",
            f"type=bind,src={mounted},dst=/work,readonly",
            "--env",
            f"HOST_CANARY_PATH={canary}",
            args.image,
            "python",
            "/work/probe.py",
        ]
        # tempfile parents are 0700 on the host; Docker daemon performs the mount.
        mounted.chmod(0o755)
        try:
            result = subprocess.run(
                command,
                timeout=60,
                env={**os.environ, "ISOLATION_HOST_CANARY_SECRET": "not-forwarded"},
            )
            if result.returncode:
                raise SystemExit(result.returncode)
            assert canary.read_text() == "fictional host secret"
            assert (mounted / "fixture.txt").read_text() == "hello"
        except subprocess.TimeoutExpired:
            # Killing the Docker CLI does not guarantee the container stopped.
            subprocess.run(["docker", "rm", "-f", name], check=False, timeout=10)
            raise SystemExit("probe timed out; named container stopped")


if __name__ == "__main__":
    main()

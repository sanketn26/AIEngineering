"""Start a real local HTTP server, approve twice, restart, and measure a load run.

Everything is temporary, including credentials and the ledger. No external model
or payment is contacted. The server is stopped even if an assertion fails.
"""

import asyncio
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
from .load import run as load
from .service import HERE


@contextmanager
def server(port: int, environment: dict, directory: Path):
    log_path = directory / "server.log"
    with log_path.open("a") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "examples.production_triage.service:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--no-access-log",
            ],
            env=environment,
            stdout=stream,
            stderr=stream,
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(
                        "reference server exited: " + log_path.read_text()[-2000:]
                    )
                try:
                    if (
                        httpx.get(
                            f"http://127.0.0.1:{port}/healthz", timeout=0.2
                        ).status_code
                        == 200
                    ):
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            else:
                raise TimeoutError("reference server did not become healthy")
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    with tempfile.TemporaryDirectory(prefix="triage-rehearsal-") as temporary:
        directory = Path(temporary)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        token = secrets.token_urlsafe(32)
        headers = {"Authorization": "Bearer " + token}
        # Never print the credential or inherited environment.
        environment = {
            **os.environ,
            "TRIAGE_TOKENS_JSON": json.dumps(
                {
                    token: {
                        "id": "rehearsal",
                        "role": "admin",
                        "scopes": ["refund:write"],
                    }
                }
            ),
            "TRIAGE_DB": str(directory / "state.sqlite"),
            "TRIAGE_RELEASE": str(HERE / "data/release-v1.json"),
        }
        environment.pop("TRIAGE_EXPECTED_DIGEST", None)
        base = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=base, headers=headers, timeout=5) as client:
            with server(port, environment, directory):
                ready = client.get("/readyz").json()
                result = client.post(
                    "/v1/triage",
                    json={
                        "ticket_id": "restart-case",
                        "text": "Package arrived but billed twice",
                    },
                ).json()
                assert result["category"] == "billing"
                proposal = result["action"]["proposal_id"]
                url = f"/v1/proposals/{proposal}/decision"
                assert client.post(url, json={"approve": True}).status_code == 200
            # Pin the reviewed bundle on the second boot; same durable state.
            environment["TRIAGE_EXPECTED_DIGEST"] = ready["digest"]
            with server(port, environment, directory):
                assert client.post(url, json={"approve": True}).status_code == 200
                old = os.environ.get("TRIAGE_BEARER")
                os.environ["TRIAGE_BEARER"] = token
                try:
                    measurements = asyncio.run(load(base, 30, 4))
                finally:
                    if old is None:
                        os.environ.pop("TRIAGE_BEARER", None)
                    else:
                        os.environ["TRIAGE_BEARER"] = old
                metrics = client.get("/ops/metrics").json()
                assert measurements["errors"] == 0
                assert metrics["n"] == 31 and metrics["successes"] == 31

            # Rehearsal only: deliberately promote an over-tight rate limit,
            # observe the incident, then restore the reviewed configuration.
            candidate = json.loads((HERE / "data/release-v1.json").read_text())
            candidate.update(version="triage-v2-bad-limit", requests_per_minute=1)
            candidate_path = directory / "candidate.json"
            candidate_path.write_text(json.dumps(candidate))
            environment["TRIAGE_RELEASE"] = str(candidate_path)
            environment.pop("TRIAGE_EXPECTED_DIGEST", None)
            ticket = {"ticket_id": "rollback-check", "text": "Forgot my password"}
            with server(port, environment, directory):
                candidate_status = client.post("/v1/triage", json=ticket).status_code
                assert candidate_status == 429
            environment["TRIAGE_RELEASE"] = str(HERE / "data/release-v1.json")
            environment["TRIAGE_EXPECTED_DIGEST"] = ready["digest"]
            with server(port, environment, directory):
                restored_status = client.post("/v1/triage", json=ticket).status_code
                assert restored_status == 200
                assert client.get("/readyz").json()["digest"] == ready["digest"]
            from .storage import Store

            with Store(directory / "state.sqlite").connect() as db:
                assert db.execute("SELECT count(*) FROM ledger").fetchone()[0] == 1
        print(
            json.dumps(
                {
                    "provider": "mock",
                    "restart_and_duplicate_approval": "passed",
                    "ledger_entries": 1,
                    "config_rollback": {
                        "candidate_status": candidate_status,
                        "restored_status": restored_status,
                    },
                    "load": measurements,
                    "metrics_before_rollback_experiment": metrics,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

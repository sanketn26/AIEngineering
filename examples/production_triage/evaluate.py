"""Offline release gate. Add --live to evaluate a configured real endpoint."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from .contracts import Principal
from .service import HERE, create_app


def evaluate(client, token: str, rows: list[dict]) -> dict:
    failures = []
    for row in rows:
        response = client.post(
            "/v1/triage",
            headers={"Authorization": "Bearer " + token},
            json={"ticket_id": row["id"], "text": row["text"]},
        )
        body = response.json()
        correct = response.status_code == 200 and all(
            body.get(k) == v for k, v in row["expect"].items()
        )
        if not correct:
            failures.append(row["id"])
    return {
        "n": len(rows),
        "passed": len(rows) - len(failures),
        "accuracy": (len(rows) - len(failures)) / len(rows),
        "failures": failures,
        "ok": not failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="use configured HTTP provider; incurs configured provider usage",
    )
    parser.add_argument("--split", choices=["dev", "release"], default="release")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (HERE / f"data/{args.split}.jsonl").read_text().splitlines()
    ]
    with tempfile.TemporaryDirectory() as directory:
        token = "offline-evaluation-token"
        config = (
            Path(os.environ["TRIAGE_RELEASE"])
            if args.live
            else HERE / "data/release-v1.json"
        )
        if args.live and json.loads(config.read_text())["provider"] != "http":
            parser.error("--live requires an HTTP release")
        app = create_app(
            release_path=config,
            db_path=Path(directory) / "eval.sqlite",
            tokens={token: Principal(id="evaluator", role="viewer")},
        )
        with TestClient(app) as client:
            result = evaluate(client, token, rows)
        result.update(
            {
                "split": args.split,
                "provider": app.state.release.provider,
                "digest": app.state.digest,
                "fixture": "synthetic regression cases; not field accuracy",
            }
        )
        print(json.dumps(result, indent=2))
        if not result["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

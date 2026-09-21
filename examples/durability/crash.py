"""Crash after an effect, before recording completion. Retry the SAME operation ID."""

import argparse
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from src.durable import Coordinator, DurableStore


def work(directory: Path, crash: bool):
    def write(ctx):
        # Receiver-enforced idempotency, in the same transaction as the effect.
        # Stable identity is job + phase, never a fresh UUID per retry.
        with sqlite3.connect(directory / "receiver.sqlite") as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS receipts (operation TEXT PRIMARY KEY)"
            )
            db.execute(
                "INSERT OR IGNORE INTO receipts VALUES (?)", ("ticket-7:refund",)
            )
        if crash:
            os._exit(73)  # receipt committed; phase_done never reached the journal
        return {"operation": "ticket-7:refund"}

    coordinator = Coordinator(
        DurableStore(directory / "events.jsonl"), ["refund"], {"refund": write}
    )
    return coordinator.run_until_gate({})


def demonstrate(directory: Path):
    first = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.durability.crash",
            "--worker",
            str(directory),
            "--crash",
        ]
    )
    if first.returncode != 73:
        raise RuntimeError("failure injection did not reach the intended point")
    assert DurableStore(directory / "events.jsonl").last("phase_done") is None
    result = work(directory, False)
    with sqlite3.connect(directory / "receiver.sqlite") as db:
        effects = db.execute("SELECT count(*) FROM receipts").fetchone()[0]
    assert effects == 1
    print(
        {
            "first_process_exit": first.returncode,
            "replayed_worker": True,
            "receiver_effects": effects,
            "recovered_phase": result["phase"],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--crash", action="store_true")
    args = parser.parse_args()
    if args.worker:
        work(args.worker, args.crash)
    else:
        with tempfile.TemporaryDirectory() as path:
            demonstrate(Path(path))

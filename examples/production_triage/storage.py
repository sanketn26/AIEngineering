"""One SQLite transaction owns approval + the simulated side effect.

A real payment API needs its own idempotency key/outbox reconciliation. A local
transaction cannot atomically commit a bank transfer and this database.
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(
                """
              CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY, ticket TEXT NOT NULL UNIQUE,
                owner TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending');
              CREATE TABLE IF NOT EXISTS ledger (
                proposal TEXT PRIMARY KEY, ticket TEXT NOT NULL, actor TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, body TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS limits (
                actor TEXT PRIMARY KEY, start REAL NOT NULL, n INTEGER NOT NULL);
            """
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=2)
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def admit(self, actor: str, limit: int, window: float = 60) -> bool:
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT start, n FROM limits WHERE actor=?", (actor,)
            ).fetchone()
            start, n = row if row and now - row[0] < window else (now, 0)
            if n >= limit:
                return False
            db.execute(
                "INSERT OR REPLACE INTO limits VALUES (?,?,?)", (actor, start, n + 1)
            )
        return True

    def propose(self, ticket: str, owner: str) -> str:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id, owner FROM proposals WHERE ticket=?", (ticket,)
            ).fetchone()
            if row:
                if row[1] != owner:
                    raise PermissionError("ticket belongs to another principal")
                return row[0]
            proposal = str(uuid.uuid4())
            db.execute(
                "INSERT INTO proposals(id,ticket,owner) VALUES (?,?,?)",
                (proposal, ticket, owner),
            )
            return proposal

    def decide(self, proposal: str, actor: str, approve: bool) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT ticket, owner, state FROM proposals WHERE id=?", (proposal,)
            ).fetchone()
            if row is None:
                raise KeyError(proposal)
            ticket, owner, state = row
            if owner != actor:
                raise PermissionError("proposal belongs to another principal")
            desired = "approved" if approve else "denied"
            if state != "pending" and state != desired:
                raise ValueError("decision already final")
            db.execute("UPDATE proposals SET state=? WHERE id=?", (desired, proposal))
            if approve:
                db.execute(
                    "INSERT OR IGNORE INTO ledger VALUES (?,?,?)",
                    (proposal, ticket, actor),
                )
            # No network between these writes. A process crash commits both or neither.
            return {
                "proposal_id": proposal,
                "state": desired,
                "effect": "simulated_refund" if approve else "none",
            }

    def record(self, event: dict):
        with self.connect() as db:
            db.execute("INSERT INTO events(body) VALUES (?)", (json.dumps(event),))

    def events(self) -> list[dict]:
        with self.connect() as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT body FROM events ORDER BY seq")
            ]

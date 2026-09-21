"""Completed five-gate reference; small enough to trace one request by hand."""

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import random
import time
import uuid
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import Field, ValidationError

from src.security import prepare_user_message
from .contracts import Classification, Decision, Principal, StrictModel, Ticket
from .provider import HTTPProvider, MockProvider, PROMPT
from .storage import Store

HERE = Path(__file__).parent
log = logging.getLogger("triage.events")


class Release(StrictModel):
    version: str
    provider: Literal["mock", "http"]
    small_model: str
    large_model: str
    deadline_s: float = Field(gt=0, le=30)
    attempts: int = Field(ge=1, le=2)
    max_output_tokens: int = Field(ge=32, le=512)
    request_budget_usd: float = Field(gt=0, le=10)
    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)
    requests_per_minute: int = Field(ge=1, le=10000)


@contextmanager
def span(request: Request, name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        request.state.trace.append(
            {"name": name, "ms": round((time.perf_counter() - start) * 1000, 3)}
        )


def create_app(
    *,
    release_path: Path | None = None,
    db_path: Path | None = None,
    tokens: dict[str, Principal] | None = None,
    provider=None,
    policies: list[dict] | None = None,
) -> FastAPI:
    release_path = release_path or Path(
        os.getenv("TRIAGE_RELEASE", HERE / "data/release-v1.json")
    )
    release = Release.model_validate_json(release_path.read_text())
    policies = (
        policies
        if policies is not None
        else json.loads((HERE / "data/policies.json").read_text())
    )
    bundle = json.dumps(
        {"release": release.model_dump(), "prompt": PROMPT, "policies": policies},
        sort_keys=True,
    )
    digest = hashlib.sha256(bundle.encode()).hexdigest()
    expected = os.getenv("TRIAGE_EXPECTED_DIGEST")
    if expected and not hmac.compare_digest(expected, digest):
        raise ValueError("release digest mismatch; refusing startup")
    if tokens is None:
        raw = json.loads(os.environ.get("TRIAGE_TOKENS_JSON", "{}"))
        tokens = {
            token: Principal.model_validate(actor) for token, actor in raw.items()
        }
    if not tokens or any(len(token) < 16 for token in tokens):
        raise ValueError(
            "configure TRIAGE_TOKENS_JSON with bearer tokens of at least 16 characters"
        )
    principals = {
        hashlib.sha256(token.encode()).hexdigest(): actor
        for token, actor in tokens.items()
    }
    store = Store(
        db_path or Path(os.getenv("TRIAGE_DB", "/tmp/triage-reference.sqlite"))
    )
    if provider is None:
        if release.provider == "http":
            endpoint = os.environ["TRIAGE_PROVIDER_ENDPOINT"]
            url = httpx.URL(endpoint)
            if url.scheme != "https" and not (
                url.scheme == "http" and url.host in {"localhost", "127.0.0.1", "::1"}
            ):
                raise ValueError("use HTTPS, or loopback HTTP for a local model")
            provider = HTTPProvider(
                endpoint, os.environ.get("TRIAGE_PROVIDER_KEY", ""), release.deadline_s
            )
        else:
            provider = MockProvider()
    app = FastAPI(title="Five gates: reference triage")
    app.state.store, app.state.release, app.state.digest = store, release, digest

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        request.state.trace, request.state.spend = [], 0.0
        request.state.attempts = 0
        request.state.estimated = False
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request.state.request_id
            return response
        finally:
            route = request.scope.get("route")
            # Route templates, never user-controlled URLs, ticket text, or tokens.
            event = {
                "request_id": request.state.request_id,
                "route": getattr(route, "path", "unmatched"),
                "status": status,
                "latency_ms": round((time.perf_counter() - start) * 1000, 3),
                "version": release.version,
                "digest": digest,
                "model": getattr(request.state, "model", None),
                "actor_hash": getattr(request.state, "actor_hash", None),
                "attempts": request.state.attempts,
                "cost_usd": request.state.spend,
                "cost_is_estimate": request.state.estimated,
                "spans": request.state.trace,
            }
            store.record(event)
            log.info(json.dumps(event))

    def principal(request: Request) -> Principal:
        value = request.headers.get("authorization", "")
        scheme, _, token = value.partition(" ")
        hashed = hashlib.sha256(token.encode()).hexdigest()
        actor = next(
            (
                actor
                for key, actor in principals.items()
                if hmac.compare_digest(key, hashed)
            ),
            None,
        )
        if scheme.lower() != "bearer" or actor is None:
            raise HTTPException(
                401, "authentication required", headers={"WWW-Authenticate": "Bearer"}
            )
        request.state.actor_hash = hashlib.sha256(actor.id.encode()).hexdigest()
        if not store.admit(actor.id, release.requests_per_minute):
            raise HTTPException(
                429, "request limit reached", headers={"Retry-After": "60"}
            )
        return actor

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(actor: Principal = Depends(principal)):
        return {"status": "ready", "version": release.version, "digest": digest}

    async def classify(text: str, request: Request) -> Classification:
        simple = "password" in text.lower() and not any(
            word in text.lower()
            for word in ("refund", "billed", "charged", "invoice", "payment")
        )
        model = release.small_model if simple else release.large_model
        request.state.model = model
        # Conservative teaching reservation: one token per UTF-8 byte + 256 for
        # the message wrapper. Supply measured/provider-tokenized bounds in prod.
        input_bound = len((PROMPT + text).encode()) + 256
        reservation = (
            input_bound * release.input_usd_per_million
            + release.max_output_tokens * release.output_usd_per_million
        ) / 1_000_000

        async def attempts():
            for attempt in range(release.attempts):
                if request.state.spend + reservation > release.request_budget_usd:
                    raise HTTPException(429, "request cost budget exhausted")
                request.state.spend += reservation
                request.state.estimated = True
                request.state.attempts += 1
                try:
                    with span(request, "model"):
                        raw, usage = await provider.classify(
                            text, model, release.max_output_tokens
                        )
                    with span(request, "validate"):
                        result = Classification.model_validate(raw)
                        inp, out = usage["prompt_tokens"], usage["completion_tokens"]
                        if (
                            type(inp) is not int
                            or type(out) is not int
                            or not 0 <= inp <= input_bound
                            or not 0 <= out <= release.max_output_tokens
                        ):
                            raise ValueError("invalid usage or reservation exceeded")
                    actual = (
                        inp * release.input_usd_per_million
                        + out * release.output_usd_per_million
                    ) / 1_000_000
                    request.state.spend += actual - reservation
                    # Failed attempts retain their reserved cost estimate.
                    request.state.estimated = attempt > 0
                    return result
                except (
                    httpx.TimeoutException,
                    httpx.TransportError,
                    httpx.HTTPStatusError,
                ) as exc:
                    retryable = not isinstance(
                        exc, httpx.HTTPStatusError
                    ) or exc.response.status_code in {429, 502, 503, 504}
                    if not retryable or attempt + 1 == release.attempts:
                        raise HTTPException(
                            503, "provider unavailable; retry later"
                        ) from exc
                    await asyncio.sleep(random.uniform(0.02, 0.05) * (2**attempt))
                except (
                    ValidationError,
                    ValueError,
                    KeyError,
                    IndexError,
                    TypeError,
                ) as exc:
                    raise HTTPException(
                        502, "provider output failed validation"
                    ) from exc

        try:
            # One deadline includes retries, backoff, and body reading.
            return await asyncio.wait_for(attempts(), timeout=release.deadline_s)
        except TimeoutError as exc:
            raise HTTPException(504, "model deadline exceeded; retry later") from exc

    @app.post("/v1/triage")
    async def triage(
        ticket: Ticket, request: Request, actor: Principal = Depends(principal)
    ):
        with span(request, "prepare"):
            text, flags, _ = prepare_user_message(ticket.text)
        result = await classify(text, request)
        with span(request, "retrieve"):
            hits = [
                p
                for p in policies
                if p["category"] == result.category
                and date.fromisoformat(p["expires"]) >= date.today()
            ]
        with span(request, "authorize"):
            proposal = None
            status = "none"
            if result.category == "billing":
                status = "denied"
                if actor.can_refund() and hits:
                    try:
                        proposal = store.propose(ticket.ticket_id, actor.id)
                        status = "pending"
                    except PermissionError as exc:
                        raise HTTPException(403, "ticket not accessible") from exc
        # Evidence is copied from the KB. No model-written policy promises.
        return {
            "category": result.category,
            "priority": result.priority,
            "evidence": [{"id": p["id"], "text": p["text"]} for p in hits],
            "citations": [p["id"] for p in hits],
            "knowledge_status": "found" if hits else "not_in_policy",
            "action": {
                "tool": "refund_customer" if result.category == "billing" else None,
                "status": status,
                "proposal_id": proposal,
            },
            "input_flagged": flags.flagged,
            "model_id": request.state.model,
            "request_id": request.state.request_id,
            "version": release.version,
        }

    @app.post("/v1/proposals/{proposal}/decision")
    def decide(
        proposal: str,
        decision: Decision,
        request: Request,
        actor: Principal = Depends(principal),
    ):
        if not actor.can_refund():
            raise HTTPException(403, "refund scope required")
        with span(request, "approval_and_ledger"):
            try:
                return store.decide(proposal, actor.id, decision.approve)
            except KeyError as exc:
                raise HTTPException(404, "proposal not found") from exc
            except PermissionError as exc:
                raise HTTPException(403, "proposal not accessible") from exc
            except ValueError as exc:
                raise HTTPException(409, "decision already final") from exc

    @app.get("/ops/metrics")
    def metrics(actor: Principal = Depends(principal)):
        if actor.role != "admin":
            raise HTTPException(403, "admin required")
        rows = [e for e in store.events() if e["route"] == "/v1/triage"]
        latency = sorted(e["latency_ms"] for e in rows)

        def percentile(p):
            return latency[max(0, math.ceil(p * len(latency)) - 1)] if latency else None

        return {
            "n": len(rows),
            "successes": sum(e["status"] == 200 for e in rows),
            "errors": sum(e["status"] >= 400 for e in rows),
            "p50_ms": percentile(0.5),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
            "cost_usd": sum(e["cost_usd"] for e in rows),
            "estimated_cost_requests": sum(e["cost_is_estimate"] for e in rows),
            "model_calls": sum(e["attempts"] for e in rows),
            "digest": digest,
        }

    return app

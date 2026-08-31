"""Support-ticket triage service — runnable, incomplete.

    uvicorn app:app --reload
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI

from model import classify, retrieve
from schemas import ProposedAction, TriageRequest, TriageResponse
from tools import propose_actions

app = FastAPI(
    title="Capstone starter — ticket triage",
    version="0.1.0",
    description=(
        "Domain-agnostic production-AI skeleton. Mock model only. "
        "See README.md and PROGRESS.md for the five gates."
    ),
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/triage", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    request_id = str(uuid.uuid4())
    result = classify(req.text)
    # Gate 3: retrieval is wired but empty — citations stay [].
    _hits = retrieve(req.text)
    proposed = propose_actions(
        result, ticket_id=req.ticket_id, actor=req.actor
    )
    return TriageResponse(
        category=result["category"],  # type: ignore[arg-type]
        priority=result["priority"],  # type: ignore[arg-type]
        rationale=result["rationale"],
        citations=[],
        proposed_actions=[ProposedAction.model_validate(a) for a in proposed],
        model_id=result["model_id"],
        request_id=request_id,
    )

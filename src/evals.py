"""Module 04 — golden-set evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    rows: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def exact_fields(
    pred: dict[str, Any], expect: dict[str, Any], fields: Iterable[str]
) -> bool:
    return all(pred.get(f) == expect.get(f) for f in fields)


def run_suite(
    rows: list[dict[str, Any]],
    predict_fn: Callable[[str], dict[str, Any]],
    fields: list[str],
    input_key: str = "input",
    expect_key: str = "expect",
) -> dict[str, Any]:
    """Run predict_fn over golden rows; return accuracy and failures."""
    if not rows:
        return {"accuracy": 0.0, "n": 0, "failures": []}
    ok = 0
    failures: list[dict[str, Any]] = []
    for row in rows:
        pred = predict_fn(row[input_key])
        expect = row[expect_key]
        if exact_fields(pred, expect, fields):
            ok += 1
        else:
            failures.append(
                {
                    "id": row.get("id"),
                    "input": row[input_key],
                    "expect": expect,
                    "pred": pred,
                }
            )
    n = len(rows)
    return {"accuracy": ok / n, "n": n, "passed": ok, "failures": failures}


def parse_success_rate(raw_outputs: list[str], parser: Callable[[str], Any]) -> float:
    if not raw_outputs:
        return 0.0
    ok = 0
    for raw in raw_outputs:
        try:
            parser(raw)
            ok += 1
        except Exception:  # noqa: BLE001
            pass
    return ok / len(raw_outputs)

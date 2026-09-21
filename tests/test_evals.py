import json
import pytest
from pathlib import Path

from src.evals import (
    exact_fields,
    load_jsonl,
    parse_success_rate,
    run_suite,
    paired_release_gate,
)


def test_exact_fields():
    assert exact_fields({"a": 1, "b": 2}, {"a": 1, "b": 2}, ["a", "b"])
    assert not exact_fields({"a": 1}, {"a": 2}, ["a"])


def test_run_suite(tmp_path: Path):
    path = tmp_path / "gold.jsonl"
    rows = [
        {"id": "1", "input": "Acme 10", "expect": {"company": "Acme", "amount": 10}},
        {"id": "2", "input": "Globex 5", "expect": {"company": "Globex", "amount": 5}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    loaded = load_jsonl(path)

    def predict(text: str) -> dict:
        if "Acme" in text:
            return {"company": "Acme", "amount": 10}
        return {"company": "wrong", "amount": 0}

    result = run_suite(loaded, predict, fields=["company", "amount"])
    assert result["n"] == 2
    assert result["passed"] == 1
    assert abs(result["accuracy"] - 0.5) < 1e-9
    assert len(result["failures"]) == 1


def test_parse_success_rate():
    def parser(s: str) -> dict:
        return json.loads(s)

    rate = parse_success_rate(['{"a":1}', "nope", '{"b":2}'], parser)
    assert abs(rate - 2 / 3) < 1e-9


def test_release_gate_distinguishes_regression_and_uncertainty():
    a = [[1.0, 1.0]] * 30 + [[0.0, 0.0]] * 10
    b = [[0.0, 0.0]] * 30 + [[1.0, 1.0]] * 10
    assert paired_release_gate(a, b)["verdict"] == "regression"
    assert paired_release_gate(b, a)["verdict"] == "pass"
    assert paired_release_gate(a[:4], b[:4])["verdict"] == "inconclusive"


def test_repeats_do_not_become_independent_cases():
    result = paired_release_gate([[1.0] * 100, [0.0] * 100], [[1.0] * 100, [1.0] * 100])
    assert result["n_cases"] == 2
    assert not result["ok"]
    assert (
        paired_release_gate([[1.0, 1.0]] * 30, [[1.0, 1.0]] * 30)["verdict"]
        == "inconclusive"
    )


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.1])
def test_release_gate_rejects_invalid_scores(bad):
    with pytest.raises(ValueError):
        paired_release_gate([[1.0]], [[bad]])

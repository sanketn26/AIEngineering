"""Paired benchmark: scoring vs generation, same cases, same loaded model.

    pip install -r requirements-decision.txt
    python -m decision.bench --model Qwen/Qwen2.5-0.5B-Instruct --repeat 4

Scoring runs one forward pass and reads the label logits. Generation runs the
same prompt through the decode loop for up to ``--max-new-tokens`` tokens and
parses the first category it names. Lanes run one after the other in one
process, so the difference isolates the decode loop; there is no server,
batching, or concurrency here. Mock timings would measure Python, not
inference, so there is no mock mode. Quote the numbers you measure.
"""

from __future__ import annotations

import argparse
import statistics
import time

from decision.calibration import load_rows
from decision.policy import decide
from decision.scoring import TRIAGE_CHOICES, MLXScorer, TransformersScorer, build_prompt, render_chat


def parse_generated(text: str, ticket: str = "") -> str | None:
    """First allowed choice named in the reply, after removing any echoed ticket.

    Small models often restate the ticket before answering, and the ticket can
    itself contain a category word. Do not read that echo as the answer.
    """
    body = text.lower()
    if ticket:
        body = body.replace(ticket.lower(), " ")
    hits = [(body.find(c.value), c.value) for c in TRIAGE_CHOICES if c.value in body]
    return min(hits)[1] if hits else None


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(rows: list[tuple[float, bool, int]]) -> dict:
    ms = [r[0] * 1000 for r in rows]
    return {
        "n": len(rows),
        "accuracy": sum(r[1] for r in rows) / len(rows),
        "mean_ms": statistics.fmean(ms),
        "p50_ms": percentile(ms, 0.50),
        "p95_ms": percentile(ms, 0.95),
        "p99_ms": percentile(ms, 0.99),
        "generated_tokens": sum(r[2] for r in rows),
    }


def run(
    scorer: TransformersScorer | MLXScorer, cases: list[dict], max_new_tokens: int, rotations: int = 1
) -> dict[str, dict]:
    score_rows, gen_rows = [], []
    for case in cases:
        t0 = time.perf_counter()
        d = decide(case["input"], scorer, rotations=rotations)
        score_rows.append((time.perf_counter() - t0, d.choice == case["expect"], 0))
    for case in cases:
        prompt, _ = build_prompt(case["input"], TRIAGE_CHOICES)
        prompt = prompt.replace("Return only the label.\nLabel:\n", "Name the category and explain briefly.")
        prompt = render_chat(scorer.tokenizer, prompt)  # same template as the scoring lane
        t0 = time.perf_counter()
        text, tokens = scorer.generate(prompt, max_new_tokens=max_new_tokens)
        gen_rows.append((time.perf_counter() - t0, parse_generated(text, case["input"]) == case["expect"], tokens))
    return {"score": summarize(score_rows), "generate": summarize(gen_rows)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("transformers", "mlx"), default="transformers")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--repeat", type=int, default=4, help="passes over the dataset")
    p.add_argument("--rotations", type=int, default=1, help="scoring passes per case (see decision.policy.decide)")
    p.add_argument("--max-new-tokens", type=int, default=32)
    args = p.parse_args(argv)

    scorer = (MLXScorer if args.backend == "mlx" else TransformersScorer)(args.model)
    rows = load_rows()
    decide(rows[0]["input"], scorer)  # warm up and fail fast on bad label tokens
    cases = rows * args.repeat
    summary = run(scorer, cases, args.max_new_tokens, args.rotations)

    print(f"model={args.model} device={scorer.device} cases={len(cases)} rotations={args.rotations} max_new_tokens={args.max_new_tokens}")
    print(f"{'lane':<9}{'acc':>6}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'gen tok':>9}")
    for lane, s in summary.items():
        print(
            f"{lane:<9}{s['accuracy']:>6.2f}{s['mean_ms']:>7.0f}ms{s['p50_ms']:>7.0f}ms"
            f"{s['p95_ms']:>7.0f}ms{s['p99_ms']:>7.0f}ms{s['generated_tokens']:>9}"
        )
    ratio = summary["generate"]["mean_ms"] / summary["score"]["mean_ms"]
    print(f"\nscoring mean latency is {ratio:.2f}x lower than generation on this run.")


if __name__ == "__main__":
    main()

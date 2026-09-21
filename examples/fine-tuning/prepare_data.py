"""Validate chat rows and write a company-level split plus its manifest.

    python examples/fine-tuning/prepare_data.py \
        --input examples/fine-tuning/data/sample.jsonl --out data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.finetune_data import split_by_company, validate_example


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--split-version", default="v1")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    failures = {i: p for i, row in enumerate(rows) if (p := validate_example(row))}
    if failures:
        for i, problems in failures.items():
            print(f"row {i}: {'; '.join(problems)}", file=sys.stderr)
        print("Fix or send to review. Nothing was written.", file=sys.stderr)
        return 1

    splits, manifest = split_by_company(rows, seed=args.split_version)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps({"messages": row["messages"]}) + "\n")
    (args.out / "split-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print({name: len(r) for name, r in splits.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

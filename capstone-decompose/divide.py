"""CLI for the divide-solve-join capstone. Run from this directory."""

from __future__ import annotations

import json
import sys

from runtime.execute import MOCK_PLAN, join_parts, run_task
from runtime.validate import accept_plan, load_yaml, validate_document


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 2 and args[0] == "check":
        result = validate_document(load_yaml(args[1]))
        print(result["status"])
        if result["errors"]:
            print(json.dumps(result["errors"], indent=2))
            return 1
        spec = result["spec"]
        print(f"{spec.task_id}: {len(spec.requirements)} requirements, {len(spec.verification)} checks")
        return 0
    if len(args) >= 2 and args[0] == "run":
        result = validate_document(load_yaml(args[1]))
        if result["status"] != "SPEC_READY":
            print(result["status"])
            return 1
        outcome = run_task(result["spec"])
        print(outcome["mode"])
        print(outcome["status"])
        return 0
    if len(args) >= 2 and args[0] == "plan":
        result = validate_document(load_yaml(args[1]))
        if result["status"] != "SPEC_READY":
            print(result["status"])
            return 1
        verdict = accept_plan(result["spec"], MOCK_PLAN)
        print(verdict["status"])
        joined = join_parts(result["spec"], MOCK_PLAN)
        print(f"join={joined['status']} checks={joined['join_checks_ran']}")
        return 0
    print("usage: python divide.py check <task.yaml> | run <task.yaml> | plan <task.yaml>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

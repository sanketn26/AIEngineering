"""CLI for the command-runtime capstone. Run from this directory."""

from __future__ import annotations

import json
import sys

from runtime.execute import execute
from runtime.validate import load_yaml, render_contract, validate_document


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 2 and args[0] == "spec" and args[1] == "check":
        document = load_yaml(args[2])
        result = validate_document(document)
        if result["status"] != "SPEC_READY":
            print(result["status"])
            print(json.dumps(result["errors"], indent=2))
            return 1
        print(result["status"])
        print(render_contract(result["spec"]))
        return 0
    if len(args) >= 2 and args[0] == "run" and args[1] == "--intent":
        outcome = execute(" ".join(args[2:]))
        print(outcome["status"])
        return 0
    if len(args) >= 2 and args[0] == "run":
        outcome = execute(load_yaml(args[1]))
        print(outcome["status"])
        print(f"policy={outcome.get('policy')} revision={outcome.get('revision')}")
        return 0
    print("usage: python cmdai.py spec check <file.yaml> | python cmdai.py run <file.yaml> | python cmdai.py run --intent \"...\"")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Patch policy and verification planning. Both are open on purpose."""

from __future__ import annotations

from runtime.models import AcceptanceCriterion, TaskSpec

# GATE 4 — planted: the mock proposes a file outside the allowlist and a new dependency.
MOCK_PATCH = """--- a/client.py
+++ b/client.py
@@ -1,1 +1,3 @@
+import time
+
 def fetch_data():
--- a/setup.py
+++ b/setup.py
@@ -0,0 +1,2 @@
+install_requires = ["tenacity"]
"""


def enforce_patch_policy(patch: str, spec: TaskSpec) -> dict[str, object]:
    del patch, spec
    # GATE 4 — planted: every candidate is PASS, including forbidden paths.
    return {"status": "PASS", "violations": []}


def verification_plan(criterion: AcceptanceCriterion) -> dict[str, object]:
    # GATE 4 — planted: the criterion's text is a shell command.
    return {"shell": True, "command": criterion.verification}


def freeze(spec: TaskSpec) -> dict[str, object]:
    # GATE 5 — planted: nothing immutable is stored, so a later edit keeps the old approval.
    return {"task_id": spec.task_id, "revision": None, "spec": spec.model_dump(mode="json")}


# GATE 4 — planted. The 4B repair budget in the spec is 2.
MAX_REPAIRS = 99


def repeated_candidate(seen: set[str], patch_hash: str) -> bool:
    del seen, patch_hash
    # GATE 4 — planted: the same diff can be submitted on every retry.
    return False

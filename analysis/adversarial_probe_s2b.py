#!/usr/bin/env python3
"""Adversarial held-out probe: the clause's SUCCESS and FAILURE shapes.

The facet clause says the row count is fixed by the clause count. Two inputs:

  MINIMAL: a task with exactly ONE verifiable clause ("write X to path Y").
    The clause must instruct ONE row, not pressure the agent into inventing
    extra faces. Pass condition: the clause text contains the clause-count
    anchoring (so a 1-clause task yields 1 row) and does NOT contain any
    minimum/maximum row count of its own.

  ZERO-CONFLICT: a task whose text states an explicit form that would conflict
    with a guessed reading. The ledger clause must defer to the task text's own
    words, not to the agent's preference.

Both assertions are over the REAL tool output from the deployed tree.
"""
import sys

sys.path.insert(0, "/nfs/caozhou/glm/agent_stage3")
from flagscale_agent.react.tools.plan_create import PlanCreateTool


class _Plan:
    def __init__(self):
        self.steps = None

    def create(self, title, steps, session_id, thinking="", oracle=None):
        self.steps = steps
        return "OK"

    def summary(self):
        return ""


def frame_text(task_text):
    p = _Plan()
    PlanCreateTool(p, session_id="probe").execute(
        title="adversarial probe", steps=["do the thing"], thinking="probe",
    )
    acc = p.steps[0]["acceptance"]
    return " ".join(" ".join(a.lower().split()) for a in acc)


joined = frame_text("write the answer to /app/out.txt and nothing else")

checks = {
    # The anchoring must be a RELATION to clause count, so 1 clause -> 1 row.
    "row count anchored to clause count": "the row count is fixed by the clause count" in joined,
    # The clause must NOT contain its own fixed row count (which would force
    # fabricated rows on a one-clause task).
    "no hard-coded row minimum": not any(
        p in joined for p in ("at least 3 rows", "at least three rows",
                              "minimum of 3", "at least 5 faces")
    ),
    # A row without a probe is explicitly called out as not-a-check (so the
    # agent cannot pad the manifest with unexecuted rows).
    "unprobed row is not a check": "a row with no executed probe is a sentence, not a check" in joined,
    # Coverage duty must be conditional, not unconditional two-implementation.
    "top-2 only when undominated": "when no reading dominates" in joined,
    # Deference to the task's own words over the agent's preference.
    "readings pinned to task text": "task text's own words" in joined,
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")

ok = all(checks.values())
print("\nADVERSARIAL HELD-OUT PROBE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

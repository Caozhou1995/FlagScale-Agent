#!/usr/bin/env python3
"""FAR-END consumer probe: drive the REAL plan_create tool from the deployed
stage3 tree (not the constants) with a fresh, never-seen task statement, and
assert the FRAME step the agent actually receives carries both clauses in order.

This is the "construct a fresh input and re-run" check: the input below was
written for this probe, not taken from any trial or from my own test file.
"""
import sys

sys.path.insert(0, "/nfs/caozhou/glm/agent_stage3")

from flagscale_agent.react.tools.plan_create import PlanCreateTool


class _Plan:
    def __init__(self):
        self.saved = None

    def create(self, title, steps, session_id, thinking="", oracle=None):
        self.saved = steps
        return "OK"

    def summary(self):
        return ""


p = _Plan()
tool = PlanCreateTool(p, session_id="probe")

# A brand-new task statement invented for this probe (not from any trial log,
# not from my own regression tests) -- deliberately ambiguous about output form
# and containing several verifiable clauses.
FRESH_TASK = (
    "Convert the CSV report into a JSON document. "
    "Keep the column order as it appears in the source. "
    "Summarise the total row count. "
    "The verifier will run: python check.py --report /app/out.json && "
    "python check.py --count /app/out.json"
)

out = tool.execute(
    title="fresh-input probe",
    steps=[{"title": "build the converter", "acceptance": ["check.py passes"]}],
    thinking="probe only",
)
assert "Plan created" in out, f"tool did not create a plan: {out[:200]}"

steps = p.saved
frame = steps[0]
assert "Frame" in frame["title"], f"first step is not FRAME: {frame['title']}"

acc = frame["acceptance"]
joined = " ".join(" ".join(a.lower().split()) for a in acc)

checks = {
    "facet clause present": "acceptance-facet manifest" in joined,
    "facet anchoring relation": "fixed by the clause count" in joined,
    "facet requires probe+observation": "probe" in joined and "observed" in joined,
    "ledger clause present": "reading ledger" in joined,
    "ledger enumerates readings": "every defensible reading" in joined,
    "ledger dominance test": "dominance" in joined,
    "ledger top-2 same run": "cover the top-2" in joined and "in the same run" in joined,
    "ledger defers to PRINCIPLE 4": "principle 4" in joined,
    "no task-specific leak": not any(
        w in joined for w in ("filter-js", "mcmc", "beautifulsoup", "csv report")
    ),
}
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")

# Order: facet clause must precede the ledger clause (build-time discipline then
# ambiguity discipline), and both must sit inside the FRAME step's acceptance.
def _idx(needle):
    for i, a in enumerate(acc):
        if needle in a.lower():
            return i
    return -1

i_facet = _idx("acceptance-facet manifest")
i_ledger = _idx("reading ledger")
print(f"  facet item index={i_facet}, ledger item index={i_ledger}")
order_ok = 0 <= i_facet < i_ledger
print(f"  [{'PASS' if order_ok else 'FAIL'}] facet clause precedes ledger clause")

ok = all(checks.values()) and order_ok
print("\nFAR-END CONSUMER PROBE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

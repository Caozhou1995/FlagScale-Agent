"""Fresh-input probe: does the DEPLOYED stage3 plan_create emit the new
clause doctrine for a task the environment never handed me?

The point is generalization, not the dev sample: a brand-new task statement with
a different number of verifiable clauses must still produce a FRAME step whose
acceptance names the acceptance-facet manifest and the reading ledger, and must
not leak any benchmark-specific word.
"""

import sys

sys.path.insert(0, "/nfs/caozhou/glm/agent_stage3")

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
from flagscale_agent.react.tools.plan_create import _FRAME_STEP_ACCEPTANCE as FRAME

# A task statement the environment never gave me: 4 verifiable clauses, an open
# output form, and an artifact name. Nothing benchmark-specific.
FRESH_TASK = (
    "Write a command-line inventory report generator. "
    "It must read TSV rows with columns id, name, qty from stdin. "
    "It must emit JSON to stdout with a top-level 'items' array, preserving input order. "
    "Rows whose qty is not a positive integer must be skipped, and their count reported "
    "on stderr as 'skipped: N'. "
    "The report must be written to report.json as well, byte-identical to stdout."
)

CLAUSES = [
    "read TSV rows with columns id, name, qty from stdin",
    "emit JSON to stdout with a top-level 'items' array, preserving input order",
    "skip rows whose qty is not a positive integer and report the count on stderr",
    "write report.json byte-identical to stdout",
]

failures = []


def check(name, ok, observed):
    print(f"{'PASS' if ok else 'FAIL'}  {name} :: {observed}")
    if not ok:
        failures.append(name)


def has(needle, hay):
    return needle in hay


flat_frame = " ".join(" ".join(FRAME).split()).lower()
flat_static = " ".join(SYSTEM_PROMPT_STATIC.split()).lower()

check("frame names the acceptance-facet manifest",
      "acceptance-facet manifest" in flat_frame, "substring present")
check("frame fixes the row count by the clause count",
      "row count is fixed by the clause count" in flat_frame, "substring present")
check("frame forbids merging rows",
      "never fold several clauses into one row" in flat_frame, "substring present")
check("frame requires an observed result per row",
      "observed result when you ran it" in flat_frame, "substring present")
check("frame requires close-out to review the manifest row by row",
      "close-out reviews this manifest row by row" in flat_frame, "substring present")
check("frame enumerates defensible readings",
      "every defensible reading of that requirement" in flat_frame,
      f"present={'every defensible reading of that requirement' in flat_frame}")
check("frame runs a dominance test on the consumer's material",
      "dominance test against the consumer's own material" in flat_frame, "substring present")
check("frame covers top-2 in the same run",
      "cover the top-2 (or top-k) in the same run" in flat_frame, "substring present")
check("frame conditions coverage on direction of loss",
      "which direction the comparison penalizes" in flat_frame, "substring present")
check("static copy carries direction-of-loss too (no drift)",
      "first determine which direction the comparison penalizes" in flat_static,
      "substring present")

# Generalization: the clause count for THIS fresh task is 4, not any number the
# clause text hardcodes. The clause must not embed a fixed row count.
import re
hardcoded = re.findall(r"\b(?:exactly|always)\s+(\d+)\s+rows?\b", flat_frame)
check("no hardcoded row count in the clause", not hardcoded, f"found={hardcoded}")

LEAK = ["mcmc", "filter-js", "beautifulsoup", "bs4", "html.parser", "sam-cell",
        "mailman", "git-multibranch", "query-optimize", "dna-assembly",
        "video-processing", "extract-moves", "gcode", "protein-assembly"]
leaks = [w for w in LEAK if w in flat_static or w in flat_frame]
check("no benchmark-specific leak in the doctrine", not leaks, f"hits={leaks}")

print()
print(f"FRESH-INPUT PROBE: {'PASS' if not failures else 'FAIL ' + str(failures)}")
sys.exit(0 if not failures else 1)

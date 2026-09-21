#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0
"""M3 end-to-end driver: real subprocess spawn + parent poll_tasks reunite.

Reuses the M2 spawn (real `flagscale-agent` worker subprocess) but drives the
parent's verdict through the M3 code path — `check_result` / `PollTasksTool` —
instead of an inline acceptance re-run. The whole point of M3 is that the
parent judges by independently VERIFYING the deliverable (running the acceptance
predicate itself) and NEVER by reading result.json (INV3); it never redoes the
task.

Variants:
  normal  — worker writes out.md (>=3 lines) + report_result  → check → DONE
  forged  — worker reports success but the acceptance spec is unsatisfiable
            (extra `exit 1`) → check → REJECTED, even with a rosy result.json.

Usage: python3 scripts/e2e_m3.py <normal|forged>
"""
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from flagscale_agent.react.multi_agent.ledger import (
    DEADLINE_MISSED, DONE, FAILED, REPORTED, REJECTED, TaskLedger,
)
from flagscale_agent.react.multi_agent.reunite import PollTasksTool, check_result
from flagscale_agent.react.multi_agent.spawn import SpawnWorkerTool


def main(variant: str):
    tmp = Path(tempfile.mkdtemp(prefix="e2e_m3_"))
    work = tmp / "work"
    work.mkdir()
    tasks_dir = str(tmp / "tasks")
    ledger = TaskLedger(tasks_dir)
    out_ptr = str(work / "out.md")

    variants = {
        "normal": {
            "goal": "write a markdown file with 3 lines of content to output_ptr",
            "acceptance": [{"kind": "check_command",
                            "check": f"test -f {out_ptr} && "
                                     f"test $(wc -l < {out_ptr}) -ge 3"}],
            "deadline_minutes": 4,
        },
        "forged": {
            # The worker will (truthfully) do its job and report, but the parent's
            # acceptance carries a forced `exit 1` — the check is unsatisfiable no
            # matter how rosy the self-report. Exercises the INV3 safety valve:
            # result.json says "done", the parent still REJECTS.
            "goal": (f"write 'hello world' to {out_ptr}, then call report_result "
                     f"to report completion; state DONE and success in the summary."),
            "acceptance": [{"kind": "check_command",
                            "check": f"test -f {out_ptr} && exit 1"}],
            "deadline_minutes": 4,
        },
    }
    v = variants[variant]
    tool = SpawnWorkerTool(ledger=ledger)
    print(f"[e2e-m3] variant={variant} tasks_dir={tasks_dir} out_ptr={out_ptr}")
    res = tool.execute(
        goal=v["goal"],
        constraints={"writable": [str(work)]},
        acceptance=v["acceptance"],
        output_ptr=out_ptr,
        deadline_minutes=v["deadline_minutes"],
    )
    print(f"[e2e-m3] spawn result: {res}")
    if not res.startswith("spawned"):
        print("[e2e-m3] SPAWN FAILED")
        return 2
    tid = res.split()[2]

    # Wait until the worker reports or dies.
    wait_until = time.time() + v["deadline_minutes"] * 60 + 90
    while time.time() < wait_until:
        rec = ledger.get(tid)
        if rec and rec.status in (REPORTED, DEADLINE_MISSED, FAILED):
            break
        time.sleep(2)
    rec = ledger.get(tid)
    print(f"[e2e-m3] worker settled: status={rec.status if rec else '?'}")

    # ── M3: parent poll_tasks check → the ONLY verdict path ──────────────────
    poll = PollTasksTool(ledger=ledger)
    out = poll.execute(action="check", task_id=tid)
    print("[e2e-m3] poll_tasks check:")
    for line in out.splitlines():
        print(f"[e2e-m3]   {line}")
    # also exercise the raw API for the verdict object
    vres = check_result(ledger, tid)
    print(f"[e2e-m3] verdict passed={vres.passed} status={vres.status} "
          f"self_report={'yes' if vres.self_report else 'no'}")

    # ── reference-only read of result.json ───────────────────────────────────
    ref = poll.execute(action="result", task_id=tid)
    print("[e2e-m3] poll_tasks result (reference only):")
    for line in ref.splitlines():
        print(f"[e2e-m3]   {line}")

    rec = ledger.get(tid)
    print(f"[e2e-m3] FINAL status={rec.status}")
    for h in rec.history:
        print(f"[e2e-m3]   {h['status']:16s} {h.get('note','')}")
    expected = DONE if variant == "normal" else REJECTED
    ok = rec.status == expected
    print(f"[e2e-m3] EXPECTED={expected} GOT={rec.status} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "normal"))

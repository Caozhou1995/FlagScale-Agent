#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0
"""M2 end-to-end driver: real subprocess spawn + parent-side acceptance.

Runs one of three variants against REAL worker processes:
  normal   — worker writes output_ptr + report_result  → parent runs the
             acceptance predicate → DONE
  missing  — worker reports but never writes the file   → REJECTED
  deadline — worker sleeps past its deadline            → DEADLINE_MISSED

The parent NEVER trusts result.json (INV3) — it independently VERIFIES the
deliverable by running the acceptance predicate itself, then transitions
REPORTED → DONE/REJECTED. It never redoes the task (the labor stays with the
worker); "verification" is the parent's job, not "re-execution".

Usage: python3 scripts/e2e_m2.py <normal|missing|deadline>
"""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from flagscale_agent.react.multi_agent.contract import Contract
from flagscale_agent.react.multi_agent.ledger import (
    DEADLINE_MISSED, DONE, FAILED, REPORTED, REJECTED, RUNNING, TaskLedger,
)
from flagscale_agent.react.multi_agent.spawn import SpawnWorkerTool


def run_acceptance(contract: Contract) -> tuple[bool, list]:
    """Re-run every acceptance check in the PARENT's cwd. Returns (all_pass, evidence)."""
    evidence = []
    all_pass = True
    for a in contract.acceptance:
        cmd = a.get("check", "")
        cwd = a.get("cwd") or os.getcwd()
        try:
            p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                               text=True, timeout=120)
            ok = p.returncode == 0
            evidence.append({"check": cmd, "exit_code": p.returncode,
                             "stdout_tail": (p.stdout or "")[-500:]})
        except subprocess.TimeoutExpired:
            ok = False
            evidence.append({"check": cmd, "exit_code": None, "stdout_tail": "TIMEOUT"})
        all_pass = all_pass and ok
    return all_pass, evidence


def main(variant: str):
    tmp = Path(tempfile.mkdtemp(prefix="e2e_m2_"))
    work = tmp / "work"
    work.mkdir()
    tasks_dir = str(tmp / "tasks")
    ledger = TaskLedger(tasks_dir)

    variants = {
        "normal": {
            "goal": "write a markdown file with 3 lines of content to output_ptr",
            "acceptance": [{"kind": "check_command",
                            "check": f"test -f {work/'out.md'} && "
                                     f"test $(wc -l < {work/'out.md'}) -ge 3"}],
            "deadline_minutes": 4,
        },
        "missing": {
            # Worker does its job and reports, but the parent's acceptance is
            # deterministically unsatisfiable by the worker — this exercises
            # the parent's INDEPENDENT verification (INV3): even a truthful
            # self-report cannot fake a pass.
            "goal": (f"write 'hello world' to {work/'out.md'}, then call "
                     f"report_result to report completion."),
            "acceptance": [{"kind": "check_command",
                            # out.md will exist, but the extra `exit 1` makes the
                            # check fail regardless — parent must REJECT.
                            "check": f"test -f {work/'out.md'} && exit 1"}],
            "deadline_minutes": 4,
        },
        "deadline": {
            "goal": (f"first sleep 600 seconds, then write the file {work/'out.md'}. "
                     f"do not finish early."),
            "acceptance": [{"kind": "check_command",
                            "check": f"test -f {work/'out.md'}"}],
            "deadline_minutes": 1,   # 60s → watchdog must kill it
        },
    }
    v = variants[variant]
    out_ptr = str(work / "out.md")
    tool = SpawnWorkerTool(ledger=ledger)
    print(f"[e2e] variant={variant} tasks_dir={tasks_dir} out_ptr={out_ptr}")
    res = tool.execute(
        goal=v["goal"],
        constraints={"writable": [str(work)]},
        acceptance=v["acceptance"],
        output_ptr=out_ptr,
        deadline_minutes=v["deadline_minutes"],
    )
    print(f"[e2e] spawn result: {res}")
    if not res.startswith("spawned"):
        print("[e2e] SPAWN FAILED")
        return 2
    tid = res.split()[2]

    # ── wait for a terminal-or-REPORTED state ────────────────────────────────
    deadline_wait = time.time() + v["deadline_minutes"] * 60 + 90
    status = None
    while time.time() < deadline_wait:
        rec = ledger.get(tid)
        status = rec.status if rec else "?"
        if status in (REPORTED, DEADLINE_MISSED, FAILED):
            break
        time.sleep(2)
    rec = ledger.get(tid)
    status = rec.status if rec else "?"
    print(f"[e2e] worker settled: status={status}")

    # ── parent-side acceptance (only meaningful for REPORTED) ────────────────
    if status == REPORTED:
        c = rec.contract
        ok, evidence = run_acceptance(c)
        for e in evidence:
            print(f"[e2e]   check={e['check']!r} exit={e['exit_code']}")
        final = DONE if ok else REJECTED
        ledger.transition(tid, final, note=f"parent verified acceptance ok={ok}")
        print(f"[e2e] parent verdict: {final}")

    rec = ledger.get(tid)
    print(f"[e2e] FINAL status={rec.status}")
    print(f"[e2e] history:")
    for h in rec.history:
        print(f"[e2e]   {h['status']:16s} {h.get('note','')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "normal"))


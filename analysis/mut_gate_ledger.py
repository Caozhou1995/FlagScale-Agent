#!/usr/bin/env python3
"""Mutation gate for the Reading-ledger clause (Step B)."""
import hashlib
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/nfs/caozhou/fa_eval/FlagScale-Agent")
PLAN = ROOT / "flagscale_agent/react/tools/plan_create.py"
TEST = "tests/test_prompt_reading_ledger.py"


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()[:8]


def run():
    r = subprocess.run([sys.executable, "-m", "pytest", TEST, "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    return r.returncode, tail[0]


def gate(name, old, new):
    orig = PLAN.read_bytes()
    before = md5(PLAN)
    bak = PLAN.with_suffix(".py.mutbak")
    bak.write_bytes(orig)
    try:
        text = orig.decode()
        assert old in text, f"{name}: anchor not found"
        PLAN.write_text(text.replace(old, new, 1))
        rc, line = run()
        print(f"[{name}] mutated rc={rc} :: {line}")
        if rc == 0:
            print(f"[{name}] FAIL - mutation did NOT turn tests red")
            return False
        return True
    finally:
        shutil.copyfile(bak, PLAN)
        bak.unlink()
        assert md5(PLAN) == before, f"{name}: restore md5 mismatch"
        rc2, line2 = run()
        print(f"[{name}] restored md5={before} rc={rc2} :: {line2}")
        if rc2 != 0:
            print(f"[{name}] FAIL - restore not green")
            raise SystemExit(1)


ok = True

# M1: delete the whole reading-ledger clause.
text = PLAN.read_text()
start = '    "Reading ledger: when the task text leaves'
end = "shown impossible.\",\n"
i = text.index(start)
j = text.index(end, i) + len(end)
ok &= gate("M1-delete-clause", text[i:j], "")

# M2: remove the top-2 coverage duty (the discipline that makes ambiguity cost
# one run instead of six) while leaving an enumeration word behind.
ok &= gate(
    "M2-drop-top2-coverage",
    "dominates, cover the top-2 (or top-k) in the SAME run",
    "dominates, pick whichever reading seems most likely to be graded",
)

print("\nMUTATION GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

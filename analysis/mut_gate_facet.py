#!/usr/bin/env python3
"""Mutation gate for the Acceptance-facet manifest clause.

Two mutations, each must turn the regression tests RED, and restoring must
bring them back GREEN with a byte-identical file (md5 check).
"""
import hashlib
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path("/nfs/caozhou/fa_eval/FlagScale-Agent")
PLAN = ROOT / "flagscale_agent/react/tools/plan_create.py"
PROMPT = ROOT / "flagscale_agent/react/prompt.py"
TEST = "tests/test_prompt_acceptance_facet_manifest.py"


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()[:8]


def run_tests():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    return r.returncode, tail[0]


def gate(name, path, old, new):
    orig = path.read_bytes()
    before = md5(path)
    backup = path.with_suffix(path.suffix + ".mutbak")
    backup.write_bytes(orig)
    try:
        text = orig.decode()
        assert old in text, f"{name}: anchor not found"
        path.write_text(text.replace(old, new, 1))
        rc, line = run_tests()
        print(f"[{name}] mutated rc={rc} :: {line}")
        if rc == 0:
            print(f"[{name}] FAIL - mutation did NOT turn tests red")
            return False
        return True
    finally:
        shutil.copyfile(backup, path)
        backup.unlink()
        assert md5(path) == before, f"{name}: restore md5 mismatch"
        rc2, line2 = run_tests()
        print(f"[{name}] restored md5={before} rc={rc2} :: {line2}")
        if rc2 != 0:
            print(f"[{name}] FAIL - restore did not bring tests green")
            raise SystemExit(1)


ok = True

# M1: delete the whole acceptance-facet clause from the PLAN template.
start = "    \"Acceptance-facet manifest: BEFORE you build"
end = "your memory of it.\",\n"
text = PLAN.read_text()
i = text.index(start)
j = text.index(end, i) + len(end)
m1_old = text[i:j]
ok &= gate("M1-delete-clause", PLAN, m1_old, "")

# M2: soften the external anchoring back to free model choice.
m2_old = "the row COUNT is fixed by the clause "
m2_new = "the row count is whatever feels sufficient "
ok &= gate("M2-remove-anchoring", PLAN, m2_old, m2_new)

print("\nMUTATION GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

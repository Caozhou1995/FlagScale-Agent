"""Red phase: prove the new tests can actually fire.

Mutations applied one at a time to the module's own message constants, then
restored from an in-memory copy. If a mutation does NOT turn the suite red, the
corresponding test has no discriminating power and must be rewritten before I
trust it.
"""
import importlib
import subprocess
import sys

PATH = "flagscale_agent/react/guard/verification.py"
TEST = "tests/test_acceptance_act_message_is_copyable.py"

MUTATIONS = {
    "example back to <placeholders>": (
        '"command": "python3 audit.py",',
        '"command": "<the exact command you ran>",',
    ),
    "exactness rule deleted": (
        "Whitespace and letter case are ignored",
        "Notes",
    ),
    "unobservable example back to slots": (
        '"observed": "No such file or directory"}',
        '"observed": "<what it returned>"}',
    ),
    "renderer back to str.format": None,  # handled separately below
}


def run_tests():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TEST, "-q"],
        capture_output=True, text=True,
    )
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
    return r.returncode, tail


def main():
    orig = open(PATH).read()
    results = {}
    try:
        # baseline
        rc, tail = run_tests()
        results["baseline (unmutated)"] = (rc, tail)

        for name, mut in MUTATIONS.items():
            if mut is None:
                continue
            old, new = mut
            assert old in orig, f"anchor not found for {name!r}: {old!r}"
            open(PATH, "w").write(orig.replace(old, new, 1))
            results[name] = run_tests()
            open(PATH, "w").write(orig)

        # renderer mutation: .replace back to .format on the no-act constant
        anchor = '_STEP_DONE_NO_ACCEPTANCE_ACT.replace("{field}", field_label)'
        assert anchor in orig, "renderer anchor not found"
        open(PATH, "w").write(
            orig.replace(anchor, '_STEP_DONE_NO_ACCEPTANCE_ACT.format(field=field_label)', 1)
        )
        results["renderer back to str.format"] = run_tests()
        open(PATH, "w").write(orig)
    finally:
        open(PATH, "w").write(orig)

    print("=== RED-PHASE LEDGER ===")
    ok = True
    for name, (rc, tail) in results.items():
        verdict = "PASS(green)" if rc == 0 else "FAIL(red)"
        print(f"  {name:42s} -> rc={rc} {verdict:12s} | {tail}")
    base_rc = results["baseline (unmutated)"][0]
    print("\nbaseline rc =", base_rc)
    for name, (rc, _) in results.items():
        if name.startswith("baseline"):
            continue
        if rc == 0:
            print(f"  !! NOT DISCRIMINATING: {name} left suite green")
            ok = False
    print("\nRED PHASE VALID:", ok)


if __name__ == "__main__":
    main()

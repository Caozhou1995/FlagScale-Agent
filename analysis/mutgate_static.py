"""Mutation gate for the STATIC-PROMPT copies of the two FRAME clauses.

Each mutation softens ONE relation in prompt.py's SYSTEM_PROMPT_STATIC and must
turn tests/test_prompt_static_clause_anti_decay.py red. Copies the tree to a
tempdir, mutates there, never touches the checked-in source.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

os.chdir("/nfs/caozhou/fa_eval/FlagScale-Agent")
ORIG = pathlib.Path("flagscale_agent/react/prompt.py").read_text()
TESTS = ["tests/test_prompt_static_clause_anti_decay.py"]

# Anchors stay INSIDE one string literal; the static prompt is built from
# adjacent literals across source lines, so a cross-line anchor never matches.
MUTS = [
    ("S1 facet-merge", "one row per verifiable clause of the task text",
     "one row per clause you think is worth checking"),
    ("S2 facet-count-free", "row count is fixed by the clause count rather than by what you happen to think of",
     "row count is whatever feels sufficient"),
    ("S3 facet-no-probe", "a row without an executed probe is a sentence, not a check",
     "a row without a probe is still useful"),
    ("S4 ledger-unpinned", "each pinned to the task text's own words and to the artifact it would produce",
     "each described in your own words"),
    ("S5 dominance-optional", "run a dominance test against the consumer's own material",
     "run a dominance test if you have time"),
    # Case matters: the source literal reads "SAME run" and "PRINCIPLE 4's".
    ("S6 drop-top2", "cover the top-2 in the SAME run instead of betting",
     "cover only one reading instead of betting"),
    ("S7 drop-principle4", "PRINCIPLE 4's hedge/bet rule applied at build time",
     "A PRACTICAL rule of thumb applied at build time"),
    ("S8 drop-direction-of-loss",
     "first determine which direction the comparison penalizes",
     "consider whether the comparison is strict"),
    ("S9 ship-both-always",
     "when extras are penalized cover with a switch defaulted to the dominant reading",
     "when extras are penalized still ship both variants"),
    ("S10 drop-closeout-review",
     "close-out reviews this manifest row by row",
     "close-out may skim this manifest"),
    ("S11 drop-segmentation-rule",
     "segment on the task text's own sentences and bullets",
     "group the clauses however seems clearest"),
]


def run(text):
    d = tempfile.mkdtemp()
    shutil.copytree("flagscale_agent", os.path.join(d, "flagscale_agent"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    pathlib.Path(d, "flagscale_agent/react/prompt.py").write_text(text)
    shutil.copytree("tests", os.path.join(d, "tests"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini"):
        if os.path.exists(f):
            shutil.copy(f, d)
    env = dict(os.environ, PYTHONPATH=d)
    r = subprocess.run([sys.executable, "-m", "pytest"] + TESTS + ["-q", "-p", "no:cacheprovider"],
                       cwd=d, capture_output=True, text=True, env=env)
    last = (r.stdout or "").strip().splitlines()[-1:] or [""]
    shutil.rmtree(d, ignore_errors=True)
    return r.returncode, last[0]


def main():
    rc0, line0 = run(ORIG)
    print(f"BASELINE (unmutated): {'GREEN' if rc0 == 0 else 'RED'} :: {line0}")
    bad = []
    for name, old, new in MUTS:
        if old not in ORIG:
            print(f"{name}: ANCHOR-MISSING")
            bad.append(name + " (anchor)")
            continue
        rc, line = run(ORIG.replace(old, new, 1))
        if rc == 0:
            bad.append(name)
        print(f"{name}: {'RED (good)' if rc else 'SURVIVED (bad)'} :: {line}")
    print()
    print("STATIC MUTATION GATE:", "PASS — every mutation turns the suite red"
          if not bad and rc0 == 0 else f"FAIL — {bad}")


if __name__ == "__main__":
    main()

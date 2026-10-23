import shutil, subprocess, sys, tempfile, os, pathlib

os.chdir("/nfs/caozhou/fa_eval/FlagScale-Agent")
ORIG = pathlib.Path("flagscale_agent/react/tools/plan_create.py").read_text()
TESTS = ["tests/test_prompt_clause_anti_decay.py",
         "tests/test_prompt_acceptance_facet_manifest.py",
         "tests/test_prompt_reading_ledger.py"]

MUTS = [
    ("F1a merge-rows", "never fold several clauses into one row",
     "you may fold several clauses into one row"),
    # Anchors must stay INSIDE a single string literal: the clause is assembled
    # from adjacent literals across source lines, so an anchor spanning a line
    # break never matches the file bytes (that is why these two were MISSING).
    ("F1b expected-not-observed", "observed result when you ran it",
     "result you expect to see"),
    ("F1c drop-closeout", "Close-out reviews THIS manifest row by row",
     "Close-out may skim this manifest"),
    ("F1d hedge-optional", "cover the top-2 (or top-k) in the SAME run",
     "cover the top-2 if you have time"),
    ("F1e dominance-optional",
     "Then run a dominance test ",
     "Then optionally run a dominance test "),
    ("F1f drop-pinning", "each pinned to the task text's own words",
     "each described in your own words"),
]


def run(module_text):
    d = tempfile.mkdtemp()
    shutil.copytree("flagscale_agent", os.path.join(d, "flagscale_agent"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    pathlib.Path(d, "flagscale_agent/react/tools/plan_create.py").write_text(module_text)
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


rc0, line0 = run(ORIG)
print(f"BASELINE (unmutated): {'GREEN' if rc0 == 0 else 'RED'} :: {line0}")
survived = []
for name, old, new in MUTS:
    if old not in ORIG:
        print(f"{name}: ANCHOR-MISSING")
        survived.append(name + " (anchor missing)")
        continue
    rc, line = run(ORIG.replace(old, new, 1))
    verdict = "RED (good)" if rc != 0 else "SURVIVED (bad)"
    if rc == 0:
        survived.append(name)
    print(f"{name}: {verdict} :: {line}")

print()
print("MUTATION GATE:", "PASS — every mutation turns the suite red"
      if not survived and rc0 == 0 else f"FAIL — survived/anchor issues: {survived}")

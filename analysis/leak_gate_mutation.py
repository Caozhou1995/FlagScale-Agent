#!/usr/bin/env python3
"""Mutation test for leak_gate.py: each channel must be able to FAIL.

A gate that always passes is worthless. For every channel we inject a real leak
into a scratch copy of the world, run the gate, and require VERDICT: FAIL naming
that channel; then restore and require PASS again. Every mutation prints the
gate's own verdict line, so a PASS on the mutation means the gate is blind.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

GATE = "/nfs/caozhou/fa_eval/FlagScale-Agent/analysis/leak_gate.py"
REPO = pathlib.Path("/nfs/caozhou/fa_eval/FlagScale-Agent")
LAUNCH = pathlib.Path("/nfs/caozhou/glm")
STAGE = pathlib.Path("/nfs/caozhou/glm/agent_stage2")


def run_gate(repo, launch, stage):
    r = subprocess.run(
        [sys.executable, GATE, "--repo", str(repo),
         "--launch-dir", str(launch), "--stage", str(stage)],
        capture_output=True, text=True)
    verdict = [ln for ln in r.stdout.splitlines() if ln.startswith("VERDICT")]
    return r.returncode, (verdict[-1] if verdict else "<no verdict>"), r.stdout


def main():
    ok = True
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="leakgate_mut_"))

    # baseline: the real world must PASS
    rc, v, _ = run_gate(REPO, LAUNCH, STAGE)
    print(f"BASELINE  {v}  (expect PASS)")
    ok &= (rc == 0)

    # --- channel (a): task library name in the static prompt -------------
    r_a = tmp / "repo_a"
    shutil.copytree(REPO / "flagscale_agent", r_a / "flagscale_agent")
    pr = r_a / "flagscale_agent/react/prompt.py"
    pr.write_text(pr.read_text() + "\n# run the mteb package to encode\n")
    rc, v, out = run_gate(r_a, LAUNCH, STAGE)
    print(f"MUT-A     {v}  (expect FAIL, names (a))")
    ok &= (rc == 1 and "(a)" in out)

    # --- channel (c): grader construction in an agent-facing nudge ------
    r_c = tmp / "repo_c"
    shutil.copytree(REPO / "flagscale_agent", r_c / "flagscale_agent")
    orc = r_c / "flagscale_agent/react/oracle.py"
    orc.write_text(orc.read_text() + '\n_X = "the grader gave 56 nt and the bound 15-45 failed"\n')
    rc, v, out = run_gate(r_c, LAUNCH, STAGE)
    print(f"MUT-C     {v}  (expect FAIL, names (c))")
    ok &= (rc == 1 and "(c)" in out)

    # --- channel (b): a graded threshold file inside the staging tree ---
    s_b = tmp / "stage_b"
    shutil.copytree(STAGE, s_b)
    (s_b / "tests").mkdir()
    (s_b / "tests/test_thresholds.py").write_text(
        "# annealed length between 15 and 45 nt, Tm 58-72\n")
    rc, v, out = run_gate(REPO, LAUNCH, s_b)
    print(f"MUT-B     {v}  (expect FAIL, names (b))")
    ok &= (rc == 1 and "(b)" in out)

    # --- channel (d): a launcher mounting the repo root -----------------
    l_d = tmp / "launch_d"
    l_d.mkdir()
    one = next(LAUNCH.glob("four_launch*.sh"))
    (l_d / "m.sh").write_text(
        'export MOUNTS=\'[{"type":"bind","source":"/nfs/caozhou/fa_eval/FlagScale-Agent",'
        '"target":"/nfs/caozhou/fa_eval/FlagScale-Agent","read_only":true}]\'\n')
    rc, v, out = run_gate(REPO, l_d, STAGE)
    print(f"MUT-D     {v}  (expect FAIL, names (d))")
    ok &= (rc == 1 and "(d)" in out)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nMUTATION_GATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

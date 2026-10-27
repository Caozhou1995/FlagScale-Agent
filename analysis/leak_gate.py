#!/usr/bin/env python3
"""Falsifiable leak gate for the tbench 4-task harness.

Checks FOUR independent channels that each independently leaked the graded
task's answers in 2026-10-20. Every check prints what it looked at, so a PASS
is auditable and a FAIL names the offending file/line.

  (a) static prompt     -- no task-identity token in the prompt text
  (b) mount source      -- no launcher bind-mounts the repo root; staging is clean
  (c) agent-facing text -- nudge/judge/verification strings carry no task nouns
                           and no "<number> <unit>" threshold patterns
  (d) launcher sweep    -- every *.sh in the launch dir uses the staging mount

Exit 0 only if all checks pass. Designed to be runnable from CI.
Usage: python3 leak_gate.py [--repo DIR] [--launch-dir DIR] [--stage DIR]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

# Tokens that identify the graded tasks. Kept as a single list so the gate and
# a human reviewing it look at exactly the same set.
TASK_TOKENS = [
    "mteb", "torchmetrics", "sentence-transformers", "sentence_transformers",
    "primers.fasta", "primer", "oligotm", "melting temperature",
    "annealed", "stolen_a1", "cli_tool", "weights.json",
    "dna_insert", "seed5",
]
# NOTE: tokens must be UNIQUE TO THE GRADED TASK. Words like "bijection" or
# "forward" are ordinary methodology/code vocabulary and are intentionally NOT
# listed -- a gate that fires on them trains people to ignore it.

# "<number><optional space><unit>" thresholds that only a task's own spec states.
THRESHOLD_PAT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:nt|bp|degrees?|°c|nm)\b", re.IGNORECASE
)

# A worked EXAMPLE with concrete numbers is a leak even when the unit is
# omitted: quoting a measured value beside the stated bound discloses the
# graded instance's answer (what the value should have been, how far off it
# was). Mechanism text must describe the SHAPE of the decision, never replay a
# production number pair.
#
# Scope deliberately narrow. A bare "two numbers on one line" rule was tried
# first and produced 37 hits on enumeration markers ("(2) reference artifacts"),
# dates ("2026-10-27") and the harness's own regression counts -- a detector
# that fires on everything has no discriminating power. The leak needs BOTH a
# worked-example marker AND bound-semantics vocabulary AND two distinct numbers,
# which is what a task-answer replay actually looks like.
EXAMPLE_MARKER_PAT = re.compile(
    r"(measured case|worked example|for example|e\.g\.|your value"
    r"|came out|this very|actual measurement)",
    re.IGNORECASE,
)
BOUND_WORD_PAT = re.compile(
    r"(value|bound|maximum|minimum|limit|margin|threshold|ceiling|cap)",
    re.IGNORECASE,
)
BARE_NUMBER_PAIR_PAT = re.compile(r"(?<![\w.])(\d{1,6})(?![\w.])")

# Files whose strings are rendered into an agent message.
AGENT_FACING = [
    "flagscale_agent/react/prompt.py",
    "flagscale_agent/react/oracle.py",
    "flagscale_agent/react/judge.py",
    "flagscale_agent/react/guard/verification.py",
    # Tool schemas and the instruments module are read by the model verbatim
    # (base.py converts `parameters` into the tool's JSON schema), so they are
    # agent-facing text in exactly the same sense as the prompt.
    "flagscale_agent/react/tools/plan_update.py",
    "flagscale_agent/react/tools/plan_create.py",
    "flagscale_agent/react/instruments.py",
]

REPO_ROOT_MOUNT = '"source":"/nfs/caozhou/fa_eval/FlagScale-Agent"'
FORBIDDEN_IN_STAGE = ["tests", "analysis", "examples", "build",
                      "flagscale_agent.egg-info"]


TOKEN_RES = [(t, re.compile(r"(?<![a-z0-9_])" + re.escape(t) + r"(?![a-z0-9_])"))
             for t in TASK_TOKENS]


def hit_tokens(line: str) -> list[str]:
    """Tokens present in `line`, matched on word boundaries."""
    low = line.lower()
    return [t for t, rx in TOKEN_RES if rx.search(low)]


def read(p: pathlib.Path) -> str:
    return p.read_text(errors="replace") if p.is_file() else ""


def check_prompt(repo: pathlib.Path) -> list[str]:
    fails = []
    p = repo / "flagscale_agent/react/prompt.py"
    text = read(p)
    if not text:
        return [f"(a) MISSING {p} -- cannot verify"]
    for i, ln in enumerate(text.splitlines(), 1):
        for tok in hit_tokens(ln):
            fails.append(f"(a) {p}:{i} contains task token {tok!r}")
    return fails


def check_agent_facing(repo: pathlib.Path) -> list[str]:
    fails = []
    for rel in AGENT_FACING:
        p = repo / rel
        text = read(p)
        if not text:
            fails.append(f"(c) MISSING {p} -- cannot verify")
            continue
        for i, ln in enumerate(text.splitlines(), 1):
            toks = hit_tokens(ln)
            if toks:
                fails.append(f"(c) {p}:{i} task token {toks} in "
                             f"agent-facing text: {ln.strip()[:110]}")
                continue
            if THRESHOLD_PAT.search(ln):
                fails.append(f"(c) {p}:{i} threshold pattern in "
                             f"agent-facing text: {ln.strip()[:110]}")
                continue
            nums = {int(m.group(1)) for m in BARE_NUMBER_PAIR_PAT.finditer(ln)}
            if (len(nums) >= 2 and EXAMPLE_MARKER_PAT.search(ln)
                    and BOUND_WORD_PAT.search(ln)):
                fails.append(
                    f"(c) {p}:{i} worked example with concrete numbers "
                    f"{sorted(nums)} in agent-facing text "
                    f"(describes the shape of the decision, never a number "
                    f"pair): {ln.strip()[:110]}")
    return fails


def check_mounts(launch_dir: pathlib.Path) -> list[str]:
    fails = []
    files = sorted(launch_dir.glob("*.sh"))
    if not files:
        return [f"(b)/(d) no *.sh under {launch_dir}"]
    for p in files:
        text = read(p)
        if REPO_ROOT_MOUNT in text:
            fails.append(f"(d) {p.name} still bind-mounts the repo root")
    return fails


def check_stage(stage: pathlib.Path) -> list[str]:
    fails = []
    if not stage.is_dir():
        return [f"(b) staging dir missing: {stage}"]
    for bad in FORBIDDEN_IN_STAGE:
        if (stage / bad).exists():
            fails.append(f"(b) staging contains forbidden path: {bad}")
    # Staging must be installable and non-empty, otherwise "clean" is vacuous.
    for need in ["flagscale_agent", "pyproject.toml", "README.md"]:
        if not (stage / need).exists():
            fails.append(f"(b) staging missing required entry: {need}")
    if stage.is_dir():
        leaks = 0
        for p in stage.rglob("*"):
            if p.suffix not in {".py", ".md", ".toml", ".txt"} or not p.is_file():
                continue
            hits = [t for ln in read(p).splitlines() for t in hit_tokens(ln)]
            if hits:
                leaks += 1
                fails.append(f"(b) staging file carries a task token: {p} {sorted(set(hits))}")
        print(f"    staging scanned, files with task tokens: {leaks}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/nfs/caozhou/fa_eval/FlagScale-Agent")
    ap.add_argument("--launch-dir", default="/nfs/caozhou/glm")
    ap.add_argument("--stage", default="/nfs/caozhou/glm/agent_stage2")
    a = ap.parse_args()

    repo = pathlib.Path(a.repo)
    launch = pathlib.Path(a.launch_dir)
    stage = pathlib.Path(a.stage)

    print(f"repo={repo}\nlaunch_dir={launch}\nstage={stage}\n")
    groups = {
        "(a) static prompt": check_prompt(repo),
        "(c) agent-facing text": check_agent_facing(repo),
        "(b) staging tree": check_stage(stage),
        "(d) launcher mounts": check_mounts(launch),
    }
    total = 0
    for name, fails in groups.items():
        print(f"{name}: {'PASS' if not fails else 'FAIL'} ({len(fails)})")
        for f in fails:
            print("   ", f)
        total += len(fails)

    print(f"\nVERDICT: {'PASS' if total == 0 else 'FAIL'} (violations={total})")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

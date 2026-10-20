#!/usr/bin/env python3
"""Per-round, per-task reward table plus a zero-leak flag.

Answers two questions with data rather than memory:
  1. what did each task score in each round?
  2. was that round leak-free? A round is counted zero-leak only if
     (a) its launcher mounted the staging tree (never the repo root), AND
     (b) its staging snapshot contained no task token.

`leak_epoch` is decided from the prompt's own git history: for each round we
read the HEAD the trial actually ran under (jobs/<round>/config.json, falling
back to the newest prompt.py commit at or before the round's start time) and
count task tokens in `prompt.py` at that commit. That makes the flag falsifiable
against `git show` rather than an assumption.

Usage: python3 round_table.py [--jobs DIR] [--repo DIR]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

TASKS = [
    ("dna", "dna-insert"),
    ("me", "model-extraction-relu-logits"),
    ("mteb", "mteb-retrieve"),
    ("pt", "pytorch-model-cli"),
]

TOKENS = ["mteb", "torchmetrics", "sentence-transformers", "primers.fasta",
          "oligotm", "melting temperature", "annealed", "stolen_a1",
          "cli_tool", "weights.json", "dna_insert", "seed5"]


def reward(trial_dir: pathlib.Path) -> str:
    f = trial_dir / "verifier" / "reward.txt"
    if not f.is_file():
        return "-"
    v = f.read_text().strip()
    return v if v else "-"


def tokens_at(repo: pathlib.Path, commit: str) -> int:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:flagscale_agent/react/prompt.py"],
            capture_output=True, text=True, check=True).stdout.lower()
    except subprocess.CalledProcessError:
        return -1
    return sum(1 for t in TOKENS if t in out)


def prompt_commits(repo: pathlib.Path) -> list[tuple[str, str]]:
    out = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%H %ct", "--",
         "flagscale_agent/react/prompt.py"],
        capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return [(c.split()[0], c.split()[1]) for c in out if c.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="/nfs/caozhou/glm/jobs")
    ap.add_argument("--repo", default="/nfs/caozhou/fa_eval/FlagScale-Agent")
    a = ap.parse_args()
    jobs = pathlib.Path(a.jobs)
    repo = pathlib.Path(a.repo)

    commits = prompt_commits(repo)          # newest -> oldest
    tcache: dict[str, int] = {}

    rounds = sorted(p for p in jobs.glob("four_m*") if p.is_dir())
    print(f"{'round':<16}{'dna':>5}{'me':>5}{'mteb':>5}{'pt':>5}{'total':>7}"
          f"{'prompt_tokens':>15}  mount")
    rows = []
    for rd in rounds:
        vals, tot = [], 0
        for _, task in TASKS:
            dirs = sorted(rd.glob(f"{task}__*"))
            v = reward(dirs[0]) if dirs else "-"
            vals.append(v)
            tot += (v == "1")

        # A round is "mounted clean" when its own config records the staging
        # mount (never the repo root). This is the round's OWN artifact, not an
        # inference, so it is the primary signal.
        mounted_clean = None
        cfg = rd / "config.json"
        if cfg.is_file():
            try:
                blob = cfg.read_text()
                mounts_repo_root = '"source": "/nfs/caozhou/fa_eval/FlagScale-Agent"' in blob
                mounts_stage = "/nfs/caozhou/glm/agent_stage" in blob
                mounted_clean = mounts_stage and not mounts_repo_root
            except Exception:
                pass

        # Which prompt.py revision was current when this round started? Use the
        # newest prompt-touching commit at or before the round dir's ctime.
        commit = None
        try:
            started = int((rd / "config.json").stat().st_mtime)
        except OSError:
            started = int(rd.stat().st_mtime)
        for c, t in commits:
            if int(t) <= started:
                commit = c
                break
        ntok = tcache.get(commit, None) if commit else None
        if ntok is None and commit:
            ntok = tokens_at(repo, commit)
            tcache[commit] = ntok

        shown = "?" if ntok is None else str(ntok)
        flag = "stage" if mounted_clean else ("repo-root" if mounted_clean is False else "?")
        print(f"{rd.name:<16}{vals[0]:>5}{vals[1]:>5}{vals[2]:>5}{vals[3]:>5}"
              f"{tot:>7}/4{shown:>15}  {flag}")
        rows.append((rd.name, vals, tot, ntok, mounted_clean))

    print()
    def summarize(label, sel):
        if not sel:
            print(f"{label}: none")
            return
        print(f"{label}: n={len(sel)} best={max(r[2] for r in sel)}/4 "
              f"totals={sorted(r[2] for r in sel)}")

    summarize("mount=stage AND prompt_tokens==0 (fully clean)",
              [r for r in rows if r[4] and r[3] == 0])
    summarize("mount=stage but prompt_tokens>0 (partially clean)",
              [r for r in rows if r[4] and r[3] and r[3] > 0])
    summarize("mount=repo-root (channel 3 open)",
              [r for r in rows if r[4] is False])
    return 0


if __name__ == "__main__":
    sys.exit(main())

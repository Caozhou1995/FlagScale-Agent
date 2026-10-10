# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KillSafetyGuard — deterministic self-kill hazard detection.

The agent's own process tree (the python agent plus the shell wrappers it
spawns) can be reached by kill-style commands the agent writes:

  • `pkill`/`killall <pattern>` self-matches its own wrapper shell, whose
    command line CONTAINS the pattern text.
  • `kill <pid>` / `kill $(pgrep ...)` can name the agent's own pid.
  • a process-list-driven `kill` (fed by ps / grep / pgrep) can SELECT the
    agent — the exact shape that killed the qemu-startup trial (a cleanup loop
    whose `kill -9` matched the agent's own argv, because the match pattern
    came from the task text the agent was reading).

The `pkill`/`killall` family and the legacy `kill ... flagscale` shape are
already neutralised by `shell.py::_protect_self_kill`, which rewrites the
pipeline to exclude the agent's pid tree before it runs. This guard does NOT
re-warn on those (no cry-wolf): it fills the GAP that no rewrite covers — bare
`kill`, command-substitution kill, and process-list-driven kill whose pattern
does not include an agent token.

Deterministic: no LLM judge, no semantic-state judgement. It surfaces the
hazard and leaves the DECISION to the LLM (inject), escalating to an
OVERRIDABLE block only when the hazard is provable (the target names the
agent's own pid, or the selection pipeline cannot be shown to exclude it).
"""

from __future__ import annotations

import os
import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# A process-listing feed: a kill whose target is resolved from a process list
# rather than a literal pid.
_PROC_FEED_RE = re.compile(r"\b(pgrep|ps|grep)\b")

# Literal numeric pid arguments to a kill (e.g. `kill -9 12345 6789`).
_NUMERIC_PID_RE = re.compile(r"\b(\d{2,7})\b")

# `kill` invoked AS A COMMAND — at the start of a clause (string start, or right
# after a shell separator ;/&/|), optionally behind a known launcher — or as the
# target of `xargs`. Requiring COMMAND POSITION is what stops the guard matching
# the mere WORD "kill" inside a filename, a quoted string, or a grep pattern:
# `cat kill.sh`, `grep -n kill README`, `echo "kill it"` must NOT trip it
# (a guard that blocks a read-only grep is worse than no guard).
_LAUNCHER = (
    r"(?:sudo|doas|nohup|command|nice|ionice|stdbuf|env)\b(?:\s+-\S+|\s+\S+=\S+)*\s+"
    r"|timeout\b(?:\s+-\S+)*\s+\S+\s+"
)
_KILL_CMD_RE = re.compile(r"(?:^|[;&|])\s*(?:" + _LAUNCHER + r")*(?P<k>kill\b)")
_XARGS_KILL_RE = re.compile(r"\bxargs\b(?:\s+-\S+)*\s+(?P<k>kill\b)")

# Shell separators that END a kill's own argument region (its target list).
_KILL_ARG_END_RE = re.compile(r"[|;&]")


def _kill_occurrence(cmd: str):
    """Return the earliest match of `kill` in COMMAND position, or None.

    'Command position' = start of a clause, right after a shell separator
    ;/&/|, or as the target of xargs — never the bare word "kill" inside an
    argument. This is the no-cry-wolf gate.
    """
    cands = [rx.search(cmd) for rx in (_KILL_CMD_RE, _XARGS_KILL_RE)]
    cands = [c for c in cands if c is not None]
    if not cands:
        return None
    return min(cands, key=lambda c: c.start("k"))


def _agent_cmdlines() -> tuple[set[int], list[str]]:
    """Return (agent_pid_set, agent_cmdline_strings).

    Reuses the same pid tree that `shell.py::_get_agent_pids` protects, then
    reads each /proc/<pid>/cmdline so a pattern can be tested against the
    command lines of processes the agent must not kill.
    """
    try:
        from flagscale_agent.react.tools.shell import _get_agent_pids
        pids = _get_agent_pids()
    except Exception:
        pids = {os.getpid(), os.getppid()}
    cmdlines: list[str] = []
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
            if raw.strip():
                cmdlines.append(raw)
        except OSError:
            pass
    return set(pids), cmdlines


class KillSafetyGuard(Guard):
    """Warn/block on kill-style commands that can reach the agent itself."""

    name = "kill_safety"
    priority = 12  # just after the LLM shell-safety judge (10)

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None
        cmd = ctx.tool_args.get("command", "")
        if not cmd or not isinstance(cmd, str):
            return None
        # Gate on `kill` in COMMAND POSITION only (start of a clause, after a
        # shell separator ;/&/|, or the target of xargs). This deliberately does
        # NOT match the mere word "kill" inside an argument — `cat kill.sh`,
        # `grep -n kill README`, `echo "kill it"` pass silently (blocking a
        # read-only grep would be worse than no guard). The pkill/killall family
        # is also outside this gate: it is neutralised by the shell rewrite and
        # not a self-select hazard on its own (no cry-wolf). The guard's job is
        # the bare-kill surface: it illuminates the hazard while the rewrite
        # carries the root-fix (defense in depth).
        km = _kill_occurrence(cmd)
        if km is None:
            return None
        kpos = km.start("k")

        # The kill's OWN argument region: from the enclosing clause boundary
        # (`;`/`&` — NOT a single `|`, which feeds the kill) through the end of
        # its argument list (next unquoted `;`/`&`/`|`). Scoping the numeric-pid
        # and process-feed checks to this region keeps a pid or a `grep` that
        # merely appears ELSEWHERE on the line from false-tripping the guard.
        clause_start = max(cmd.rfind(";", 0, kpos), cmd.rfind("&", 0, kpos)) + 1
        args_end = len(cmd)
        for sep in re.finditer(r"[;&|]", cmd[kpos:]):
            args_end = kpos + sep.start()
            break
        region = cmd[clause_start:args_end]

        pids, _cmdlines = _agent_cmdlines()
        pid_str = ", ".join(str(p) for p in sorted(pids))

        # Rule 1 (PROVABLE): an explicit numeric pid IN THE KILL'S ARGUMENTS that
        # IS one of the agent's own pids → a literal self-kill. Block (overridable).
        for m in _NUMERIC_PID_RE.finditer(region):
            try:
                cand = int(m.group(1))
            except ValueError:
                continue
            if cand in pids:
                return GuardVerdict.block(
                    f"[KillSafety] This command targets pid {cand}, which IS "
                    f"part of your own process tree (agent pids: {pid_str}). "
                    "Running it kills the agent mid-task and aborts the run. "
                    "If you meant a different process, resolve its pid "
                    "explicitly (e.g. `pgrep -f <unique-name>` and pick the "
                    "non-agent pid); to proceed anyway, give an override reason.",
                    reason=f"kill_targets_agent_pid_{cand}",
                    category="kill_safety",
                )

        # Rule 2 (PROVABLE-ISH): the target set is SELECTED by a process-list
        # feed (ps/grep/pgrep) — either inside the kill's args (`kill $(...)`)
        # or in the pipeline feeding it (`... | xargs kill`). The match pattern
        # lives in the wrapper shell's own command line, so the selection can
        # include the agent. Block (overridable) — the qemu-startup self-kill shape.
        if _PROC_FEED_RE.search(region):
            return GuardVerdict.block(
                "[KillSafety] This command kills processes SELECTED by a "
                "process-list pipeline (ps/grep/pgrep). Your own command line "
                "contains the match pattern, so the selection can include the "
                f"agent itself (agent pids: {pid_str}) — this is how a cleanup "
                "loop kills its own run. Make the selection exclude your pid "
                "(`grep -v <agent-pid>`, or match a name that cannot occur in "
                "your own argv), then re-issue. To proceed anyway, give an "
                "override reason.",
                reason="kill_from_process_list_may_select_agent",
                category="kill_safety",
            )

        # Rule 3 (ADVISORY): a bare `kill` whose target is not a literal,
        # verifiable pid — cannot be proven safe, so surface the hazard without
        # blocking and let the LLM decide.
        return GuardVerdict.inject(
            "[KillSafety] Your command runs `kill` on a target that is not a "
            "literal, verifiable pid. Double-check the target set does not "
            f"include your own process tree (agent pids: {pid_str}) before it "
            "runs — a self-kill aborts the whole task.",
            reason="bare_kill_unverified_target",
            category="kill_safety",
        )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        return bool(reason and len(reason.strip()) > 5)

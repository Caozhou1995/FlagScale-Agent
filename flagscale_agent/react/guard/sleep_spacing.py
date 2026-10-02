# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License near the top of this file.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SleepSpacingGuard — post-inject on sleep-padded rapid-fire polling.

Anti-pattern (observed in-session): LongTimeShellGuard caps a foreground sleep
at 30s, so the agent learns to issue `sleep 28; <the same probe>` in a tight
loop — each round re-reads the same PENDING state at full LLM-iteration cost
(~74-84s startup + full prompt). The sleep does not make it productive waiting;
it is bare polling wearing a sleep costume, and it also dodges the guard that
exists precisely to prevent this shape.

Companion to PollSpacingGuard (which catches bare poll_tasks/dispatch_many
rounds). This guard covers the shell-channel variant: a FOREGROUND shell
command that (a) contains a `sleep` invocation and (b) reads state it has
already probed this streak — today approximated by (b') the streak counter:
consecutive sleep-bearing probe commands with no intervening non-probe tool.

Design:
- post-only + inject-only: fires AFTER an executed sleep-probe shell round;
  never blocks. LongTimeShellGuard (pre) still owns the >30s block; this guard
  only pattern-nudges the repeated <=30s variant.
- background=true shell calls NEVER count: a backgrounded wait + doing other
  work is the doctrine's answer, and `sleep_jobs`-style background loops are
  legitimate.
- streak advanced ONLY for executed, non-blocked, foreground, sleep-bearing
  shell calls; dedup per LLM iteration via len(ctx.messages) (same mechanism
  and rationale as PollSpacingGuard: check_post runs per tool call, but one
  assistant message can emit several calls, and tool results are appended
  after the per-call guard loop).
- message escalates at streak >= 4 (after 3 nudges it is spinning, not
  waiting); reset when a non-sleep shell command or any other tool runs, or
  on a fresh user turn (reset_turn).
- the streak counter measures ROUNDS, and a round that genuinely waits ≥60s
  (multiple sleeps summing past that) is treated as patient enough — it does
  not advance the streak (sparse polling is what we want; only the 25-30s
  rapid-fire shape is flagged).
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# Consecutive rapid-fire sleep-probe rounds before the first inject.
INJECT_THRESHOLD = 3
# Harder message once the agent has ignored this many nudges.
ESCALATE_THRESHOLD = 4
# A single command whose sleeps sum to at least this is "patient waiting" —
# it does not advance the rapid-fire streak.
PATIENT_SLEEP_SECONDS = 60.0

_QUOTED_SPAN_RE = re.compile(r'"[^"]*"|\'[^\']*\'')
_DURATION_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)(?P<suffix>[smhd]?)$",
                          re.IGNORECASE)
_SUFFIX_TO_SEC = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def _split_segments(command: str) -> list[list[str]]:
    """Token lists per shell segment (;, &&, ||, |, newline) — quoted spans
    are data, not command syntax."""
    segments: list[list[str]] = []
    for seg in re.split(r"(?:&&|\|\||[;|\n])", _QUOTED_SPAN_RE.sub("", command)):
        toks = seg.split()
        if toks:
            segments.append(toks)
    return segments


def _sleep_seconds_in_segment(toks: list[str]) -> float:
    """Total seconds of a `sleep N [N...]` invocation in command position of
    this segment (0.0 when absent). Mirrors LongTimeShellGuard._scan_sleep."""
    i = 0
    while i < len(toks) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[i]):
        i += 1
    if i >= len(toks) or toks[i] != "sleep":
        return 0.0
    total = 0.0
    seen = False
    for tok in toks[i + 1:]:
        m = _DURATION_RE.match(tok)
        if not m:
            break
        total += float(m.group("num")) * _SUFFIX_TO_SEC[m.group("suffix").lower()]
        seen = True
    return total if seen else 0.0


def _probe_profile(command: str) -> tuple[bool, float]:
    """(is_sleep_probe, total_sleep_seconds) for a shell command."""
    total = 0.0
    seen = False
    for toks in _split_segments(command):
        s = _sleep_seconds_in_segment(toks)
        if s > 0.0:
            total += s
            seen = True
    return seen, total


def _is_sleep_probe_shell(tool_name: str, tool_args: dict) -> bool:
    """A foreground shell call that carries at least one sleep invocation."""
    if tool_name != "shell":
        return False
    if bool(tool_args.get("background", False)):
        return False
    command = str(tool_args.get("command", ""))
    seen, _ = _probe_profile(command)
    return seen


class SleepSpacingGuard(Guard):
    """Post-tool guard: nudge when sleep-padded probes come rapid-fire."""

    name = "sleep_spacing"
    priority = 71  # advisory tier, next to PollSpacingGuard (70)

    def __init__(self):
        self._streak = 0
        self._last_marker: int | None = None

    def reset_turn(self):
        self._streak = 0
        self._last_marker = None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None  # post-only

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        name = ctx.tool_name or ""
        if name == "shell":
            args = ctx.tool_args or {}
            if bool(args.get("background", False)):
                # Backgrounded waits are the doctrine — the wait context ends.
                self._streak = 0
                self._last_marker = None
                return None
            seen, total = _probe_profile(str(args.get("command", "")))
            if not seen:
                # An ordinary shell command — wait context over.
                self._streak = 0
                self._last_marker = None
                return None
            if total >= PATIENT_SLEEP_SECONDS:
                # Patient single-command wait: legitimate spacing, no advance.
                self._last_marker = len(ctx.messages)
                return None
            result = ctx.tool_result or ""
            if "[BLOCKED BY GUARD]" in result:
                return None  # never advanced — do not count or reset
            marker = len(ctx.messages)
            if marker != self._last_marker:
                self._streak += 1
                self._last_marker = marker
            if self._streak >= INJECT_THRESHOLD:
                return self._advisory(self._streak)
            return None
        # Any other tool ends the wait context.
        self._streak = 0
        self._last_marker = None
        return None

    def _advisory(self, streak: int) -> GuardVerdict:
        if streak >= ESCALATE_THRESHOLD:
            body = (
                f"[SleepSpacing] {streak} consecutive short-sleep probes with no "
                "other work between them — sleeping ~28s to slide under the 30s "
                "guard is still bare polling, one full LLM iteration each. STOP "
                "this loop: do real work between looks (prep next steps, update "
                "plan/memory, audit the output contract), or background the long "
                "job (background=true) and check shell_jobs with bounded waits. "
                "If nothing else can proceed, take ONE sleep >= 60s in a single "
                "command — patient spacing is not flagged."
            )
        else:
            body = (
                f"[SleepSpacing] {streak} sleep-padded probes in a row. Chained "
                "`sleep 28; probe` rounds are polling in costume — each burns a "
                "full LLM iteration to re-read unchanged state. Between looks, "
                "do real work, or issue ONE longer sleep (>=60s) instead of "
                "many 25-30s ones, or background the wait."
            )
        return GuardVerdict.inject(
            body,
            reason="rapid_fire_sleep_probes",
            category="sleep_spacing",
        )

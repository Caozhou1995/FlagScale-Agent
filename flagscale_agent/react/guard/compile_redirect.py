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

"""CompileRedirectGuard — pre-launch block on long/verbose build commands that do
NOT persist their output to a file.

Motivation (near/far + monitor-stall):
  A compile command (make -j / cmake --build / cargo build ...)
  can run for many minutes and spew huge output. Two failure modes surface:
    1. Piped to a pager/filter (`make -j | tail`) the child full-buffers its stdout;
       nothing appears until it exits, so the run looks hung.
    2. The monitor's health judge can only advise AFTER the command already launched
       — it cannot retroactively add a redirect to a running command.
  Teeing the output (`... 2>&1 | tee build.log`) keeps it visible to the live monitor
  AND inspectable after; a BARE redirect (`> build.log 2>&1`) hides all output from the
  monitor for the whole run — a black box that looks hung. `tail -f` on a redirected
  log streams it back to the terminal and also qualifies.

This guard is STATELESS: it keys off the command text itself. A compile command whose
output is not visible to the monitor (no tee / no tail -f, incl. bare `> file`) is
blocked; once the LLM re-issues it WITH tee (or a streaming tail -f) it passes
naturally. Genuine exceptions (command already streams its own progress, output
intentionally discarded) are released with a one-line _override_reason.
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Build/compile drivers whose invocation is typically long-running and verbose.
# Matched at command start OR right after a shell separator (; && || | & newline).
_COMPILE_TOKENS = (
    r"make",
    r"cmake\s+--build",
    r"ninja",
    r"g\+\+", r"gcc", r"cc", r"c\+\+", r"clang\+\+", r"clang",
    r"cargo\s+(?:build|test|install)",
    r"go\s+build",
    r"mvn", r"gradle", r"\./gradlew", r"javac",
    r"opam\s+(?:install|reinstall|switch\s+create)",
    r"cabal\s+build",
    r"bazel\s+build",
    r"python[0-9.]*\s+setup\.py\s+(?:build|install)",
)

# Leading wrapper/env-setter tokens that may precede the real build driver:
#   sudo, VAR=val env prefixes, env -u VAR / env -i / env VAR=val,
#   stdbuf -oL -eL, nice -n 5, time, nohup, ...
# Each may carry its own flags/args, so consume greedily up to the driver.
# NB: the `env` branch must eat its OWN options (`-u VAR` unset pairs, `-i`,
# `VAR=val` assignments); otherwise `env -u HTTP_PROXY opam install ...` slips
# past the whole guard (the `-u` breaks the match right after `env `), and a
# `... | tail` build then runs invisible to the monitor. Seen live on compcert.
_WRAPPER = (
    r"(?:sudo\s+|[A-Za-z_][A-Za-z0-9_]*=\S+\s+|"
    r"env(?:\s+(?:-[A-Za-z]+\s+\S+|-[A-Za-z]+|[A-Za-z_][A-Za-z0-9_]*=\S+))*\s+|"
    r"stdbuf(?:\s+-\S+)*\s+|nice(?:\s+-\S+)*\s+|time\s+|nohup\s+)*"
)

# A compile token appearing at the start of the command or just after a separator.
# Trailing boundary is a lookahead for whitespace/end/separator rather than \b —
# tokens like g++ / c++ end in '+', which has no \b against a following space.
_COMPILE_RE = re.compile(
    r"(?:^|[|;&\n]|&&|\|\|)\s*" + _WRAPPER
    + r"(?:" + "|".join(_COMPILE_TOKENS) + r")(?=\s|$|[|;&])"
)

# The build driver anchored at the START of a segment (no leading separator), used
# to inspect what immediately follows the driver within one segment.
_DRIVER_HEAD_RE = re.compile(
    r"^" + _WRAPPER + r"(?:" + "|".join(_COMPILE_TOKENS) + r")(?=\s|$|[|;&])"
)

# Segment separators. `&&`/`||` are subsumed by the `&`/`|` char class; `2>&1`
# redirects also split here but leave harmless fragments (the driver segment keeps
# its version/help flag as its first token, which is all the transient check reads).
_SEG_SPLIT_RE = re.compile(r"[|;&\n]+")

# Transient sub-invocations of a build driver that do NOT start a real build:
# version/help queries and dry-run/print-only flags. These return near-instantly
# with tiny output, so the monitor-visibility concern does not apply. Matched as
# the FIRST token after the driver.
_TRANSIENT_FLAG_RE = re.compile(
    r"^(?:--version|-version|-V|-v|--help|-h|--usage|--dry-run|-n|--just-print|"
    r"--recon|-p|--print-data-base|--query|-q|--numeric-version)$"
)


def _has_real_build(cmd: str) -> bool:
    """True if the command contains at least one GENUINE build invocation.

    A build driver that appears only as an argument to a probe (`which make`,
    `type gcc`) never matches here — it is not at a segment head. A driver invoked
    with a version/help/dry-run flag (`make --version`, `gcc -v`) is a transient
    query, not a build, and is excluded. Only a driver at a segment head whose
    first argument is not a transient flag counts as a real build.
    """
    for seg in _SEG_SPLIT_RE.split(cmd):
        seg = seg.strip()
        if not seg:
            continue
        dm = _DRIVER_HEAD_RE.match(seg)
        if not dm:
            continue  # segment does not START with a build driver
        after = seg[dm.end():].strip()
        first_tok = after.split()[0] if after.split() else ""
        if first_tok and _TRANSIENT_FLAG_RE.match(first_tok):
            continue  # transient version/help/dry-run query — not a real build
        return True
    return False


def _streams_to_monitor(cmd: str) -> bool:
    """True if the command's output stays VISIBLE to the live monitor.

    The monitor reads the child's stdout pipe in real time. A bare redirect
    (`> file 2>&1`, `&> file`, `>> file`) sends everything to the file and leaves
    the monitor's stdout pipe EMPTY for the whole run — the health judge can only
    peek the file out-of-band and typically sees stale/buffered content, so the
    run looks hung even though it is progressing. Two forms keep the monitor fed:

      • `... | tee file`   — child writes to the terminal AND the file
      • `tail -f file`     — a redirected file is streamed back to the terminal
                             (e.g. `<cmd> > f.log 2>&1 & tail -f f.log`)

    Only these qualify. A pure file redirect does NOT.
    """
    # tee: streams to terminal + file simultaneously (preferred).
    if re.search(r"\btee\b", cmd):
        return True
    # tail -f / -F / -nf / -fn20 ... : streams a (redirected) file back to stdout.
    if re.search(r"\btail\b[^|;&\n]*-[A-Za-z]*[fF]", cmd):
        return True
    return False


_MESSAGE = """[CompileRedirectGuard] Long-running build/compile command whose output is not visible to the monitor.

The live monitor watches the child's stdout. Two forms make a build LOOK HUNG while it progresses:
  • piped to a pager/filter (`| tail` / `| head`) — the child full-buffers; nothing until it exits.
  • a BARE file redirect (`> build.log 2>&1`) — all output goes to the file, the monitor's stdout pipe stays empty, out-of-band peeking sees stale content. A pure redirect is a BLACK BOX and does NOT satisfy this guard.

Keep output flowing to the terminal AND a file, visible live and inspectable after:
  <build cmd> 2>&1 | tee build.log                  # preferred: terminal + file
  <build cmd> > build.log 2>&1 &  tail -f build.log # background + stream the log back
`tee` (or `tail -f` streaming the log back) is what the monitor needs — not a bare `> file`.

Parallelism — probe first, then build with BOUNDED parallelism:
  • Probe single-threaded FIRST: a parallel first build buries the first real config/toolchain/dep error in noise; a clean probe surfaces it:
      timeout 120 make -j1 2>&1 | tee build.probe.log   # probe: clean first error
  • NEVER bare `make -j` or `make -j$(nproc)` without ALL THREE caps:
      1. CGROUP cap: in a container `nproc` and /proc/meminfo report the HOST, not your slice. A 2GB/1-CPU cgroup with `-j32` forks 32 cc1plus needing ~1GB each → instant OOM-kill (`Killed signal terminated program cc1plus`) and 1-CPU thrash. Read the cgroup limit FIRST and take the min against the host:
      Q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null | awk '{if($1!="max")print int($1/$2)}')   # cores v2
      [ -z "$Q" ] && Q=$(awk 'END{if($1>0)print int($1/p)}' p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null) /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null)  # v1
      NP=$(nproc); C=${Q:-$NP}; [ "$C" -gt "$NP" ] 2>/dev/null && C=$NP; [ "$C" -lt 1 ] 2>/dev/null && C=1
      LIM=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); [ "$LIM" = "max" ] && LIM=""      # mem v2
      [ -z "$LIM" ] && LIM=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)  # v1
      HOSTKB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
      if [ -n "$LIM" ] && [ "$LIM" -lt $((HOSTKB*1024)) ] 2>/dev/null; then AVKB=$((LIM/1024)); else AVKB=$HOSTKB; fi
      M=$(( AVKB/2000000 ))                              # ~2GB/compiler, memory-bounded
      J=$C; J=$(( M<J ? M : J )); J=$(( J<1 ? 1 : J ))   # min(cgroup-cores, mem-bound)
      J=$(( J>32 ? 32 : J ))                             # absolute cap: >32 thrashes cache/BW
      make -j$J 2>&1 | tee build.log                     # bounded-parallel build
      2. MEMORY cap: each compiler uses 1-2GB; an unbounded `-j` forks as many as the graph allows and OOMs/thrashes a small (4-8GB) box.
      3. ABSOLUTE cap (32): a big host (256 cores / ~1TB) passes the memory cap with -j256 — NOT faster; that many heavy C++ compilers saturate cache and memory bandwidth and starve the monitor. Builds are fastest around -j8..32; beyond ~32 returns are flat-to-negative.
  • Other build systems, same memory-bounded parallelism:
      cargo build -j "$J" 2>&1 | tee build.log
      cmake --build . -j "$J" 2>&1 | tee build.log
      ninja -j "$J" 2>&1 | tee build.log

OVERLAP THE BUILD WITH INDEPENDENT EXPENSIVE STEPS — a build is long, and other costly steps usually do NOT depend on its output. A dataset/model/asset download needs only the network; an unrelated install needs only the index. Running them AFTER the build wastes wall-clock the budget cannot spare — the download idles behind the compile. Launch the build in the BACKGROUND (background=true) and kick off the dependency-free steps NOW so they run concurrently, then join:
  <build cmd> 2>&1 | tee build.log     # background=true → returns a job handle
  wget/curl <dataset-url> ... 2>&1 | tee fetch.log   (also background=true)
Only steps that truly need the build's output stay ordered. Total wall-clock is the LONGEST branch, not the SUM.

After any build/install: check the exit code before assuming success — a long install (pip, cargo, apt) can fail silently: the log scrolls past the error and the prompt returns, but the package was NOT installed.
  echo "EXIT: $?"                     # right after the command
  which <binary>; <binary> --version  # confirm the artifact exists and runs

Installing multiple dependencies — install CRITICAL packages ONE BY ONE, not all at once. A multi-package install can freeze for minutes with no visible progress: the manager resolves all dependencies together, so one slow/failing package stalls the whole batch. Install each individually, verifying each succeeds before moving on:
  pip install <critical-pkg> 2>&1 | tee install.log; echo "EXIT: $?"
  pip install <next-pkg>    2>&1 | tee install.log; echo "EXIT: $?"
Applies to all package managers (apt, pip, cargo, npm): install the most critical / most likely-to-fail packages first, one at a time.

If this command is short, already streams its own progress, or output is intentionally discarded, override with "_override_reason" explaining why."""


class CompileRedirectGuard(Guard):
    """Pre-launch block on verbose build commands lacking a file redirect."""

    name = "compile_redirect"
    priority = 8  # after backup(5), before safety(10)

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None
        cmd = (ctx.tool_args or {}).get("command", "")
        if not cmd or not isinstance(cmd, str):
            return None
        if not _COMPILE_RE.search(cmd):
            return None
        # Exclude transient invocations: `make --version`, `gcc -v`, `which make`,
        # dry-runs, etc. return instantly with tiny output, so the monitor-
        # visibility concern (a long silent build) does not apply. Only block when
        # a GENUINE build is present.
        if not _has_real_build(cmd):
            return None
        if _streams_to_monitor(cmd):
            return None
        return GuardVerdict.block(
            message=_MESSAGE,
            reason="compile_command_no_redirect",
            category="compile_redirect",
            overridable=True,
        )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def reset_turn(self):
        pass

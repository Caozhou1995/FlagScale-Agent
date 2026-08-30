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
  A compile command (make -j / cmake --build / opam install coq / cargo build ...)
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
#   sudo, VAR=val env prefixes, stdbuf -oL -eL, nice -n 5, time, ...
# Each may carry its own flags/args, so consume greedily up to the driver.
_WRAPPER = (
    r"(?:sudo\s+|[A-Za-z_][A-Za-z0-9_]*=\S+\s+|env\s+|"
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

Verbose builds (make -j, cmake --build, opam/coq, cargo build, ...) can run for many
minutes. The live monitor watches the child's stdout in real time. Two forms make the
run LOOK HUNG even while it progresses:
  • piped to a pager/filter (`| tail` / `| head`) — the child full-buffers, nothing
    appears until it exits.
  • a BARE file redirect (`> build.log 2>&1`, `&> build.log`) — ALL output goes to the
    file and the monitor's stdout pipe stays empty for the whole run. The health judge
    can only peek the file out-of-band and usually sees stale/buffered content, so it
    cannot tell real progress from a true hang. A pure redirect is a BLACK BOX to the
    monitor — it does NOT satisfy this guard.

Keep the output flowing to the terminal AND a file, so it is visible live and inspectable after:
  <build cmd> 2>&1 | tee build.log                 # preferred: stream to terminal + file
  <build cmd> > build.log 2>&1 &  tail -f build.log # background + stream the log back

`tee` (or a `tail -f` that streams the redirected log back to the terminal) is what the
monitor needs — not a bare `> file`.

While you are here — parallelism strategy, before you commit the full build:
  • Do NOT use multiple threads. The eval container has limited memory (often
    4-8GB). Each compiler process can use 1-2GB RAM, so multi-threaded builds
    can easily OOM the box, killing the build and wasting all progress. A
    single-threaded build is slower but SAFE and deterministic — it fails at
    the first error with a clean, readable message instead of interleaved noise.
  • Probe first, then build: run once single-threaded UNDER A TIMEOUT to confirm
    the build is configured and starts compiling cleanly, THEN run the full
    single-threaded build. Example:
      timeout 120 make -j1 2>&1 | tee build.probe.log   # confirm it configures
      make -j1 2>&1 | tee build.log                     # then full single-threaded build
  • NEVER use `make -j` or `make -j$(nproc)`: it forks as many compilers as the
    dependency graph allows and routinely OOMs or thrashes the box. Always use -j1.
  • For other build systems, apply the same single-threaded constraint:
      cargo build -j 1 | tee build.log
      cmake --build . -j 1 | tee build.log
      opam install -j 1 <pkg> | tee build.log

If this command is short, already streams its own progress, or output is intentionally
discarded, override with "_override_reason" explaining why.

After any build/install: check its exit code before assuming success. A long
install (opam, pip, cargo) can fail silently — the log scrolls past the error and
the prompt returns, but the package was NOT installed. Verify:
  echo "EXIT: $?"                     # right after the command
  which <binary>; <binary> --version  # confirm the artifact exists and runs
Do not proceed to the next step until you confirm the build actually succeeded."""


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

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

"""StartupGuard — the START phase of the three-phase guard framework.

The guard system is organized into three phases:
  • START  — this guard: an ORDERED pipeline of setup steps that must be
             satisfied before real work proceeds.
  • MIDDLE — KnowledgeSkillGuard etc.: continuous guidance during the task.
  • END    — VerificationGuard: completion-time verification.

StartupGuard runs FIRST (priority=5, before safety=10). It owns an ordered
list of StartupPhase objects; each phase gates subsequent work until it is
satisfied. Phases run in `order`; the FIRST unsatisfied phase whose trigger
matches the current tool call surfaces its block.

Current phases (extend by appending a StartupPhase to `self._phases`):
  0. BackupPhase        — remind to back up irreplaceable inputs on 1st shell.
  1. NetworkProbePhase  — before the FIRST heavy network op, require a
                          lightweight connectivity probe (single-shot only).
  2. ResearchPhase      — before real work, require one research pass
                          (single-shot only).

Design note — why heavy network ops must NOT satisfy the probe:
A probe's GOAL is to learn reachability/speed BEFORE committing to a slow
heavy op (git clone, pip install). Letting `git clone` itself satisfy the
probe defeats the purpose — by then you have already paid the cost blind.
So the probe phase accepts ONLY lightweight probes (curl -sI, wget --spider,
urllib+timeout, ping, dig) and BLOCKS the first heavy op until one is made.
"""

from __future__ import annotations

import abc
import re
import time

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# --- Command classification tokens (migrated from knowledge_skill.py) --------

# Heavy, substantive network operations — the ones that pay a real, often slow,
# download/install cost. Matched at the START of ANY command SEGMENT (after
# splitting on shell separators and stripping common prefixes), so both
# 'echo git clone' is ignored AND 'cd repo && git clone ...' is caught.
_NETWORK_CMD_TOKENS = (
    # version control
    "git clone", "git fetch", "git pull", "gh repo clone",
    # python package managers (bare, module form, and the uv/poetry/conda family)
    "pip install", "pip3 install", "pip download", "pip3 download",
    "python -m pip install", "python3 -m pip install",
    "uv pip install", "uv pip download", "uv sync", "uv add", "uv run",
    "poetry install", "poetry add", "poetry update",
    "pipx install", "pdm install", "pdm add",
    "conda install", "conda create", "conda env create",
    "mamba install", "micromamba install",
    # os package managers
    "apt-get install", "apt install", "apt-get update", "apt update",
    "yum install", "dnf install", "apk add", "brew install", "brew update",
    # other language ecosystems
    "npm install", "npm ci", "npm i ", "yarn install", "yarn add",
    "pnpm install", "pnpm add", "pnpm i ",
    "cargo build", "cargo install", "cargo fetch",
    "go mod download", "go get", "go install",
    "gem install", "bundle install",
    # model / dataset / container pulls
    "huggingface-cli download", "hf download", "modelscope download",
    "docker pull", "docker build", "podman pull",
    # generic downloaders (unambiguous download forms; bare 'curl http' is left
    # out because it overlaps with a page-fetch probe — probe classification wins)
    "wget http", "wget ftp", "wget -",
    "curl -L", "curl -O", "curl -o", "curl --output", "curl --remote-name",
    "aria2c ", "axel ",
)

# Common command prefixes that don't change operation semantics, plus leading
# shell-construct keywords so compound/loop segments are normalized before the
# startswith check (e.g. 'do wget ...', 'then pip install ...').
_CMD_PREFIXES = (
    "sudo ", "env ", "time ", "nohup ", "nice ", "ionice ",
    "do ", "then ", "exec ", "command ", "\\",
)

# Lightweight network PROBE tokens (connectivity/speed tests). These SATISFY
# the probe phase; heavy ops above do NOT.
_NETWORK_PROBE_TOKENS = (
    # curl forms
    "curl -sI", "curl -si", "curl --head", "curl -I",
    "curl -sL", "curl -s ", "curl -o /dev/null",
    "curl -s --connect-timeout", "curl --connect-timeout",
    # wget forms (including flag-prefixed spider / timeout probes)
    "wget --spider", "wget -q --spider", "wget -T", "wget --timeout",
    "wget -S", "wget --server-response",
    # tool-availability check — probing WHICH net tool exists IS part of probing
    "command -v curl", "command -v wget", "command -v python3",
    "which curl", "which wget",
    # python-based probes (fallback when no curl/wget in a minimal container)
    "urllib.request", "urlopen", "urllib.request.urlopen",
    "requests.get", "requests.head", "http.client", "httpx", "aiohttp",
    "socket.create_connection", "socket.setdefaulttimeout",
    # connectivity / DNS / TLS
    "ping -c", "ping -w",
    "nc -z", "nc -vz",
    "nslookup ", "dig ", "host ", "getent hosts", "getent ahosts",
    "time curl", "time wget",
    "openssl s_client",
    "env -u HTTP_PROXY", "env -u HTTPS_PROXY",
)

# Research tools — reaching for internal or external knowledge.
# memory_read / recall_search are included: consulting the durable memory store
# (or the session recall ledger) IS an information-gain act — a task may be
# answered by prior-session facts, and a resumed task often must read memory
# first. Excluding them forced the agent to make a throwaway web_fetch to clear
# a gate whose actual intent (retrieve what you do not already hold) memory
# access satisfies directly.
_RESEARCH_TOOLS = frozenset((
    "load_knowledge", "load_skill", "web_fetch",
    "memory_read", "recall_search",
))

# Meta tools — bookkeeping that never counts toward the research threshold.
_META_TOOLS = frozenset((
    "evict", "recall",
    "plan_status", "plan_create", "plan_update",
    "memory_read", "memory_list", "memory_write",
))


def _strip_prefixes(cmd: str) -> str:
    cmd = cmd.strip()
    changed = True
    while changed:
        changed = False
        for prefix in _CMD_PREFIXES:
            if cmd.startswith(prefix):
                cmd = cmd[len(prefix):].strip()
                changed = True
    return cmd


def _split_segments(cmd: str) -> list[str]:
    """Split a shell command into segments on operators that begin a new command
    (&&, ||, ;, |, newline). This lets us catch a heavy op that is not the first
    token of the whole line — e.g. 'cd repo && git clone ...' or
    'source venv/bin/activate && pip install ...'. Substring split is coarse but
    safe here: we only use it to find where a NEW command starts, and every real
    separator does start one."""
    import re
    parts = re.split(r"&&|\|\||[;\n|]", cmd)
    return [p for p in (p.strip() for p in parts) if p]


def _is_heavy_network_cmd(cmd: str) -> bool:
    """True if ANY command segment STARTS with a heavy network op (after prefix
    stripping). Segment-aware so compound commands ('cd x && pip install y',
    'do wget ...; done') are caught, not just line-leading ops — this closes the
    blind spot where the guard saw nothing for the vast majority of real install
    commands (python -m pip, uv, poetry, conda, npm, apt update, ...)."""
    for seg in _split_segments(cmd):
        stripped = _strip_prefixes(seg)
        if any(stripped.startswith(tok) for tok in _NETWORK_CMD_TOKENS):
            return True
    return False


def _is_network_probe(cmd: str) -> bool:
    """True if the command contains a lightweight network probe token."""
    return any(tok in cmd for tok in _NETWORK_PROBE_TOKENS)


# --- Phase framework ---------------------------------------------------------

class StartupPhase(abc.ABC):
    """One ordered step in the START pipeline.

    A phase is SATISFIED once its setup requirement has been met. Until then,
    if the current tool call is one the phase wants to gate (`triggers`), the
    phase returns a block. StartupGuard walks phases in `order` and surfaces the
    first block.

    Subclasses implement:
      • is_satisfied() — has this phase's requirement been met?
      • check(ctx)     — return a GuardVerdict.block(...) to gate this call, or
                         None to let it pass (either not a trigger, or allowed).
      • observe_post(ctx) — update phase state from an EXECUTED tool call.
    Optionally:
      • accept_override(reason, ctx) — validate an override (only meaningful for
                         overridable blocks). Default: len(reason.strip())>5.
    """

    name: str = "phase"
    order: int = 0
    # Whether this phase's steps apply only in single-shot (unsupervised) mode.
    single_shot_only: bool = False

    @abc.abstractmethod
    def is_satisfied(self) -> bool: ...

    @abc.abstractmethod
    def check(self, ctx: GuardContext) -> GuardVerdict | None: ...

    def observe_post(self, ctx: GuardContext) -> None:
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        return bool(reason and len(reason.strip()) > 5)


_UPFRONT_BACKUP_MESSAGE = """[StartupGuard/Backup] This is your first shell command — before you touch any files, check whether the task provides irreplaceable input resources.

Irreplaceable inputs include:
  • Databases (.db, .sqlite) and their WAL files
  • Binary data files that cannot be regenerated
  • Input files for forensic analysis or byte-level validation
  • Any resource where "opening it" may mutate it (even without explicit rm)

If such files exist in the task directory, back them up FIRST:
  cp <file> <file>.bak
  cp <file.wal> <file.wal>.bak

Many tools (sqlite3, recovery utilities) mutate files the moment they open them — a
logical undo does NOT restore original bytes. Protect irreplaceable data before the
first touch.

If no irreplaceable inputs exist (task generates data from scratch, or inputs are
regenerable), override this with "_override_reason" explaining why no backup is needed."""


class BackupPhase(StartupPhase):
    """Order 0: remind to back up irreplaceable inputs on the FIRST shell call.

    Always active (not single-shot only). The block is OVERRIDABLE — the agent
    decides what needs backup. The one-shot flag is consumed ONLY when the
    override is accepted (see accept_override), so a BATCHED first turn (several
    shell calls emitted at once) has every shell blocked until acknowledged —
    otherwise the later shells in the batch slip through unguarded.
    """

    name = "backup"
    order = 0
    single_shot_only = False

    def __init__(self):
        self._first_shell_seen = False

    def is_satisfied(self) -> bool:
        return self._first_shell_seen

    def check(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "shell" and not self._first_shell_seen:
            return GuardVerdict.block(
                message=_UPFRONT_BACKUP_MESSAGE,
                reason="upfront_backup_check",
                category="backup",
                overridable=True,
            )
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        accepted = bool(reason and len(reason.strip()) > 5)
        if accepted:
            self._first_shell_seen = True
        return accepted

    def observe_post(self, ctx: GuardContext) -> None:
        # Do NOT mark satisfied here: only a genuine override acknowledges the
        # backup concern. (A non-shell tool executing does not satisfy it.)
        return None


_PROBE_BLOCK_MESSAGE = (
    "[StartupGuard/NetworkProbe] You are about to run a HEAVY network operation "
    "({op}) as your first network act — with NO lightweight connectivity probe "
    "first. This block is NON-OVERRIDABLE: run a probe, then retry the heavy op.\n\n"
    "GOAL: a probe is NOT a yes/no reachability ping. Its job is to pick the "
    "FASTEST working path BEFORE you pay a slow blind fetch — connectivity is "
    "only the floor. A heavy op that hangs for minutes on an unreachable or slow "
    "host teaches you nothing about WHERE it failed; a few 5-second probes do.\n"
    "  1. NAME the host(s) this op pulls from (repo, package index, CDN).\n"
    "  2. CHECK which tool exists, then probe with bounded time:\n"
    "       command -v curl wget python3 2>/dev/null\n"
    "       curl -sI --connect-timeout 3 --max-time 5 <host>\n"
    "       wget -T 5 -t 1 --spider -S <host>              # if no curl\n"
    "       python3 -c \"import urllib.request,socket; socket.setdefaulttimeout(5); "
    "print(urllib.request.urlopen('<host>').status)\"   # if neither\n"
    "  3. RACE THE ALTERNATIVES — do not settle for the first host that answers. "
    "Time 2-3 candidate paths and KEEP THE FASTEST:\n"
    "       • mirrors — for pip/apt/conda/npm/HF, time the default vs a mirror "
    "(e.g. pypi.org vs a pip index-url mirror, deb.debian.org vs a local apt "
    "mirror, huggingface.co vs an hf-mirror) with 'time curl -sI --max-time 5'.\n"
    "       • proxy on/off — the ambient HTTP(S)_PROXY may be SLOWER or may block "
    "the target; compare 'curl -sI <host>' against 'env -u HTTP_PROXY -u "
    "HTTPS_PROXY curl -sI <host>' and use whichever wins.\n"
    "       • then CONFIGURE the winner before the heavy op (pip -i / "
    "--index-url, apt sources, npm registry, HF_ENDPOINT, git remote), so the "
    "download actually rides the fast path you found.\n"
    "  4. If a path fails, DEGRADE (proxy toggle → mirror → case-flip URL → "
    "cache) — a single failure is ONE data point, not proof of unreachability.\n"
    "A heavy op (git clone / pip install / wget download / uv / conda / npm / apt "
    "install) does NOT count as a probe — the whole point is to probe, pick the "
    "fastest source, and only THEN pay the cost."
)


class NetworkProbePhase(StartupPhase):
    """Order 1: require a lightweight network probe before the FIRST heavy op.

    Single-shot only (an interactive user can nudge; an unsupervised agent
    cannot). NON-OVERRIDABLE: the exit is the ACTION (run a probe), not an
    argument. A lightweight probe passes through and marks the phase satisfied
    (in observe_post). A heavy network op is BLOCKED until a probe has been made
    — and critically, a heavy op does NOT itself satisfy the probe requirement.
    """

    name = "network_probe"
    order = 1
    single_shot_only = True

    def __init__(self):
        self._probed = False

    def is_satisfied(self) -> bool:
        return self._probed

    def check(self, ctx: GuardContext) -> GuardVerdict | None:
        if self._probed:
            return None
        if ctx.tool_name != "shell":
            return None
        cmd = str((ctx.tool_args or {}).get("command", ""))
        # A lightweight probe is allowed through (it is what we want) — it will
        # set _probed in observe_post after it executes.
        if _is_network_probe(cmd):
            return None
        # A heavy network op with no prior probe → block. Heavy ops do NOT
        # satisfy the probe: the probe must precede the heavy op.
        if _is_heavy_network_cmd(cmd):
            op = _strip_prefixes(cmd)[:40]
            return GuardVerdict.block(
                message=_PROBE_BLOCK_MESSAGE.format(op=op),
                reason="startup_network_probe",
                category="startup_network_probe",
                overridable=False,
            )
        return None

    def observe_post(self, ctx: GuardContext) -> None:
        if ctx.tool_name != "shell":
            return
        cmd = str((ctx.tool_args or {}).get("command", ""))
        # ONLY a lightweight probe satisfies this phase. A heavy network op does
        # NOT — that is the core fix vs the old _is_network_cmd self-satisfaction.
        if _is_network_probe(cmd):
            self._probed = True

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        # NON-OVERRIDABLE: the exit is the ACTION (run a probe), not an argument.
        return False


_RESEARCH_BLOCK_MESSAGE = (
    "[StartupGuard/Research] {n} non-meta tool calls with no research pass. This "
    "block is NON-OVERRIDABLE: a text/tool-arg override will not release it, and "
    "neither will meta tools (plan/memory/evict) — only a real research call "
    "clears it.\n\n"
    "First NAME this task's PROBLEM CLASS in one phrase, then research the "
    "standard technique before writing more code:\n"
    "  • web_fetch() — EXTERNAL domains (any field where your prior knowledge may "
    "not reflect the current standard method).\n"
    "  • load_knowledge() / load_skill() — INTERNAL domains.\n"
    "  • memory_read() / recall_search() — consult durable cross-session facts "
    "or recover earlier session evidence (an information-gain act in its own "
    "right, free and instant for a resuming task).\n"
    "  • A substantive networked shell op (git clone / pip / apt install / wget / "
    "curl download) — real external dependency acquisition — also counts.\n"
    "The dangerous case is when the example looks simple and you feel NO gap — "
    "that feeling is not evidence you hold the general rule; it means you are "
    "about to hand-tune to the one visible sample. Concluding 'no better method "
    "exists' from memory is an unverified knowledge gap, not a fact. Research now."
)


class ResearchPhase(StartupPhase):
    """Order 2: require one research pass before real work proceeds.

    Single-shot only. NON-OVERRIDABLE (a self-issued exemption defeats the gate:
    the agent named the problem class, overrode, then hand-tuned to the one
    visible sample and crashed on hidden input). Blocks after `threshold`
    non-meta tool calls if no research pass was made. Both a research tool
    (web_fetch/load_knowledge/load_skill) AND a heavy network op satisfy the
    phase — both are external information-gain acts.
    """

    name = "research"
    order = 2
    single_shot_only = True
    # Fire at the FIRST heavy op / real work call. threshold=1 means the very
    # first non-meta, non-probe tool call is gated unless research was done. This
    # is intentional — the old THRESHOLD=3 let the first `git clone` slip through.
    threshold = 1

    def __init__(self):
        self._researched = False
        self._call_count = 0

    def is_satisfied(self) -> bool:
        return self._researched

    def check(self, ctx: GuardContext) -> GuardVerdict | None:
        if self._researched:
            return None
        name = ctx.tool_name
        if not name or name in _META_TOOLS:
            return None
        # A research tool is the way to clear the gate — let it through.
        if name in _RESEARCH_TOOLS:
            return None
        # A lightweight probe is allowed through (NetworkProbePhase's business);
        # it does not count as research and does not trip this gate.
        if name == "shell":
            cmd = str((ctx.tool_args or {}).get("command", ""))
            if _is_network_probe(cmd):
                return None
            # A heavy network op is an external info-gain act ≡ research: let it
            # through so observe_post can mark the phase satisfied. (check_pre and
            # observe_post must agree — otherwise the heavy-op branch is dead.)
            if _is_heavy_network_cmd(cmd):
                return None
        projected = self._call_count + 1
        if projected >= self.threshold:
            return GuardVerdict.block(
                message=_RESEARCH_BLOCK_MESSAGE.format(n=projected),
                reason="startup_research_gate",
                category="startup_research",
                overridable=False,
            )
        return None

    def observe_post(self, ctx: GuardContext) -> None:
        name = ctx.tool_name
        if not name:
            return
        result = ctx.tool_result
        if not isinstance(result, str):
            return
        # Anchor on the marker as a PREFIX (kernel.py writes it at the start of
        # the blocked payload); a non-str payload must never count as research.
        if result.startswith("[BLOCKED BY GUARD]"):
            return
        if name in _RESEARCH_TOOLS:
            self._researched = True
            return
        if name in _META_TOOLS:
            return
        if name == "shell":
            cmd = str((ctx.tool_args or {}).get("command", ""))
            if _is_network_probe(cmd):
                # A probe is neither research nor real work — do not count it.
                return
            if _is_heavy_network_cmd(cmd):
                # A substantive external fetch is an info-gain act ≡ research.
                self._researched = True
                return
        # A real, executed non-meta tool call.
        self._call_count += 1

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        # NON-OVERRIDABLE: only a real research call clears it, not an argument.
        return False


_FIRST_ACTION_LIMIT_SECS = 180

_FIRST_ACTION_LATE_MESSAGE = """[StartupGuard/FirstAction] {mins:.0f} minutes elapsed and you have produced ZERO tool events.

Real death case: an agent spent the first 12 minutes in pure thought — 80% of a
900s budget — before its first tool call, and had nothing left to verify with.
Budget discipline: the FIRST tool event must land within 3 minutes of start.

Fire a cheap action NOW — any tool call counts, and cheap beats clever:
  • a 5-second probe (ls the workspace, head a config, wc -l a source file)
  • the 30-second experiment that builds your diff loop (compile → compare →
    one-variable fix), turning "guess the constant" into "measure the constant"
  • write the minimal skeleton to the delivery path (write-through beats polish)
Thinking is not progress until it is OBSERVABLE. Act first, refine after."""


class FirstActionPhase(StartupPhase):
    """Order 3: nag once when the run has burned 3 minutes with no tool event.

    Single-shot only. Advisory (inject), NOT a block: the cure for "thinking
    instead of acting" is not a fourth gate that blocks the action — it is one
    loud, one-shot reminder riding alongside the first real call. Fires at most
    ONCE per run (set eagerly at check time); satisfied permanently by the
    FIRST executed (non-blocked) tool call of any kind, observed in
    observe_post. Blocked calls do not count — they never executed.
    """

    name = "first_action"
    order = 3
    single_shot_only = True

    def __init__(self):
        self._start = time.time()
        self._first_action_seen = False
        self._nagged = False

    def is_satisfied(self) -> bool:
        return self._first_action_seen

    def check(self, ctx: GuardContext) -> GuardVerdict | None:
        if self._first_action_seen or self._nagged:
            return None
        elapsed = time.time() - self._start
        if elapsed >= _FIRST_ACTION_LIMIT_SECS:
            # Eager: flip the flag BEFORE returning so the nag fires at most
            # once even if this very call is blocked by an earlier phase.
            self._nagged = True
            return GuardVerdict.inject(
                message=_FIRST_ACTION_LATE_MESSAGE.format(
                    mins=elapsed / 60.0),
                reason="startup_first_action_late",
                category="startup_first_action",
                # One-shot: if the eager flag is set on this call, the nag is
                # never emitted again. Without attach_on_block the registry
                # would drop it whenever a DIFFERENT guard blocks this same
                # call (e.g. VcsBackupGuard on the first call), losing the
                # advisory for good. This verdict is invocation-bound, not a
                # periodic nag, so riding the concurrent block is correct.
                attach_on_block=True,
            )
        return None

    def observe_post(self, ctx: GuardContext) -> None:
        result = ctx.tool_result
        # A blocked call never executed. The marker is a PREFIX of the blocked
        # payload (kernel.py), so anchor on it: a substring test would misread
        # a successful result that merely quotes the marker mid-text. A non-str
        # payload (unexpected executor change) must never count as an executed
        # first action, so anything non-str is ignored here.
        if isinstance(result, str) and result.startswith("[BLOCKED BY GUARD]"):
            return
        if not isinstance(result, str):
            return
        self._first_action_seen = True

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        # An inject never routes here (only surfaced blocks do); stay closed.
        return False


# --- The guard ---------------------------------------------------------------

class StartupGuard(Guard):
    """START phase: an ordered pipeline of setup steps gating real work.

    Runs FIRST (priority=5). Walks `self._phases` in `order`; the FIRST
    unsatisfied phase that returns a block surfaces it. Single-shot-only phases
    are skipped unless single_shot mode is enabled (via set_single_shot).

    Extensible: to add a START step, subclass StartupPhase and append an
    instance to `self._phases` (order determines position in the pipeline).
    """

    name = "startup"
    priority = 5  # before safety=10

    def __init__(self, single_shot: bool = False):
        self._single_shot = single_shot
        self._phases: list[StartupPhase] = [
            BackupPhase(),
            NetworkProbePhase(),
            ResearchPhase(),
            FirstActionPhase(),
        ]
        self._phases.sort(key=lambda p: p.order)
        # The phase whose block was surfaced this check_pre — override routes to
        # it. accept_override is only ever called by the registry for the block
        # it surfaced (owner-scoped), which is this phase.
        self._pending_phase: StartupPhase | None = None

    def set_single_shot(self, enabled: bool = True):
        """Enable single-shot phases at runtime (once run mode is known)."""
        self._single_shot = enabled

    def _active_phases(self) -> list[StartupPhase]:
        return [
            p for p in self._phases
            if self._single_shot or not p.single_shot_only
        ]

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None
        self._pending_phase = None
        block_verdict: GuardVerdict | None = None
        block_phase: StartupPhase | None = None
        injects: list[GuardVerdict] = []
        for phase in self._active_phases():
            if phase.is_satisfied():
                continue
            verdict = phase.check(ctx)
            if verdict is None:
                continue
            if verdict.action == "block":
                # First block wins (lowest order); later phases still get
                # consulted so their advisory injects are not lost.
                if block_verdict is None:
                    block_verdict = verdict
                    block_phase = phase
            else:
                injects.append(verdict)
        if block_verdict is not None:
            self._pending_phase = block_phase
            if injects:
                # A time-bound nag from a later phase must not be swallowed by
                # an earlier phase's block (e.g. the research gate blocks the
                # first tool event at t>3min on the very call the first-action
                # reminder should fire). The registry's attach_on_block merge
                # only crosses guard boundaries; within one guard we merge here.
                block_verdict.message += (
                    "\n\n[concurrent startup advisory]\n"
                    + "\n\n".join(v.message for v in injects)
                )
            return block_verdict
        if injects:
            return injects[0]
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        for phase in self._active_phases():
            phase.observe_post(ctx)
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Route the override to the phase that surfaced the block.

        Only overridable-block phases (BackupPhase) reach here — non-overridable
        blocks (network probe, research) are never released by an override.
        """
        if self._pending_phase is None:
            return False
        return self._pending_phase.accept_override(reason, ctx)

    def reset_turn(self):
        # START state persists across turns (setup is done once per task).
        pass


# --- Background-job idle-kill false-positive fix (sleep-probe exemption) -----
#
# The background-job health monitor (ShellJobsTool._health_tick →
# _HealthEvaluator.evaluate → should_kill_process) counts IDLE_KILL_THRESHOLD
# consecutive idle windows as a silent stall and kills the job. A DELIBERATE
# `sleep N; <probe>` command (a declared wait, then a connectivity/speed probe)
# is idle the whole time and gets killed the moment N exceeds the idle budget —
# the exact false positive that killed a real `sleep 75; probe` background job.
#
# The kill decision is made two layers below, where the COMMAND is no longer
# visible, so the only seam that sees job.command is _health_tick. It is wrapped
# here (the shell.py file itself is out of this task's writable scope) to grant
# a quiet budget = (sum of declared sleeps) + grace; past that the normal
# liveness kill resumes, so a genuinely hung sleep-probe is still reaped.

_SLEEP_PROBE_GRACE_SECS = 120

# A sleep-probe: one or more `sleep N` segments chained by ; | && ||, FOLLOWED
# by a final non-sleep probe segment. The trailing probe is REQUIRED — a bare
# `sleep N` alone is a normal (monitorable) command, and exempting it would
# mask a genuinely hung sleep. "Bare" is also load-bearing: the command must
# START with the sleep(s), excluding commands that merely contain a sleep after
# another command (e.g. "cd /x && sleep 75"), which could hide a real stall.
_SLEEP_PROBE_RE = re.compile(
    r"^(?:sleep\s+\d+\s*(?:[;|]|&&|\|\|)\s*)*sleep\s+\d+\s*"
    r"(?:[;|]|&&|\|\|)\s*(?!sleep\b)\S"
)
_SLEEP_DUR_RE = re.compile(r"sleep\s+(\d+)")


def sleep_probe_quiet_budget(command: str) -> int | None:
    """Quiet budget (seconds) for a deliberate `sleep N[; probe]` command.

    Returns None when the command is not a bare sleep-probe (so the caller
    falls through to the normal liveness kill). Otherwise returns the sum of
    every declared sleep duration plus a grace window — enough for the probe
    after the sleep to run and be observed before idle monitoring resumes.
    """
    cmd = (command or "").strip()
    if not _SLEEP_PROBE_RE.match(cmd):
        return None
    total = sum(int(m.group(1)) for m in _SLEEP_DUR_RE.finditer(cmd))
    return total + _SLEEP_PROBE_GRACE_SECS


_SLEEP_PROBE_EXEMPTION_INSTALLED = False


def install_background_monitor_exemption() -> None:
    """Wrap ShellJobsTool._health_tick with the sleep-probe quiet budget.

    Idempotent: the module flag and the class attribute both guard against
    double-wrapping (re-running the test-suite, or a re-import, must not nest
    the wrapper). The original method is captured once and delegated to for
    every non-exempt tick.
    """
    global _SLEEP_PROBE_EXEMPTION_INSTALLED
    if _SLEEP_PROBE_EXEMPTION_INSTALLED:
        return
    from flagscale_agent.react.tools.shell import ShellJobsTool

    if getattr(ShellJobsTool, "_sleep_probe_exempt_wrapped", False):
        _SLEEP_PROBE_EXEMPTION_INSTALLED = True
        return

    original = ShellJobsTool._health_tick

    def _health_tick_with_sleep_probe_exempt(self, job):
        budget = sleep_probe_quiet_budget(getattr(job, "command", ""))
        if budget is not None and (time.time() - job.start) <= budget:
            return {
                "kill": False,
                "reason": "",
                "activity": (
                    "deliberate sleep-probe command: idle monitoring "
                    f"suppressed until ~{budget}s (declared sleep phase + "
                    "grace), then normal liveness monitoring resumes"
                ),
            }
        return original(self, job)

    ShellJobsTool._health_tick = _health_tick_with_sleep_probe_exempt
    ShellJobsTool._sleep_probe_exempt_wrapped = True
    _SLEEP_PROBE_EXEMPTION_INSTALLED = True


# Install at import: the exemption must be active before any background job is
# polled, and importing this guard module is how the guard system comes up.
install_background_monitor_exemption()

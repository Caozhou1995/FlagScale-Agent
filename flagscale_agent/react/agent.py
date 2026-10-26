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

"""FlagScale Agent - ReAct loop with composable Guard/Judge architecture.

No Mixin inheritance. State is owned by Guard instances.
Scene + Profile parameterize behavior without subclassing.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import signal
import sys
import threading
import time
import uuid
import weakref
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import vi_insert_mode, emacs_insert_mode
from prompt_toolkit.styles import Style as PromptStyle

from flagscale_agent.react import display
from flagscale_agent.react.config import AgentConfig
from flagscale_agent.react.history import HistoryManager
from flagscale_agent.react.providers import get_provider
from flagscale_agent.react.retry import retry_with_backoff, _is_context_limit_error
from flagscale_agent.react.session import (
    save_conversation, load_conversation, mark_completed,
    find_resumable_sessions, list_sessions, get_session_dir,
    acquire_session_lock, release_session_lock, get_session_lock_holder,
    SessionLockedError, dir_empty_except_lock, SESSION_LOCK_FILE,
    clean_stale_empty_sessions,
)
from flagscale_agent.react.skills import SkillManager
from flagscale_agent.react.tools import ToolRegistry
from flagscale_agent.react.tools.edit_file import EditFileTool
from flagscale_agent.react.tools.load_knowledge import LoadKnowledgeTool
from flagscale_agent.react.tools.load_skill import LoadSkillTool
from flagscale_agent.react.tools.read_file import ReadFileTool
from flagscale_agent.react.tools.read_image import ReadImageTool
from flagscale_agent.react.tools.shell import ShellTool, ShellJobsTool
from flagscale_agent.react.tools.write_file import WriteFileTool
from flagscale_agent.react.tools.web_fetch import WebFetchTool
# find_log removed - merged into monitor

from flagscale_agent.react.memory import Memory
from flagscale_agent.react.multi_agent.dispatch import DispatchManyTool
from flagscale_agent.react.multi_agent.report_result import ReportResultTool
from flagscale_agent.react.multi_agent.reunite import PollTasksTool
from flagscale_agent.react.multi_agent.resume import ResumeChildTool
from flagscale_agent.react.multi_agent.spawn import SpawnWorkerTool
from flagscale_agent.react.multi_agent.wiring import (
    resolve_worker_query, resolve_resume_query, finalize_worker_if_no_report,
    is_worker, WORKER_ROLE_PREFIX, MULTI_AGENT_NUDGE_SUFFIX,
)
from flagscale_agent.react.tools.memory_write import MemoryWriteTool
from flagscale_agent.react.tools.memory_read import MemoryReadTool
from flagscale_agent.react.tools.memory_list import MemoryListTool
from flagscale_agent.react.proposals import ProposalRegistry
from flagscale_agent.react.tools.proposal import ProposalTool
from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.tools.monitor import FlagScaleTrainMonitorTool
from flagscale_agent.react.tools.plan_create import PlanCreateTool
from flagscale_agent.react.tools.plan_update import PlanUpdateTool
from flagscale_agent.react.tools.plan_status import PlanStatusTool
from flagscale_agent.react.tools.inspect_checkpoint import InspectCheckpointTool
from flagscale_agent.react.tools.evict import EvictTool

from flagscale_agent.react.tools.recall import RecallTool

from flagscale_agent.react.guard.safety import ShellSafetyGuard

from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
from flagscale_agent.react.guard.plan import PlanGuard
from flagscale_agent.react.guard.training_monitor import TrainingMonitorGuard

from flagscale_agent.react.guard.package_search import PackageSearchGuard

from flagscale_agent.react.guard.find_guard import FindGuard
from flagscale_agent.react.guard.shell_jobs_wait import ShellJobsWaitGuard

from flagscale_agent.react.guard.unit_test import UnitTestGuard
from flagscale_agent.react.guard.knowledge_index import KnowledgeIndexGuard
from flagscale_agent.react.guard.post_edit_far_end import PostEditFarEndGuard
from flagscale_agent.react.guard.deliverable_survival import DeliverableSurvivalGuard
from flagscale_agent.react.guard.kill_safety import KillSafetyGuard
from flagscale_agent.react.guard.poll_spacing import PollSpacingGuard
from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
from flagscale_agent.react.guard.memory_post_check import MemoryPostCheckGuard
from flagscale_agent.react.guard.first_e2e_run import FirstE2eRunGuard
from flagscale_agent.react.guard.time_budget import TimeBudgetGuard
from flagscale_agent.react.guard.reviewer_deadline import ReviewerDeadlineGuard
from flagscale_agent.react.guard.post_evict_recovery import PostEvictRecoveryGuard
from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard
from flagscale_agent.react.guard.arg_type import ArgTypeGuard
from flagscale_agent.react.guard.verification import VerificationGuard

from flagscale_agent.react.prompt_builder import PromptBuilder
from flagscale_agent.react.tool_executor import ToolExecutor, tool_display_summary

from flagscale_agent.react.judge import Judge
from flagscale_agent.react.commands import CommandHandler


# NOTE: there is deliberately NO internal default time budget. When the
# FLAGSCALE_AGENT_TIME_BUDGET_SEC env var is unset/unparseable/non-positive, no
# external harness is enforcing a wall, so _task_budget_stats() reports None and
# no budget figure is surfaced to the agent or the health judge. Fabricating a
# large default (e.g. 24h) would falsely signal abundant time and invite
# leisurely work — see _task_budget_stats for the full rationale.


# ── WorkerAgent ──────────────────────────────────────────────────────────────

def _detect_char_collapse(tail: str, min_run: int) -> bool:
    """Char-level collapse inside ONE line of the thinking stream.

    Fourth filler mode (measured, v28 filter-js trial 2026-10-25 16-27-04):
    the stream emitted a single line of 119,373 consecutive '0' bytes —
    122K of the 145KB log was one byte value. The line-window detector is
    structurally blind to this: it only evaluates on completed lines
    (`while "\n" in _intra_buf`), so one newline-free filler run never
    reaches the window. Scan the buffered tail of the CURRENT line for a
    monotone run of one byte >= min_run; such a run cannot be genuine
    reasoning (a proof names its terms; a payload names its delimiter).
    """
    if min_run <= 0 or not tail:
        return False
    if len(tail) < min_run:
        return False
    counts = Counter(tail)
    ch, _cnt = counts.most_common(1)[0]
    return _cnt >= min_run


def _detect_phrase_cycle(tail: str, tail_chars: int, distinct_max: int) -> bool:
    """Single-line filler CYCLE inside the thinking stream.

    Fifth filler mode: short filler phrases repeating with NO newline at all
    ("Go. OK. Let me run. Running. ..." over tens of KB). The line-window
    detector only evaluates completed lines, so a cycle that never emits
    "\\n" never reaches it; _detect_char_collapse needs a monotone run of ONE
    byte, which a multi-phrase cycle never produces. A 4096-char window of a
    real filler cycle repeats a tiny phrase set (<= 12 distinct bytes);
    genuine reasoning always names new terms — calibrated on 3 healthy
    2026-10-25 trials whose worst 4096-char window holds >= 65 distinct
    bytes. 0 disables via tail_chars.
    """
    if tail_chars <= 0 or not tail:
        return False
    if len(tail) < tail_chars:
        return False
    return len(Counter(tail)) <= distinct_max


def _intra_stream_update(buf: str, delta: str, lines: list, window: int,
                         distinct_max: int, min_run: int, tail_chars: int,
                         cycle_distinct_max: int):
    """One incremental update of the intra-stream degeneracy detectors.

    Consumes ONE thinking delta against the pending (newline-free) buffer.
    `lines` is the caller-owned rolling window of completed lines — mutated
    in place, because the line-window predicate needs the ACCUMULATED window
    across deltas (a fresh per-call list never reaches `window` lines and
    the mode-3 check would be dead — measured during integration).

    distinct_max governs the LINE window (mode 3); cycle_distinct_max
    governs the newline-free CYCLE window (mode 5) — two different
    calibrations, never conflate them (conflating them with the line value
    made mode 5 unreachable — measured during integration).

    Returns (buf, fired). fired: 0 none | 1 char-collapse |
    2 single-line phrase-cycle | 3 line-window repeat.
    """
    buf += delta
    keep = 4 * window + max(min_run, tail_chars)

    # Mode 4: monotone one-byte run, newline-free.
    if _detect_char_collapse(buf[-keep:], min_run):
        return buf[-keep:], 1
    # Mode 5: multi-phrase cycle with no newline at all.
    if _detect_phrase_cycle(buf[-tail_chars:], tail_chars,
                            cycle_distinct_max):
        return buf[-keep:], 2

    while "\n" in buf:
        ln, buf = buf.split("\n", 1)
        ln = " ".join(ln.split())[:200]
        if not ln:
            continue
        lines.append(ln)
        if len(lines) > window:
            del lines[:-window]
        if len(lines) >= window and len(set(lines)) <= distinct_max:
            return buf, 3
    # No newline pending: all detectors are tail-based, so trim the pending
    # line to the scan horizon (bounds memory on a long newline-free stream).
    return buf[-keep:], 0


def _run_single_shot_append_nudge(query, worker=False, resume=False):
    """Append the multi-agent nudge to a single-shot USER task.

    Only headless user-task runs get the nudge: worker contracts and parent
    resume messages are parent-authored text and must never be rewritten.
    Idempotent — a query already carrying the nudge is returned unchanged.
    """
    if worker or resume or not query:
        return query
    if MULTI_AGENT_NUDGE_SUFFIX in query:
        return query
    return f"{query}\n\n{MULTI_AGENT_NUDGE_SUFFIX}"


class WorkerAgent:
    """Single agent class with composable Guard/Judge architecture.

    No Mixin inheritance. State that belongs to Guards is owned
    by Guard instances. All infrastructure is composed via __init__.
    """

    def __init__(self, config: AgentConfig,
                 # ── Shared infrastructure (for Orchestrator injection) ──
                 _provider=None, _tool_registry=None, _skill_manager=None,
                 _memory=None, _task_plan=None):
        self.config = config

        # ── Display policy (verbosity control) ──
        # Sync config → display module once; kernel.py and the streaming loop
        # consult display.thinking_enabled()/guard_advisory_enabled() with no
        # config plumbing. Defaults hide thinking + soft advisories (concise
        # output); enabling the policy restores the pre-policy display. Re-run
        # this same sync after a hot config reload (commands.py /reload config).
        self._sync_display_policy()

        # ── Infrastructure ──
        self.skill_manager = _skill_manager or SkillManager(config.skill_dirs)
        self.tool_registry = _tool_registry or ToolRegistry()

        # Knowledge system
        from flagscale_agent.knowledge import KnowledgeManager
        self._knowledge_manager = KnowledgeManager()

        # Nested session home: a worker's own session dir is a CHILD of its
        # parent's — <parent>/subagents/<task_id> — passed down through the spawn
        # env. Precedence: an explicit config.session_dir wins, then the
        # inherited env, then the global default root. Only the session dir
        # nests; memory/proposals stay global (they key on FLAGSCALE_HOME).
        from flagscale_agent.react.paths import get_sessions_root, get_memory_dir, get_proposals_dir
        self._session_id = (os.environ.get("FLAGSCALE_SESSION_ID")
                            or uuid.uuid4().hex[:8])
        sessions_root = (config.session_dir
                         or os.environ.get("FLAGSCALE_SESSION_ROOT")
                         or get_sessions_root())
        session_dir = os.path.join(sessions_root, self._session_id)
        os.makedirs(session_dir, exist_ok=True)
        self._session_dir = session_dir
        self._sessions_root = sessions_root
        # Concurrency guard: a session dir is single-owner state (every save
        # rewrites conversation.json/full.json wholesale, swap_store keys by
        # external index, plans/active.yaml is one pointer). Take the lock
        # BEFORE anything reads or writes the directory; the fresh uuid makes
        # a collision here essentially impossible, but the check is cheap and
        # turns a silent corruption into a loud error if it ever happens.
        try:
            self._session_lock_fd = acquire_session_lock(session_dir)
        except SessionLockedError as e:
            raise RuntimeError(
                f"new session directory is unexpectedly locked "
                f"(existing content at {session_dir}?): {e}"
            ) from e


        # Provider and history must be created before ContextManager
        if not config.api_key:
            raise ValueError(
                "API key not found. Set ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
            )
        self.provider = _provider or get_provider(
            config.provider, config.model, config.api_key,
            config.base_url, config.max_output_tokens,
            thinking_budget=config.thinking_budget,
            temperature=config.temperature,
        )

        self.history = HistoryManager(max_context_tokens=config.max_context_tokens)

        # Swap store for context management V3 (evict/recall)
        from flagscale_agent.react.swap_store import SwapStore
        from flagscale_agent.react.context_manager import ContextManager
        self.context_manager = ContextManager(
            history=self.history,
            swap_store=SwapStore(os.path.join(session_dir, "swap_store")),
        )

        memory_dir = get_memory_dir()
        self.memory = _memory or Memory(memory_dir)

        plan_dir = os.path.join(session_dir, "plans")
        self.task_plan = _task_plan or TaskPlan(plan_dir)

        # Global, cross-session improvement-proposal registry. Lives outside the
        # per-session dir on purpose: an unreviewed proposal raised in one
        # session must resurface at a later session's wrap-up. This is what the
        # wrap-up HARNESS GAP item reads to re-report unanswered proposals.
        self.proposals = ProposalRegistry(get_proposals_dir())

        if not _tool_registry:
            self._register_tools()

        # ── Command handler ──
        self._command_handler = CommandHandler(self)

        # ── Prompt builder ──
        self._prompt_builder = PromptBuilder(self.skill_manager)

        # ── Tool executor ──
        self._tool_executor = ToolExecutor(self)

        # ── Composed components ──
        self.judge = Judge(self.provider)
        self._loaded_skills: set[str] = set()

        self._init_runtime_state()
        atexit.register(self._atexit_hook)
        self._install_signal_handlers()

    def _init_runtime_state(self):
        """Initialize mutable per-session state. Called from __init__.

        Extracted to keep __init__ focused on dependency wiring.
        Can be re-called for tests or worker resets.
        """
        self.turn_count: int = 0
        self._session_input_history: list[str] = []  # All user inputs in this session
        self._interrupted: bool = False
        self._last_tool_calls_deque = deque(maxlen=5)
        self._turn_iteration_count: int = 0

        self._total_iterations: int = 0
        self._original_user_task: str = ""
        self._session_start: float = time.time()
        # Wall-clock start of the CURRENT turn. Re-stamped at the top of every
        # _react_loop so task-budget accounting resets per turn (interactive mode
        # runs many turns over a long-lived session; only the current turn's
        # elapsed time is meaningful against a per-task budget). In single-shot
        # mode a turn == the whole session, so this coincides with _session_start.
        # Initialized here so it is always set even before the first turn.
        self._turn_start: float = self._session_start
        self._session_input_tokens: int = 0
        # Cumulative count of messages evicted this session (dashboard gauge).
        self._evict_count: int = 0
        self._session_output_tokens: int = 0


        self._last_checkpoint_tokens: int = 0
        self._last_tool_call: tuple | None = None
        self._tool_call_cache: dict[tuple, str] = {}

        self._streaming_in_code_block: bool = False
        self._last_compaction_count: int = 0
        self._recent_iters: list[dict] = []

        self._refresh_system_prompt()

        # ── Initialize Kernel ──
        self._kernel = self._build_kernel()

    def _build_kernel(self):
        """Build AgentKernel with injected dependencies."""
        from flagscale_agent.react.kernel import AgentKernel, KernelDeps
        from flagscale_agent.react.guard import GuardRegistry

        guard_registry = GuardRegistry()
        # Register native guards - all guards registered unconditionally
        guard_registry.register(ArgTypeGuard(tool_registry=self.tool_registry))
        # StartupGuard runs FIRST (priority=5) — the START phase of the
        # three-phase framework. Its ordered pipeline gates real work:
        # BackupPhase (back up irreplaceable inputs before the 1st shell) →
        # NetworkProbePhase (lightweight probe before the 1st heavy net op,
        # single-shot only) → ResearchPhase (one research pass before real work,
        # single-shot only). Replaces the standalone BackupGuard.
        from flagscale_agent.react.guard.startup import StartupGuard
        self._startup_guard = StartupGuard()
        guard_registry.register(self._startup_guard)
        from flagscale_agent.react.guard.compile_redirect import CompileRedirectGuard
        guard_registry.register(CompileRedirectGuard())
        guard_registry.register(ShellSafetyGuard())
        guard_registry.register(KillSafetyGuard())
        # VcsBackupGuard: targeted block on destructive git ops (checkout --,
        # reset --hard, clean -f, stash drop/clear, ...) — protects uncommitted
        # work from irreversible loss (e.g. a checkout discarding a patch).
        from flagscale_agent.react.guard.vcs_backup import VcsBackupGuard
        guard_registry.register(VcsBackupGuard())
        # IpPortGuard: post-check advisory when a shell command touches an IP
        # literal or an SSH-family port flag. IPs and ports are a proven
        # hallucination hotspot (two host segments, three SSH port roles) — the
        # inject-only reminder pushes verification against the authoritative
        # memory fact and on-disk hostfile instead of recall. Never blocks.
        from flagscale_agent.react.guard.ip_port import IpPortGuard
        guard_registry.register(IpPortGuard())
        # ResourceProbeGuard: fresh inject on EVERY command that sizes
        # parallelism (nproc/-jN/workers) or launches a background job. Host
        # observation signals (nproc, /proc/meminfo) have repeatedly been
        # mistaken for the slice's actual ceiling inside containers — the
        # structured profile anchors CONSTRAINT signals (cgroup cpu.max /
        # cpuset / memory / pids) with per-field source, and advertises the
        # step-ladder fallback when no constraint is readable. Never blocks.
        from flagscale_agent.react.guard.resource_probe import ResourceProbeGuard
        guard_registry.register(ResourceProbeGuard())
        # LongTimeShellGuard: block long foreground sleep/timeout (>30s) on
        # non-background shell calls. The agent has repeatedly burned minutes
        # on `sleep 180`-style foreground waits instead of backgrounding the
        # job and polling shell_jobs — the guard forces the doctrine at the
        # right layer (block+override), background=true passes freely.
        from flagscale_agent.react.guard.longtimeshell import LongTimeShellGuard
        guard_registry.register(LongTimeShellGuard())
        # SleepSpacingGuard: post-inject on rapid-fire short-sleep probes.
        # LongTimeShellGuard blocks >30s foreground sleeps, which taught the
        # agent to chain `sleep 28; <same probe>` rounds — polling in costume,
        # one full LLM iteration each. This guard pattern-nudges that shape
        # (post-only, never blocks; background waits and single >=60s sleeps
        # stay legitimate).
        from flagscale_agent.react.guard.sleep_spacing import SleepSpacingGuard
        guard_registry.register(SleepSpacingGuard())
        # FirstArtifactGuard: block the next tool call once a run has spent its
        # turn budget with no write to the delivery path. Measured failure: a
        # run produced ~200KB of design reasoning across 4 turns, issued one
        # shell command and ZERO writes, and the verifier's first read was
        # 'no such file'. Every existing text-only nudge stayed unreachable
        # (all three fired 0 times) because that run makes FEW turns each with
        # one tool call — so the distinguishing quantity is "turns consumed
        # before the delivery path was first mutated", which only this guard
        # measures. One-shot and overridable, so a healthy run is never
        # livelocked.
        from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard
        guard_registry.register(FirstArtifactGuard())

        # Reliability guards (P7)

        guard_registry.register(ContextPressureGuard(
            working_window_tokens=self.history.working_window if self.history else 0
        ))
        self._plan_guard = PlanGuard(task_plan=self.task_plan)
        guard_registry.register(self._plan_guard)

        # Plan enforcement guard
        from flagscale_agent.react.guard.plan_update import PlanUpdateGuard
        guard_registry.register(PlanUpdateGuard(task_plan=self.task_plan))
        guard_registry.register(TrainingMonitorGuard())
        guard_registry.register(PackageSearchGuard())
        guard_registry.register(FindGuard())
        guard_registry.register(ShellJobsWaitGuard())

        guard_registry.register(UnitTestGuard())
        # KnowledgeIndexGuard (always active, inject-only): editing a knowledge
        # doc shifts the line numbers cached in indexes/<group>.idx, so
        # load_knowledge would read the wrong lines. Reminds to regenerate the
        # index after such an edit. Mirrors UnitTestGuard. Never blocks.
        guard_registry.register(KnowledgeIndexGuard())
        # PostEditFarEndGuard (always active, inject-only): after EVERY successful
        # write_file/edit_file, remind the agent to verify the FAR end — valid-for-
        # type on the edited file, the consumer's read path, and (for agent source)
        # that the live process still runs old code until /reload. Generic across
        # file types, unlike UnitTestGuard. Never blocks.
        guard_registry.register(PostEditFarEndGuard())
        # DeliverableSurvivalGuard (always active, inject-only): after a
        # destructive command (rm/mv/kill/pkill/git clean/reset --hard/
        # truncate) actually executed, demand a re-verification that the
        # task's deliverables still exist at their delivery paths — a
        # self-kill or an overbroad rm destroys verified state while the
        # agent's belief in it survives. Never
        # blocks.
        guard_registry.register(DeliverableSurvivalGuard())
        # PollSpacingGuard (always active, inject-only): after consecutive
        # poll_tasks/dispatch_many rounds that return not-ready with nothing in
        # between, nudge toward real work between looks (or one bounded sleep)
        # instead of back-to-back bare polling — each empty round is a full LLM
        # iteration spent re-reading the same tokens. Never blocks.
        guard_registry.register(PollSpacingGuard())
        # WriteParamLossGuard (always active, inject-only): when an executed
        # write_file returns the registry's missing-content error, explain the
        # parallel-batch truncation cause and the solo-resend fix — the raw
        # error names the symptom but not the cause. Never blocks.
        from flagscale_agent.react.guard.write_param_loss import WriteParamLossGuard
        guard_registry.register(WriteParamLossGuard())
        # Memory discipline guard (always active)
        guard_registry.register(MemoryDisciplineGuard())
        # Memory post-check guard (always active, inject-only): reconciles the
        # moment a memory_read/write succeeds — the write/read is not "done"
        # until the agent checks for duplicate keys, temp-vs-durable, staleness.
        guard_registry.register(MemoryPostCheckGuard())
        # Time-budget awareness guard: injects escalating wall-clock advisories
        # (50/75/90%) so the agent itself — not just the health judge — reacts to
        # cumulative task time. Silent when no concrete wall was injected.
        guard_registry.register(TimeBudgetGuard(stats_fn=self._task_budget_stats))
        # Progress-order guard: at 25%/50% wall-clock budget spent, block (overridable)
        # once per turn to force a write-through / first-e2e checkpoint.
        guard_registry.register(FirstE2eRunGuard(stats_fn=self._task_budget_stats, task_text_fn=lambda: self._mech_original_task()))
        # Reviewer deadline guard (always active, budget-aware): hard-stops
        # review waiting that would otherwise eat the tail of the enforced
        # wall-clock budget — 2 not-ready polls of the same reviewer-class task
        # near the deadline refuse the next poll (adjudicate from evidence
        # already held); far from the deadline it stays fully silent.
        guard_registry.register(ReviewerDeadlineGuard(stats_fn=self._task_budget_stats))
        # Reviewer feedback guard (completion-time gate): a reviewer-class child
        # this session spawned that is still un-adjudicated blocks text-only
        # [TASK_COMPLETE] — poll_tasks(check) it and fold its findings in first,
        # or override naming the reviewer task_id. Reads the per-session spawn
        # registry (spawn.record_spawn); the global ledger cannot scope to this
        # closing session. See guard/reviewer_feedback.py.
        from flagscale_agent.react.guard.reviewer_feedback import (
            ReviewerFeedbackGuard)
        guard_registry.register(ReviewerFeedbackGuard(
            session_dir_fn=lambda: self._session_dir))
        # Post-evict recovery guard (always active)
        guard_registry.register(PostEvictRecoveryGuard())
        # Knowledge-first guard (always active, inject-only)
        self._knowledge_guard = KnowledgeSkillGuard()
        guard_registry.register(self._knowledge_guard)
        # Verification discipline guard — now keeps ONLY the plan step_done
        # evidence gate; the terminal completion gates moved into the plan's
        # fixed review step (tools/plan_create.py). Re-enabled.
        guard_registry.register(VerificationGuard(plan=self.task_plan, proposals=self.proposals, task_text_fn=lambda: self._mech_original_task() + "\n\n" + self._mech_last_user_turn(), full_log_fn=lambda: list(self.history._full_log), stats_fn=self._task_budget_stats))

        deps = KernelDeps(
            provider=self.provider,
            history=self.history,
            tool_registry=self.tool_registry,
            judge=self.judge,
            guard_registry=guard_registry,
            config=self.config,
            display=display,
            get_schemas_fn=self._get_all_schemas,
            inject_message_fn=self._inject_message,
            append_advisory_fn=self._append_advisory,
            append_tool_results_fn=self._append_tool_results,
            format_tool_result_fn=self.provider.format_tool_result,
            execute_tools_fn=self._execute_tools,
            is_context_limit_error_fn=self._is_context_limit_error,
            call_llm_fn=self._call_llm_stream,
            task_plan=self.task_plan,
            time_budget_stats_fn=self._task_budget_stats,
            on_response_fn=self._on_kernel_response,
            on_tool_results_fn=self._on_kernel_tool_results,
        )
        return AgentKernel(deps)

    # ── Initialization helpers ───────────────────────────────────────────────

    def _register_tools(self):
        # Core file and shell tools
        self.tool_registry.register(ReadFileTool())
        self.tool_registry.register(WriteFileTool())
        self.tool_registry.register(EditFileTool())
        # Vision: let the model LOOK at an image instead of inferring it.
        self.tool_registry.register(ReadImageTool())
        _shell_tool = ShellTool(
            remind_interval=self.config.shell_remind_interval,
            env=self.config.shell_env,
            health_judge_fn=self._health_judge,
            auto_detach_grace_secs=self.config.shell_auto_detach_grace_secs,
        )
        self.tool_registry.register(_shell_tool)
        # Background-job control tool: poll/wait/list/kill jobs started with
        # shell(background=true) or auto-detached from the monitor loop. Wire the
        # SAME health judge + the ShellTool instance's history/resource helpers
        # so a backgrounded job that later hangs is caught by the same liveness
        # logic (and gets the same judge context) the synchronous monitor uses.
        self.tool_registry.register(
            ShellJobsTool(
                health_judge_fn=self._health_judge,
                history_fn=_shell_tool._build_history_str,
                resources_fn=_shell_tool._detect_resources,
            )
        )
        
        # Knowledge and skill tools
        self.tool_registry.register(LoadSkillTool(self.skill_manager))
        self.tool_registry.register(LoadKnowledgeTool(self._knowledge_manager))
        
        # Memory and plan tools
        from flagscale_agent.react.tools.memory_write import MemoryWriteTool
        from flagscale_agent.react.tools.memory_read import MemoryReadTool
        from flagscale_agent.react.tools.memory_list import MemoryListTool
        from flagscale_agent.react.tools.plan_create import PlanCreateTool
        from flagscale_agent.react.tools.plan_update import PlanUpdateTool
        from flagscale_agent.react.tools.plan_status import PlanStatusTool
        from flagscale_agent.react.tools.recall_search import RecallSearchTool
        self.tool_registry.register(MemoryWriteTool(self.memory, self._session_id, task_plan=self.task_plan))
        self.tool_registry.register(MemoryReadTool(self.memory))
        self.tool_registry.register(MemoryListTool(self.memory))
        self.tool_registry.register(ProposalTool(self.proposals, self._session_id))
        self.tool_registry.register(PlanCreateTool(self.task_plan, self._session_id))
        self.tool_registry.register(PlanUpdateTool(self.task_plan))
        self.tool_registry.register(PlanStatusTool(self.task_plan))
        self.tool_registry.register(RecallSearchTool(self._session_dir))
        
        # Web and infrastructure tools
        self.tool_registry.register(WebFetchTool(proxies=self._build_proxies()))
        self.tool_registry.register(FlagScaleTrainMonitorTool(classify_fn=self._judge_confirm))
        self.tool_registry.register(InspectCheckpointTool())
        
        # Context management tools
        self.tool_registry.register(EvictTool())
        self.tool_registry.register(RecallTool())

        # Hard reset - LLM-initiated full context reset
        from flagscale_agent.react.tools.hard_reset import HardResetTool
        self.tool_registry.register(HardResetTool(self))

        # ── Multi-agent: spawn a worker ──────────────────────────────────────
        # spawn_worker is always registered: its OWN execute() refuses when the
        # env carries FLAGSCALE_TASK_ID, so a worker holding the tool can
        # never use it — the refusal is the invariant, not the registry.
        self.tool_registry.register(SpawnWorkerTool(session_dir=self._session_dir))
        # resume_child is the parent<->child dialogue channel: continue a
        # REJECTED/FAILED child (or, by adopting a dead-parent orphan, a
        # descendant) WITH A MESSAGE. Registered unconditionally like spawn —
        # a worker at depth>=1 IS a parent of its own children and must resume
        # them; authorization (caller task id, adoption-when-parent-dead) is the
        # gate, not the registry.
        self.tool_registry.register(ResumeChildTool(session_dir=self._session_dir))
        # report_result is the worker-side ONLY added tool; register it iff we
        # are a worker, so the parent's tool surface does not grow.
        if is_worker():
            self.tool_registry.register(ReportResultTool())
        # ── Multi-agent: reunite / poll ─────────────────────────────────────
        # poll_tasks is the PARENT-side reunite tool (acceptance re-run). Only
        # the parent judges workers, so a worker never gets it.
        if not is_worker():
            self.tool_registry.register(PollTasksTool())
        # ── Multi-agent: parent fan-out ─────────────────────────────────────
        # dispatch_many is the PARENT-side fan-out dispatcher: N workers,
        # bounded concurrency, reunite by POINTER records (never worker text).
        # ASYNC by default: it starts the fan-out on a background thread and
        # returns a dispatch_id at once; the parent polls with action='poll'
        # until state='complete' (wait=true = legacy blocking mode).
        # Parent-only — a worker cannot fan out.
        if not is_worker():
            self.tool_registry.register(
                DispatchManyTool(session_dir=self._session_dir))

    def _build_proxies(self) -> dict[str, str]:
        proxies = {}
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            val = os.environ.get(var)
            if val:
                proxies[var.lower()] = val
        return proxies

    def _sync_display_policy(self):
        """Push the display verbosity policy from config into the display module.

        kernel.py and the streaming loop read display.thinking_enabled() /
        display.guard_advisory_enabled() with no config plumbing, so the policy
        lives in the display module. Called at construction AND after a hot
        config reload (commands.py /reload config), so a config edit takes
        effect without a process restart.
        """
        from flagscale_agent.react import display as _display_mod
        _display_mod.set_policy(
            show_thinking=self.config.show_thinking,
            show_guard_advisory=self.config.show_guard_advisory,
            show_guard_block=self.config.show_guard_block,
        )

    # ── System prompt ────────────────────────────────────────────────────────

    def _refresh_system_prompt(self, memory_context: str = "",
                               plan_context: str = "", bump_turn: bool = True):
        tool_names = [t.name for t in self.tool_registry.all_tools()]
        # Feed the runtime gauges to the dashboard before each rebuild.
        try:
            self._prompt_builder.runtime_stats = {
                "evict_count": self._evict_count,
                "budget": self._task_budget_stats(),
                # Pure stopwatch for THIS turn (seconds since _turn_start), shown
                # when no external deadline exists. It is NOT a budget: the
                # dashboard renders it as "elapsed" only, and the budget guards
                # keep reading _task_budget_stats (None -> silent) — so this never
                # fabricates a deadline nor wakes a guard.
                "turn_elapsed": max(0.0, time.time() - self._turn_start),
            }
        except Exception:
            pass  # dashboard degrades to existing lines, never crashes the turn
        self._prompt_builder.refresh(
            history=self.history,
            active_skill_content={},
            shared_storage_paths=getattr(self, "_shared_storage_paths", []),
            memory_context=memory_context,
            plan_context=plan_context,
            tool_names=tool_names,
            session_dir=self._session_dir,
            bump_turn=bump_turn,
        )

    # ── Health judge (delegates to unified Judge) ───────────────────────────

    def _health_judge(self, command: str, recent_output: str, elapsed: str,
                      output_changed: bool = True, stall_count: int = 0,
                      activity: str = "", command_history: str = "",
                      container_resources: str = "", health_advisory: str = "",
                      **_ignored) -> dict:
        # The shell monitor loop forwards richer context (command_history,
        # container_resources, health_advisory). Accept and forward the fields
        # judge.health understands; **_ignored absorbs any future field the
        # loop adds without breaking this call (a mismatch here silently
        # disables the whole LLM judge, since the bounded worker swallows the
        # TypeError and returns None).
        expectation = self._current_expectation_anchor()
        return self.judge.health(
            command, recent_output, elapsed, output_changed, stall_count,
            expectation=expectation, activity=activity,
            command_history=command_history,
            container_resources=container_resources,
            task_budget=self._task_budget_summary(),
        )

    def _task_budget_stats(self) -> dict | None:
        """Resolve the per-turn wall-clock budget as structured numbers.

        Returns a dict {elapsed, budget, remaining, pct} (all floats/seconds)
        ONLY when a concrete per-turn wall exists, from either source (config
        field time_budget_sec, e.g. via --time-budget-sec, takes precedence over
        the env var FLAGSCALE_AGENT_TIME_BUDGET_SEC that an external harness
        injects). Returns None otherwise.

        Design: the tb-adapter always exports this env to the value it actually
        enforces via asyncio.wait_for (the real harbor wall, or its own concrete
        fallback). The ONLY case where the env is unset is a standalone /
        interactive run with NO external time enforcement — there, a real
        deadline does not exist, so we must NOT fabricate one. In particular we
        deliberately do NOT fabricate a large code-uniformity default (e.g.
        24h): reporting that as a "budget" would tell the agent (and the health
        judge) it has abundant time and invite it to work leisurely. Unset /
        unparseable / non-positive env => None => no budget
        reporting anywhere (the health prompt stays byte-identical to the
        no-budget version, and the TimeBudgetGuard stays silent).

        Elapsed is measured from the CURRENT turn's start (self._turn_start,
        re-stamped each _react_loop), not from session start: an interactive
        session may span many turns over hours, and only the active turn's time
        is meaningful against a per-task budget. In single-shot mode a turn is
        the whole session, so the two coincide.
        """
        # Config field (e.g. via --time-budget-sec) takes precedence; fall back
        # to the env var that an external harness injects. Either source carries
        # the same meaning: a per-turn wall-clock budget that drives time
        # warnings + wrap-up (it is NOT a hard kill on its own).
        budget = None
        cfg_budget = getattr(self.config, "time_budget_sec", 0.0) or 0.0
        if cfg_budget and cfg_budget > 0:
            budget = float(cfg_budget)
        else:
            raw = os.environ.get("FLAGSCALE_AGENT_TIME_BUDGET_SEC", "").strip()
            if not raw:
                # No external harness enforcing a wall -> no real deadline exists.
                return None
            try:
                budget = float(raw)
            except (TypeError, ValueError):
                # Unparseable -> treat as no injected wall rather than fabricating
                # the 24h default.
                return None
        if budget <= 0:
            # Explicit 0 / negative disables budget reporting.
            return None
        elapsed = max(0.0, time.time() - self._turn_start)
        remaining = budget - elapsed
        pct = (elapsed / budget) * 100.0 if budget else 0.0
        # Read the kernel's LIVE per-turn iteration count: _turn_iteration_count
        # is only assigned after run_turn() returns, so inside the turn it is
        # always 0 and a consumer would silently fall back (the same inert-mechanism
        # trap as a never-accumulating counter). Fall back to the post-turn value
        # when no kernel is wired (unit tests).
        rounds = 0
        _kernel = getattr(self, "_kernel", None)
        if _kernel is not None and hasattr(_kernel, "iterations_live"):
            rounds = int(getattr(_kernel, "iterations_live", 0) or 0)
        if rounds <= 0:
            rounds = int(getattr(self, "_turn_iteration_count", 0) or 0)
        return {
            "elapsed": elapsed,
            "budget": budget,
            "remaining": remaining,
            "pct": pct,
            # Rounds spent so far THIS turn. Consumers derive the turn's own
            # measured cost-per-action (elapsed/rounds) instead of assuming one:
            # a percentage alone cannot say how many actions are left, and the
            # measured harness spends very different wall per round across models.
            "rounds": rounds,
        }

    def _task_budget_summary(self) -> str:
        """Free-text summary of cumulative wall-clock vs the injected budget.

        Returns "" when no concrete wall was injected (see _task_budget_stats),
        so the health prompt stays byte-identical to the no-budget version and
        no 24h-default figure ever leaks to the judge. When a real wall exists,
        reports elapsed-since-turn-start, the total, percent used and remaining.
        """
        stats = self._task_budget_stats()
        if stats is None:
            return ""

        def _fmt(sec: float) -> str:
            sec = int(max(0.0, sec))
            m, s = divmod(sec, 60)
            h, m = divmod(m, 60)
            if h:
                return f"{h}h{m:02d}m"
            return f"{m}m{s:02d}s"

        return (
            f"cumulative task time elapsed {_fmt(stats['elapsed'])} of total budget "
            f"{_fmt(stats['budget'])} ({stats['pct']:.0f}% used, "
            f"~{_fmt(stats['remaining'])} remaining)"
        )

    def _current_expectation_anchor(self) -> str:
        """Assemble an expectation anchor from the active plan's current step.

        The anchor is whatever the agent declared for the step it is executing —
        its title, scratchpad notes, and acceptance criteria. This is where the
        agent states intent BEFORE acting, so the health judge can test the live
        run against it. Returns "" when there is no active plan or no in-progress
        step, which keeps health monitoring in its generic no-anchor mode.
        """
        try:
            plan = self.task_plan.get_active()
        except Exception:
            return ""
        if not plan:
            return ""
        steps = plan.get("steps") or []
        # Prefer the step currently marked "doing"; fall back to the first
        # pending step so the anchor still reflects imminent intent.
        current = next((s for s in steps if s.get("status") == "doing"), None)
        if current is None:
            current = next((s for s in steps if s.get("status") == "pending"), None)
        if current is None:
            return ""
        parts = []
        title = (current.get("title") or "").strip()
        if title:
            parts.append(f"Current step: {title}")
        notes = (current.get("notes") or "").strip()
        if notes:
            parts.append(f"Notes: {notes}")
        acceptance = current.get("acceptance") or []
        if acceptance:
            joined = "; ".join(str(a).strip() for a in acceptance if str(a).strip())
            if joined:
                parts.append(f"Acceptance: {joined}")
        return "\n".join(parts)

    def _judge_confirm(self, category: str, matched_text: str, context: str = "") -> bool:
        return self.judge.classify(category, {"text": matched_text, "context": context}, default=True)

    # ── Atexit ──────────────────────────────────────────────────────────────

    def _atexit_hook(self):
        try:
            self._save_conversation(completed=False)
        except Exception:
            pass

    def _install_signal_handlers(self):
        """Persist the trajectory on termination signals.

        atexit does NOT run on SIGTERM (default disposition terminates the
        process without stack unwinding, so no atexit and no finally:) nor on
        SIGKILL. Harbor / Terminal-Bench enforces timeouts by sending SIGTERM
        first, then SIGKILL after a grace period. SIGKILL is uncatchable, but
        SIGTERM is — installing a handler lets us flush conversation/memory to
        disk before the process dies, covering the common timeout case.

        signal.signal() only works on the main thread; in worker threads or
        embedded contexts it raises ValueError, which we swallow (atexit +
        the single-shot finally: still apply there).
        """
        def _handler(signum, frame):
            try:
                self._save_conversation(completed=False)
            except Exception:
                pass
            # Reap any still-running background shell jobs so a timeout kill
            # does not leave orphaned build/train processes behind.
            try:
                from flagscale_agent.react.tools.shell import _JOB_REGISTRY
                _JOB_REGISTRY.cleanup_all()
            except Exception:
                pass
            # Restore the default disposition and re-raise so the process
            # exits with the correct signal status instead of swallowing the
            # termination request.
            try:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            except Exception:
                # Last resort: exit with the conventional 128+signum code.
                os._exit(128 + signum)

        for _sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(_sig, _handler)
            except (ValueError, OSError):
                # Not on the main thread, or signal unsupported on this
                # platform — rely on atexit / finally fallbacks instead.
                pass

    def _save_conversation(self, completed: bool = False, session_summary: str = None):
        if not self.history.messages:
            return
        # Keep the resume-list preview in sync with the actual conversation.
        # If no explicit summary is provided, regenerate one from the current
        # input history on EVERY save — otherwise a summary written by an old
        # crash/exception save stays frozen on disk forever (session.py
        # preserves any existing session_summary), so the resume list shows
        # stale turns while turn_count keeps advancing.
        if session_summary is None and self._session_input_history:
            session_summary = self._generate_session_summary()
        save_conversation(
            self._session_dir, self._session_id,
            self.history.messages,
            loaded_skills=list(self._loaded_skills),
            completed=completed,
            session_summary=session_summary,
            session_input_tokens=self._session_input_tokens,
            session_output_tokens=self._session_output_tokens,
            turn_count=self.turn_count,
            session_input_history=self._session_input_history,
        )
        # Save full (pre-eviction) conversation alongside
        self._save_conversation_full()

    def _save_conversation_full(self):
        """Save the full pre-eviction conversation to conversation_full.json.

        Uses _full_log from HistoryManager which captures all messages
        at append-time before any eviction modifies them.
        """
        import tempfile
        full_log = self.history._full_log
        if not full_log:
            return
        path = os.path.join(self._session_dir, "conversation_full.json")
        data = {
            "session_id": self._session_id,
            "messages": full_log,
            "index_offset": self.history._index_offset,
            "reset_count": self.history._reset_count,
            "turn_count": self.turn_count,
            "session_input_history": self._session_input_history,
        }
        # Atomic write
        try:
            fd, tmp = tempfile.mkstemp(
                dir=self._session_dir, prefix=".tmp_conv_full_", suffix=".json"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _auto_save(self):
        """Auto-save conversation after each completed turn.

        Silent - no user-facing output. Failures are swallowed to avoid
        disrupting the interactive flow.
        """
        try:
            self._save_conversation(completed=False)
        except Exception:
            pass

    def _generate_session_summary(self) -> str:
        """Generate session summary by showing first 2 and last 2 user inputs.
        
        Uses _session_input_history (pure user inputs, no system injections).
        
        Format:
        [1] <first input, truncated to 80 chars>
        [2] <second input, truncated to 80 chars>
        ...
        [N-1] <second-to-last input, truncated to 80 chars>
        [N] <last input, truncated to 80 chars>
        """
        try:
            user_inputs = self._session_input_history
            if not user_inputs:
                return "(no user input)"
            
            def truncate_text(text: str, max_len=80):
                if len(text) > max_len:
                    return text[:max_len] + "..."
                return text
            
            lines = []
            total = len(user_inputs)
            
            if total <= 4:
                for i, text in enumerate(user_inputs, 1):
                    lines.append(f"[{i}] {truncate_text(text)}")
            else:
                for i in range(2):
                    lines.append(f"[{i+1}] {truncate_text(user_inputs[i])}")
                lines.append("...")
                for i in range(total-2, total):
                    lines.append(f"[{i+1}] {truncate_text(user_inputs[i])}")
            
            return "\n".join(lines)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[Session Summary] Failed: {e}")
            return "(summary generation failed)"

    def _generate_hard_reset_summary(self) -> str:
        """Call LLM to generate a work-state summary for hard reset continuation.

        Uses all current messages as context. The summary should capture:
        - What task is being worked on
        - Key progress and decisions made
        - Current step / what to do next
        - Critical paths, values, configs discovered

        Returns empty string on failure (fallback to programmatic summary).
        """
        from flagscale_agent.react import display
        
        try:
            # Use current messages as context for the summary
            # Strip internal fields (like _ext_idx) that APIs do not accept
            messages = []
            for msg in self.history.messages:
                clean = {k: v for k, v in msg.items() if not k.startswith("_")}
                messages.append(clean)

            # Also include plan status if available
            plan_info = ""
            try:
                from flagscale_agent.react.tools.plan import PlanManager
                pm = PlanManager()
                status = pm.get_status()
                if status:
                    plan_info = f"\nCurrent plan:\n{status}"
            except Exception:
                pass  # Plan fetch failure is non-critical

            prompt_msgs = [
                {"role": "user", "content": (
                    "You are generating a context continuation summary. The conversation "
                    "is being compacted to free up context space. Summarize the FULL work "
                    "state in a structured format that will allow seamless continuation.\n\n"
                    "Include:\n"
                    "1. TASK: What is being worked on (one paragraph). State the task's "
                    "hard constraints VERBATIM (exact version, tool, format, method the task "
                    "named) — these are the task's identity and must survive compaction intact.\n"
                    "2. PROGRESS: Key steps completed, decisions made, paths/values discovered\n"
                    "3. CURRENT STATE: What was just happening, any pending operations\n"
                    "4. NEXT STEPS: What to do next\n"
                    "5. CRITICAL CONTEXT: File paths, configs, error messages, any values "
                    "that would be lost without this summary\n"
                    "6. CONSTRAINTS & RULED-OUT APPROACHES: Every approach ALREADY TRIED AND "
                    "REJECTED and WHY (e.g. 'downgrading to v3.7 is NOT acceptable — task "
                    "requires v2.2', 'config X deadlocks', 'path Y is a dead end'). This is the "
                    "MOST easily lost and MOST damaging to lose: after compaction you will "
                    "re-infer the task from leftover files and risk reverting to a degraded "
                    "approach you already proved wrong. Preserve these negative constraints "
                    "explicitly — a ruled-out approach stays ruled out after compaction.\n\n"
                    f"{plan_info}\n\n"
                    "Be comprehensive but concise. This summary replaces the full conversation "
                    "history. Output the summary directly, no preamble."
                )}
            ]

            # Prepend the current conversation as context
            context_msgs = list(messages) + prompt_msgs
            
            # Call LLM with error handling
            # chat_stream returns Iterator[Dict] yielding events directly
            stream = self.provider.chat_stream(context_msgs, [])
            if stream is None:
                display.warn("[Hard Reset Summary] chat_stream returned None")
                return ""
            
            result_text = ""
            for event in stream:
                if isinstance(event, dict):
                    if event.get("type") == "text":
                        result_text += event.get("content", "")
            
            result_text = result_text.strip()
            if not result_text:
                display.warn("[Hard Reset Summary] LLM returned empty result")
            return result_text
            
        except TypeError as e:
            display.warn(f"[Hard Reset Summary] Type error: {e}")
            return ""
        except Exception as e:
            display.warn(f"[Hard Reset Summary] Unexpected error: {type(e).__name__}: {e}")
            return ""

    def _build_programmatic_summary(self) -> str:
        """Build a programmatic fallback summary when LLM call fails."""
        parts = ["[Context Hard Reset - conversation auto-compacted]"]

        # Plan status
        try:
            from flagscale_agent.react.tools.plan import PlanManager
            pm = PlanManager()
            status = pm.get_status()
            if status:
                parts.append(f"\n## Current Plan\n{status}")
        except Exception:
            pass

        # Session metadata
        parts.append(f"\nSession dir: {self._session_dir}")
        parts.append(f"Reset count: {self.history._reset_count}")
        parts.append(f"Full conversation log: {self._session_dir}/conversation_full.json")
        parts.append("\nUse read_file on conversation_full.json or memory_list() for more context.")
        parts.append(
            "\n[!] CONSTRAINTS WARNING: This compaction may have dropped the task's hard "
            "constraints and the approaches you already ruled out. Before acting, recover them "
            "from plan notes and memory (memory_read/memory_list) — do NOT re-infer the task "
            "from leftover files in the workdir and revert to a degraded approach you already "
            "rejected. A ruled-out approach stays ruled out."
        )

        return "\n".join(parts)

    # ── Mechanical state injection for hard reset ─────────────────────────────
    # A hard reset otherwise relies on the MODEL's own summary prose, which is
    # unverified and can silently drop the task's identity. These helpers
    # re-inject the load-bearing state VERBATIM (not a model paraphrase) so the
    # fresh window cannot lose: (1) the original task text, (1b) the most
    # recent real user turn (the CURRENT working directive - a session spans
    # many turns and each turn's instruction governs its own stretch of work),
    # (2) the active plan, (3) the memory keys touched this session. Each
    # degrades INDEPENDENTLY — one failing never aborts the reset. Sizes are
    # capped so the block cannot defeat the point of the reset.
    _MECH_TASK_CAP = 4000
    _MECH_PLAN_CAP = 3000
    _MECH_MEMORY_MAX_KEYS = 40
    _MECH_LAST_TURN_CAP = 2000
    _MECH_LAST_TURN_DEPTH = 5

    # Intra-stream degenerate-repeat detection (see _call_llm_stream). A single
    # response can emit an unbounded cycle of filler lines inside its THINKING
    # channel, which the kernel's per-response window cannot see. Calibrated on
    # a real trial: the degenerate region shows 6 distinct lines in a 16-line
    # window (longest 18 chars) while genuine reasoning shows 16 distinct in the
    # same window (longest 583 chars). The bound is 8, not 6, so that a balanced
    # 8-phrase cycle (the same dead loop with more filler variety) is caught too
    # without reaching genuine reasoning's 16. 0 disables.
    INTRA_STREAM_REPEAT_WINDOW = 16
    INTRA_STREAM_REPEAT_DISTINCT_MOST = 8

    # In-stream wall-clock cap for a PURE-thinking stream. Measured on the
    # GLM Anthropic-compat endpoint (~4.6 tok/s served): a 16384-token token
    # cap corresponds to ~2200-3500s of wall clock, so no TOKEN-based cap can
    # ever fire inside a 900s budget — a single slow stream eats 45% of the
    # turn (v12.1 trial o3NAGa9: one thinking stream burned 405.8s of 781.7s
    # and the turn ended honest-BLOCKED with only a placeholder). The cap is
    # TIME, not tokens: abort the stream once it has been pure thinking for
    # this many seconds and no visible output has appeared. 0 disables.
    IN_STREAM_THINKING_WALL_SEC = 240

    # Char-collapse floor (see _detect_char_collapse in this module): the
    # measured v28 trial produced ONE line of 119,373 consecutive '0' bytes
    # (145KB log, 122K zeros) — filler WITHOUT newlines, which the line-window
    # above never evaluates (its `while "\n" in _intra_buf` gate stays closed).
    # The tail buffer is scanned after every thinking delta; a monotone run of
    # one byte this long cannot be genuine reasoning (a proof sketch names its
    # terms; a payload names its delimiter). 0 disables.
    INTRA_STREAM_CHAR_COLLAPSE_MIN_RUN = 1024

    # Single-line phrase-cycle detection (fifth filler mode): a newline-free
    # cycle of short filler phrases (measured: the v28 filter-js stream also
    # ran long single-line loops). The line-window above only evaluates
    # completed lines, so a cycle with no newline is invisible to it. The
    # calibration probe (/tmp/cycle_probe.py over 3 healthy trials from
    # 2026-10-25, whole log incl. tool echo — a SUPERSET of the thinking-only
    # stream this scans) bottoms out at 65 distinct chars in any 4096-char
    # window; an English filler cycle repeats a tiny phrase set (a 4-6 phrase
    # loop holds ~19-30 distinct bytes). The bound 32 keeps 2x margin under
    # the healthy floor and catches any realistic phrase cycle. 0 disables.
    INTRA_STREAM_CYCLE_TAIL_CHARS = 4096
    INTRA_STREAM_CYCLE_DISTINCT_MAX = 32

    def _mech_original_task(self) -> str:
        """First user turn of the session = the original task text, verbatim.

        Known limit (inherent): a real FIRST task that quotes the exact
        harness continuation header shape is indistinguishable from a prior
        continuation and will be skipped here.
        """
        try:
            for msg in self.history._full_log:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content", "")
                if content is None:
                    continue
                if isinstance(content, list):
                    parts = []
                    for b in content:
                        if isinstance(b, dict):
                            parts.append(str(b.get("text", "") or b.get("content", "")))
                        else:
                            parts.append(str(b))
                    text = "".join(parts)
                else:
                    text = str(content)
                text = text.strip()
                if not text:
                    continue
                # Skip ONLY a harness-injected continuation — identified by the
                # exact header _build_continuation_message emits:
                #   "[Context Hard Reset #<n> - conversation auto-compacted]".
                # A real task that merely STARTS with "[Context Hard Reset" (e.g. it
                # quotes an earlier reset paste) must NOT be dropped, and a task that
                # itself begins with the marker must still be returned.
                if (text.startswith("[Context Hard Reset #")
                        and " - conversation auto-compacted]" in text[:120]):
                    continue
                if len(text) > self._MECH_TASK_CAP:
                    text = text[: self._MECH_TASK_CAP] + "\n...[truncated]"
                return text
            return "(unavailable: no user turn found in full log)"
        except Exception as e:
            return f"(unavailable: {type(e).__name__})"

    def _mech_last_user_turn(self) -> str:
        """Last (up to 5) REAL user turns = the CURRENT directive chain, verbatim.

        A session spans many turns and each turn's instruction governs its own
        stretch of work; the current working directive is often a SHORT CHAIN
        of recent turns ("run the tests" -> "not right, check the log" ->
        "commit the fix"), so only the last turn is not enough. Returns up to
        _MECH_LAST_TURN_DEPTH=5 turns as an oldest->newest timeline with [-N]
        markers; [-1] is the CURRENT directive. A single turn renders as plain
        text (no [-1] marker) for backward compatibility.

        Primary source: _session_input_history — only genuine user inputs enter
        it (slash commands are filtered before append; guard/tool-result/
        continuation messages never enter). Fallback (single-shot/worker paths
        where that list is empty): backward scan of _full_log skipping
        tool-result user messages and harness continuation headers. Known
        limits, both inherent and disclosed: (a) guard block/escalate
        injections are bare unmarked user messages and cannot be excluded
        here; (b) a real user task that QUOTES the exact harness continuation
        header shape is indistinguishable from a genuine prior continuation
        and will be skipped in this fallback (the primary source does not
        have this problem — only genuine inputs enter it).
        """
        try:
            texts = []  # type: list
            inputs = getattr(self, "_session_input_history", None)
            if inputs:
                for raw in inputs[-self._MECH_LAST_TURN_DEPTH:]:
                    text = str(raw).strip()
                    if text:
                        texts.append(text)
            else:
                from flagscale_agent.react.history import _is_tool_result
                for msg in reversed(self.history._full_log):
                    if msg.get("role") != "user":
                        continue
                    content = msg.get("content", "")
                    if content is None:
                        continue
                    if _is_tool_result(msg):
                        continue
                    if isinstance(content, list):
                        parts = []
                        for b in content:
                            if isinstance(b, dict):
                                parts.append(str(b.get("text", "") or b.get("content", "")))
                            else:
                                parts.append(str(b))
                        text = "".join(parts)
                    else:
                        text = str(content)
                    text = text.strip()
                    if not text:
                        continue
                    if (text.startswith("[Context Hard Reset #")
                            and " - conversation auto-compacted]" in text[:120]):
                        continue
                    texts.append(text)
                    if len(texts) >= self._MECH_LAST_TURN_DEPTH:
                        break
                # fallback scanned newest->oldest; normalize to oldest->newest
                # so both sources share one timeline order.
                texts.reverse()
            if not texts:
                return "(unavailable: no real user turn found in full log)"
            for i, t in enumerate(texts):
                if len(t) > self._MECH_LAST_TURN_CAP:
                    texts[i] = t[: self._MECH_LAST_TURN_CAP] + "\n...[truncated]"
            if len(texts) == 1:
                return texts[0]
            numbered = []
            n_total = len(texts)
            for idx, t in enumerate(texts):
                n = idx - n_total  # -n_total, ... -2, -1
                marker = "[-1] (CURRENT directive)" if n == -1 else f"[{n}]"
                numbered.append(f"{marker} {t}")
            return "\n".join(numbered)
        except Exception as e:
            return f"(unavailable: {type(e).__name__})"

    def _mech_active_plan(self) -> str:
        """The active plan rendered verbatim (statuses + acceptance + notes)."""
        try:
            plan = self.task_plan.get_active()
            if not plan:
                return "(no active plan)"
            text = self.task_plan._format_plan(plan)
            if len(text) > self._MECH_PLAN_CAP:
                text = text[: self._MECH_PLAN_CAP] + "\n...[truncated]"
            return text
        except Exception as e:
            return f"(unavailable: {type(e).__name__})"

    def _mech_memory_keys(self) -> str:
        """Keys of memory entries written or updated in THIS session.

        Keys only (not content) — the content survives the reset on disk; the
        agent re-reads what it needs with memory_read(key=...).
        """
        try:
            entries = self.memory.list_entries()
            sid = getattr(self, "_session_id", "")
            keys = sorted({
                e.get("key", "")
                for e in entries
                if e.get("key") and sid and (
                    e.get("created_session") == sid
                    or e.get("updated_session") == sid
                )
            })
            if not keys:
                return "(no memory keys written this session)"
            shown = keys[: self._MECH_MEMORY_MAX_KEYS]
            lines = [f"- {k}" for k in shown]
            if len(keys) > len(shown):
                lines.append(f"- ...and {len(keys) - len(shown)} more (memory_list())")
            return "\n".join(lines)
        except Exception as e:
            return f"(unavailable: {type(e).__name__})"

    def _build_mechanical_state_block(self) -> str:
        """Assemble the verbatim mechanical-state block for the continuation.

        Returns "" on total failure — the reset must proceed regardless.
        """
        try:
            return (
                "## MECHANICAL STATE - harness-injected, VERBATIM (not a model summary)\n"
                "Re-injected by the harness on every hard reset and authoritative: the\n"
                "model summary below is SUPPLEMENTARY and may be stale or lossy. Re-read\n"
                "these before acting.\n\n"
                "### (1) ORIGINAL TASK - first user turn, verbatim\n"
                f"{self._mech_original_task()}\n\n"
                "### (1b) LAST 5 USER TURNS - current directive chain, oldest->newest\n"
                "    ([-1] = CURRENT directive; older ones may be superseded by it)\n"
                f"{self._mech_last_user_turn()}\n\n"
                "### (2) ACTIVE PLAN - verbatim\n"
                f"{self._mech_active_plan()}\n\n"
                "### (3) MEMORY KEYS WRITTEN THIS SESSION - read with memory_read(key=...);\n"
                "    their content survives the reset even though the conversation did not\n"
                f"{self._mech_memory_keys()}"
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"[Hard Reset] mechanical state block failed: {e}")
            return ""

    def _build_continuation_message(self, summary: str) -> str:
        """Build the full continuation message to inject after hard reset."""
        # This builder runs BEFORE history.hard_reset() increments _reset_count,
        # so the reset being built is (#current + 1). Label it with the actual
        # reset number so the header agrees with the persisted reset_count.
        reset_count = self.history._reset_count + 1
        total_messages = len(self.history._full_log)

        header = (
            f"[Context Hard Reset #{reset_count} - conversation auto-compacted]\n"
            f"Previous conversation: {total_messages} messages total, "
            f"saved in conversation_full.json.\n"
            f"Session: {self._session_dir}\n"
        )

        footer = (
            "\n---\n"
            "If you need more historical context, use:\n"
            "- plan_status() for current task state\n"
            "- memory_list() for cross-session knowledge\n"
            "- read_file on conversation_full.json for full history\n"
            "- recall(index=N) for specific messages by index"
        )

        # Mechanical verbatim block comes FIRST (authoritative); the model
        # summary is supplementary. Empty block degrades to the old format.
        mech = self._build_mechanical_state_block()
        mech_part = f"{mech}\n\n" if mech else ""
        return f"{header}\n{mech_part}{summary}\n{footer}"

    def _hard_reset_context(self):
        """Execute hard reset: generate summary, clear context, rebuild.

        This is called when should_hard_reset() is True. Uses the 40%
        remaining headroom to call LLM for summary before clearing.
        """
        from flagscale_agent.react import display

        display.warn("[Hard Reset] Context pressure high, compacting conversation...")

        # 1. Generate LLM summary (we have headroom since working_window < max)
        summary = self._generate_hard_reset_summary()
        if not summary:
            display.warn("[Hard Reset] LLM summary failed, using programmatic fallback.")
            summary = self._build_programmatic_summary()

        # 2. Build continuation message
        continuation = self._build_continuation_message(summary)

        # 3. Execute hard reset on history manager
        stats = self.history.hard_reset(continuation, preserve_last_n=4)

        # 4. Save conversation state (persist full_log)
        self._save_conversation_full()
        
        # 5. Notify guards that recovery happened
        if hasattr(self, 'kernel') and self.kernel and self.kernel.deps.guard_registry:
            for guard in self.kernel.deps.guard_registry.guards:
                if hasattr(guard, 'notify_recovery'):
                    guard.notify_recovery()

        display.warn(
            f"[Hard Reset] Done. Cleared {stats['cleared_count']} messages, "
            f"kept last {stats['preserved_count']}. Reset #{stats['reset_count']}."
        )

    def _exit(self):
        display.goodbye()
        # Generate session summary before saving
        summary = self._generate_session_summary()
        self._save_conversation(completed=False, session_summary=summary)
        # Reap any still-running background shell jobs before exiting.
        try:
            from flagscale_agent.react.tools.shell import _JOB_REGISTRY
            _JOB_REGISTRY.cleanup_all()
        except Exception:
            pass
        sys.exit(0)

    # ── Main entry ──────────────────────────────────────────────────────────

    def run(self, single_shot_query: str | None = None):
        if single_shot_query:
            self._run_single_shot(single_shot_query)
            return

        # Check for --auto-resume from /reload command
        auto_resume_id = None
        for arg in sys.argv:
            if arg.startswith("--auto-resume="):
                auto_resume_id = arg.split("=", 1)[1]
                break

        extra = self._startup_hints()
        display.banner(self.config.provider, self.config.model,
                       context_window=self.config.max_context_tokens, extra_lines=extra)
        self._check_proxy()

        if auto_resume_id:
            # Auto-resume after /reload - find and restore the session
            self._auto_resume(auto_resume_id)
        else:
            self._check_resume()

        from flagscale_agent.react.paths import get_input_history_file
        history_file = get_input_history_file()
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        completer = WordCompleter(
            ["/quit", "/reload", "/resume", "/session"],
            sentence=True,
        )
        # Key bindings: Enter submits, but pasted newlines are preserved
        kb = KeyBindings()

        @kb.add("enter", filter=~vi_insert_mode & ~emacs_insert_mode)
        @kb.add("enter", filter=vi_insert_mode | emacs_insert_mode)
        def _submit(event):
            """Enter always submits (even in multiline mode)."""
            event.current_buffer.validate_and_handle()

        def _build_session():
            return PromptSession(
                history=FileHistory(history_file),
                completer=completer,
                multiline=True,
                key_bindings=kb,
                style=PromptStyle.from_dict({
                    "prompt": "#87d787 bold",
                    "": "#e4e4e4",
                }),
            )

        session = _build_session()

        # ── Interactive input watchdog ──
        # prompt_toolkit can occasionally wedge on its input fd: the process
        # stays alive and the pane keeps rendering, but keystrokes are never
        # consumed (observed as a main thread parked in ep_poll while stdin
        # has pending, unread bytes). guard_state["at_prompt"] is True only
        # while blocked in session.prompt(), so a long model/tool turn can
        # never trip the watchdog. On a confirmed wedge it escalates to
        # SIGINT; watchdog_state["tripped"] then tells the except below to
        # rebuild the prompt session instead of treating it as a user exit.
        from flagscale_agent.react.prompt_watchdog import (
            PromptWatchdog,
            foreign_tty_readers,
            reap_tty_readers,
        )
        guard_state = {"at_prompt": False}
        watchdog_state = {"tripped": False}

        def _on_watchdog_sigint():
            watchdog_state["tripped"] = True

        def _tty_reader_probe() -> bool:
            """Third wedge class: a foreign process is draining our input.

            ``shell`` launches every child with ``stdin=DEVNULL``, so a healthy
            background job never holds the tty. A process still reading our
            terminal (e.g. a leaked ``head -1``) steals keystrokes before
            prompt_toolkit sees them — invisible to both ``select`` and the
            userspace backlog. Report it so the watchdog can reap it.
            """
            return bool(foreign_tty_readers())

        def _on_tty_reader():
            # Remove the cause, not the symptom: a SIGINT to *us* leaves the
            # thief alive and still eating keys. Kill the thief instead.
            reap_tty_readers(logger=display.warn)

        def _pending_input_backlog() -> bool:
            """Keys read off the tty but not yet dispatched by prompt_toolkit.

            The classic wedge leaves bytes unread in the kernel (caught by
            ``select``). A second wedge class has the reader *consume* the
            keystroke and then stall before dispatching it, so ``select`` on the
            fd sees nothing while the key sits in prompt_toolkit's userspace
            queues. Consult both ``KeyProcessor.input_queue`` (fed but not
            processed) and ``Vt100Input._buffer`` (parsed but not fed).
            """
            app = getattr(session, "app", None)
            if app is None:
                return False
            kp = getattr(app, "key_processor", None)
            if kp is not None and getattr(kp, "input_queue", None):
                return True
            inp = getattr(app, "input", None)
            buf = getattr(inp, "_buffer", None) if inp is not None else None
            return bool(buf)

        watchdog = PromptWatchdog(
            is_at_prompt=lambda: guard_state["at_prompt"],
            on_sigint=_on_watchdog_sigint,
            pending_probe=_pending_input_backlog,
            tty_reader_probe=_tty_reader_probe,
            on_tty_reader=_on_tty_reader,
            logger=display.warn,
        )
        watchdog.start()

        while True:
            guard_state["at_prompt"] = True
            try:
                user_input = session.prompt([("class:prompt", "> ")]).strip()
            except (EOFError, KeyboardInterrupt):
                guard_state["at_prompt"] = False
                if watchdog_state["tripped"]:
                    # A wedged input loop was broken by the watchdog: rebuild a
                    # fresh prompt session and keep going rather than exiting.
                    watchdog_state["tripped"] = False
                    display.warn("input loop was stuck — prompt session rebuilt")
                    try:
                        session = _build_session()
                    except Exception:
                        pass
                    continue
                self._exit()
                break
            except BaseException:
                guard_state["at_prompt"] = False
                raise
            guard_state["at_prompt"] = False

            if not user_input:
                continue

            # Collapse multi-line pasted input display
            if "\n" in user_input:
                display.pasted_input(user_input)

            if self._command_handler.handle_slash_command(user_input):
                continue

            self._inject_context()
            self._session_input_history.append(user_input)
            self.history.append({"role": "user", "content": user_input})
            try:
                self._react_loop()
            except KeyboardInterrupt:
                display.interrupted()
                self._interrupted = True
                self._auto_save()
                continue

            # Auto-save after each complete user turn
            self._auto_save()


    def _run_single_shot(self, query: str):
        # Resume mode (multi-agent dialogue channel): the parent is continuing
        # THIS worker WITH A MESSAGE. We re-entered the worker's EXISTING session
        # dir (same id, inherited via env — so this whole object already loaded
        # that history) and the parent's message is the next user turn. The child
        # keeps its history/state and carries on in-context.
        resume_msg = resolve_resume_query()
        if resume_msg is not None:
            if getattr(self, "_plan_guard", None) is not None:
                self._plan_guard.set_single_shot(True)
            if getattr(self, "_startup_guard", None) is not None:
                self._startup_guard.set_single_shot(True)
            self._restore_for_resume()
            self._inject_context()
            self.history.append({"role": "user", "content": resume_msg})
            try:
                self._react_loop()
            except Exception:
                display.warn("WorkerAgent._run_single_shot() react loop failed")
            finally:
                self._finalize_single_shot()
            return

        # Unsupervised run: the plan (with acceptance/verification) stands in
        # for the absent human supervisor, so PlanGuard enforces it (block
        # after an observation budget) rather than merely reminding.
        if getattr(self, "_plan_guard", None) is not None:
            self._plan_guard.set_single_shot(True)
        # Same rationale: no human to nudge toward setup discipline, so the
        # StartupGuard enables its single-shot phases (NetworkProbePhase requiring
        # a lightweight probe before the first heavy net op, and ResearchPhase
        # requiring one research pass before real work).
        if getattr(self, "_startup_guard", None) is not None:
            self._startup_guard.set_single_shot(True)
        self._inject_context()
        # Worker mode: the CLI positional arg is a contract PATH; resolve
        # to the full contract text + role prefix before the loop sees it.
        query = resolve_worker_query(query)
        # Headless user-task runs (not a resumed worker, not a worker contract)
        # get a one-line nudge appended to the user's own task text: the task
        # should be tackled in multi-agent mode, fanning out concurrent workers
        # for efficiency and completion quality. Workers and resumed workers are
        # excluded — their contract/resume text is parent-authored and must not
        # be rewritten.
        query = _run_single_shot_append_nudge(
            query, worker=is_worker(), resume=resolve_resume_query() is not None)
        self.history.append({"role": "user", "content": query})
        try:
            self._react_loop()
        except Exception:
            display.warn("WorkerAgent._run_single_shot() react loop failed")
        finally:
            self._finalize_single_shot()

    def _finalize_single_shot(self):
        # Headless single-shot has no per-turn REPL save (unlike the
        # interactive loop), so persist explicitly here to guarantee a
        # normally-completed run is durable. A harness timeout that kills
        # the process mid-run is handled separately by the SIGTERM handler
        # installed in _install_signal_handlers().
        self._auto_save()
        # Worker that exits without calling report_result would leave the
        # ledger stuck in RUNNING; close it now.
        try:
            finalize_worker_if_no_report()
        except Exception:
            pass
        try:
            from flagscale_agent.react.tools.shell import _JOB_REGISTRY
            _JOB_REGISTRY.cleanup_all()
        except Exception:
            pass

    def _restore_for_resume(self):
        """Re-enter THIS worker's existing nested session in-context.

        Called at the top of a resumed single-shot run. The session id/dir came
        in via env (FLAGSCALE_SESSION_ROOT/ID), so __init__ already bound and
        locked the existing dir; here we load the persisted conversation into
        history so the resumed turn continues the SAME context, not a fresh one.
        """
        conv_path = os.path.join(self._session_dir, "conversation.json")
        if not os.path.isfile(conv_path):
            return  # nothing to restore — first touch of this dir
        try:
            with open(conv_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            display.warn(f"[resume] could not load prior session: {e}")
            return
        # Reuse the same loader the interactive /reload path uses, minus the
        # directory rebinding (we are already bound to the right dir).
        self._load_session_data(data)
        # The resumed run's node is already bound; keep the inherited id.
        self._session_id = data.get("session_id", self._session_id)

    def _load_session_data(self, data: dict):
        """Load persisted conversation messages + counters into this agent.

        Extracted from _restore_session so both interactive resume/reload and
        the worker's resume path share one loader. The caller is responsible for
        having bound self._session_dir correctly before calling.
        """
        # Restore _full_log from conversation_full.json if it exists (preserves
        # the complete audit trail across /reload and hard resets).
        session_dir = self._session_dir
        full_log_path = os.path.join(session_dir, "conversation_full.json")
        full_log_seeded = False
        if os.path.isfile(full_log_path):
            try:
                with open(full_log_path, "r", encoding="utf-8") as f:
                    full_data_on_disk = json.load(f)
                full_msgs = full_data_on_disk.get("messages", [])
                if full_msgs:
                    import copy
                    self.history._full_log = [copy.deepcopy(m) for m in full_msgs]
                    self.history._index_offset = full_data_on_disk.get("index_offset", 0)
                    self.history._reset_count = full_data_on_disk.get("reset_count", 0)
                    full_log_seeded = True
            except Exception:
                pass

        messages = data.get("messages", [])
        # Skip the old system prompt - we already have a fresh one from __init__
        for msg in messages:
            if msg.get("role") == "system":
                continue
            if full_log_seeded:
                self.history._messages.append(msg)
                msg["_ext_idx"] = msg.get("_ext_idx", len(self.history._full_log))
            else:
                self.history.append(msg)
        self._session_input_history = data.get("session_input_history", [])
        self.turn_count = data.get("turn_count", len(self._session_input_history))
        loaded = data.get("loaded_skills", [])
        self._session_input_tokens = data.get("session_input_tokens", 0)
        self._session_output_tokens = data.get("session_output_tokens", 0)
        for skill_name in loaded:
            try:
                content = self.skill_manager.load(skill_name)
                if content:
                    self._loaded_skills.add(skill_name)
            except Exception:
                pass
        self._refresh_system_prompt()

    def _restore_session(self, data: dict, session_dir: str):
        """Restore a previous session - take over its session_id and dir."""
        # Concurrency guard: refuse to bind a directory another live agent
        # process holds. This must happen BEFORE any state is re-pointed.
        holder = get_session_lock_holder(session_dir)
        if holder:
            print(display.red(
                f"[resume] REFUSED: session directory is held by another live agent "
                f"process (PID {holder.get('pid')}, started {holder.get('start', '?')}: "
                f"{holder.get('cmd', '?')}). Two processes writing the same session "
                f"would corrupt each other's history. Resume a different session or "
                f"stop the holder first."
            ))
            raise SessionLockedError(session_dir, holder)

        # Rebinding: release the old directory's lock only after the new one
        # is secured. Same-dir rebind (already-bound directory re-restored)
        # and cross-dir rebind both end holding exactly one lock.
        old_session_dir = self._session_dir
        old_lock_fd = getattr(self, "_session_lock_fd", None)
        new_lock_fd = acquire_session_lock(session_dir)  # raises SessionLockedError if taken meanwhile
        self._session_id = data.get("session_id", self._session_id)
        self._session_dir = session_dir
        if new_lock_fd is not old_lock_fd:
            if old_lock_fd is not None:
                release_session_lock(old_lock_fd)
            self._session_lock_fd = new_lock_fd

        # Re-point plan to old session's dirs
        self.task_plan._dir = os.path.join(session_dir, "plans")
        # Re-point swap store to restored session's dir
        from flagscale_agent.react.swap_store import SwapStore
        from flagscale_agent.react.context_manager import ContextManager
        self.context_manager = ContextManager(
            history=self.history,
            swap_store=SwapStore(os.path.join(session_dir, "swap_store")),
        )

        # Re-point recall_search to the restored session's log. The tool captured
        # self._session_dir at registration time (the fresh, empty dir __init__
        # created); after a resume/reload rebind it must search the RESTORED
        # directory's conversation_full.json, not the abandoned one. Registry
        # register() overwrites by name, so re-registering replaces the instance.
        from flagscale_agent.react.tools.recall_search import RecallSearchTool
        self.tool_registry.register(RecallSearchTool(session_dir))

        # Same capture-at-registration problem for tools that captured
        # self._session_id: restore has just replaced _session_id, but the old
        # instances would keep tagging memory entries / plans with the
        # abandoned fresh session id. Re-register under the restored id.
        from flagscale_agent.react.tools.memory_write import MemoryWriteTool
        from flagscale_agent.react.tools.plan_create import PlanCreateTool
        self.tool_registry.register(MemoryWriteTool(
            self.memory, self._session_id, task_plan=self.task_plan))
        self.tool_registry.register(PlanCreateTool(self.task_plan, self._session_id))
        # Re-register the proposal tool under the restored session id (same
        # capture-at-registration issue as memory_write/plan_create above).
        self.tool_registry.register(ProposalTool(self.proposals, self._session_id))

        # Same capture-at-registration issue for the multi-agent tools: they
        # captured self._session_dir at __init__ (the fresh uuid dir) and pass
        # it as FLAGSCALE_SESSION_ROOT/FLAGSCALE_SESSION_ID through _build_env,
        # so spawned workers nest under <captured_dir>/subagents/<task_id>.
        # After a resume/reload the captured dir is the ABANDONED fresh dir, so
        # workers would nest under a stale sibling session directory instead of
        # the restored one. Re-register under the restored dir; spawn/resume are
        # always present, dispatch_many only for a non-worker (mirrors __init__).
        self.tool_registry.register(SpawnWorkerTool(session_dir=session_dir))
        self.tool_registry.register(ResumeChildTool(session_dir=session_dir))
        if not is_worker():
            self.tool_registry.register(DispatchManyTool(session_dir=session_dir))

        # Clean up the empty new session dir if it's different. The emptiness
        # predicate must ignore dotfiles (the lock file lives there) AND empty
        # subdirectories (SwapStore/TaskPlan __init__ makedirs their dirs), else
        # a fresh dir that only ever held .session.lock + swap_store/ is never
        # removed and empty shells accumulate across reloads.
        if old_session_dir != session_dir and dir_empty_except_lock(old_session_dir):
            try:
                import shutil
                shutil.rmtree(old_session_dir, ignore_errors=True)
            except Exception:
                pass

        # Opportunistic sweep: every resume/reload is a natural point to reap
        # stale empty shells (abandoned __init__ dirs from past reloads) across
        # the whole sessions root. Three-condition guard inside: empty-except-
        # lock + no live lock holder + mtime older than CLEAN_MIN_AGE_SEC.
        try:
            clean_stale_empty_sessions(self._sessions_root)
        except Exception:
            pass

        # Load persisted conversation + counters via the shared loader (the
        # worker resume path uses the same loader so the two can't diverge).
        self._load_session_data(data)

    def _check_resume(self):
        sessions = find_resumable_sessions(self._sessions_root)
        if not sessions:
            return
        resumable = [s for s in sessions if s.get("user_turns", 0) >= 1]
        if not resumable:
            return

        # Check how many need summary generation
        missing_summary = [s for s in resumable if not s.get("session_summary")]
        if missing_summary:
            print(display.dim(f"\n[resume] Generating summaries for {len(missing_summary)} session(s)..."))
            self._generate_missing_summaries(missing_summary)

        print(display.yellow(f"\n[resume] {len(resumable)} resumable session(s):"))
        for i, s in enumerate(resumable, 1):
            sid = s.get("session_id", "?")[:8]
            ts = time.strftime("%m-%d %H:%M", time.localtime(s.get("timestamp", 0)))
            turns = s.get("user_turns", 0)
            summary = s.get("session_summary", "")
            if not summary:
                summary = "(summary unavailable)"
            print(display.dim(f"  {i}. {sid}  {ts} ({turns} turns):"))
            for line in summary.strip().split("\n"):
                print(display.dim(f"     {line}"))
        print(display.dim("Type: resume <number> or resume <session_id>"))

    def _generate_missing_summaries(self, sessions: list):
        """Generate simple summaries for sessions, reading fresh from conversation file each time.
        
        This method is called on every resume/reload to generate real-time summaries.
        Uses first 2 + ... + last 2 user messages. No LLM calls, no caching.
        """
        import json as _json
        import logging
        logger = logging.getLogger(__name__)
        
        for s in sessions:
            session_dir = s.get("session_dir", "")
            if not session_dir:
                continue
            
            # Always regenerate from file (real-time refresh, no cache)
            conv_path = os.path.join(session_dir, "conversation.json")
            if not os.path.isfile(conv_path):
                continue
            
            try:
                with open(conv_path, "r", encoding="utf-8") as f:
                    conv_data = _json.load(f)
                
                user_inputs = conv_data.get("session_input_history", [])
                
                if not user_inputs:
                    s["session_summary"] = "(no user input)"
                    continue
                
                # Use same truncation logic as _generate_session_summary
                def truncate_text(text: str, max_len=80):
                    if len(text) > max_len:
                        return text[:max_len] + "..."
                    return text
                
                lines = []
                total = len(user_inputs)
                
                if total <= 4:
                    for i, text in enumerate(user_inputs, 1):
                        lines.append(f"[{i}] {truncate_text(text)}")
                else:
                    # First 2
                    for i in range(2):
                        lines.append(f"[{i+1}] {truncate_text(user_inputs[i])}")
                    lines.append("...")
                    # Last 2
                    for i in range(total-2, total):
                        lines.append(f"[{i+1}] {truncate_text(user_inputs[i])}")
                
                summary = "\n".join(lines)
                s["session_summary"] = summary
                
            except Exception as e:
                logger.warning(f"[Missing Summary] Failed for {session_dir}: {e}")
                continue

    def _auto_resume(self, session_id: str):
        """Auto-resume a session after /reload (process restart).

        Finds the session by ID, restores it, and prints a confirmation.
        """
        import json
        sessions = find_resumable_sessions(self._sessions_root)
        target = None
        for s in sessions:
            if s.get("session_id", "").startswith(session_id):
                target = s
                break

        if not target:
            print(display.yellow(f"[reload] Session {session_id} not found, starting fresh."))
            return

        session_dir = get_session_dir(target["session_id"])
        # Concurrency guard before reading state: if another live agent holds
        # this directory, refuse. Continuing here would fork the history.
        holder = get_session_lock_holder(session_dir)
        if holder:
            err = SessionLockedError(session_dir, holder)
            print(display.red(f"\n[reload] REFUSED: {err}\n"
                              f"[reload] Stop the other agent process first, then retry /reload.\n"))
            sys.exit(1)
        # Load full conversation data (find_resumable_sessions only returns metadata)
        conv_path = os.path.join(session_dir, "conversation.json")
        try:
            with open(conv_path, "r", encoding="utf-8") as f:
                full_data = json.load(f)
        except Exception as e:
            print(display.yellow(f"[reload] Failed to load session data: {e}"))
            return

        self._restore_session(full_data, session_dir)
        print(display.yellow(
            f"\n[reload] Code reloaded successfully. Session {session_id[:8]} restored "
            f"({self.turn_count} turns, {len(self.history.messages)} messages)."
        ))
        print(display.dim("All code changes are now active.\n"))

    # ── Context injection ───────────────────────────────────────────────────

    def _inject_context(self):
        # Memory is no longer injected into system prompt (accessed via tools).
        # Plan context is only used for the dashboard line at the end.
        plan_context = self._build_plan_context()
        self._refresh_system_prompt(plan_context=plan_context)

    def _build_plan_context(self) -> str:
        active = self.task_plan.get_active()
        if not active:
            return ""
        steps = active.get("steps", [])
        icons = {"pending": "⬜", "doing": "🔄", "done": "✅", "skipped": "⏭", "blocked": "🚫"}
        lines = [f'<active-plan title="{active.get("title", "")}">']
        # Plan-level problem model (the agent's CURRENT hypothesis). Rendered in
        # full — never truncated — so the dashboard can carry it permanently.
        # Absent/empty thinking omits the block entirely (byte-identical to the
        # pre-hypothesis behavior).
        thinking = (active.get("thinking") or "").strip()
        if thinking:
            lines.append("<current-hypothesis>")
            lines.append(thinking)
            lines.append("</current-hypothesis>")
        current_step = None
        for s in steps:
            icon = icons.get(s.get("status", "pending"), "?")
            title = s.get("title", "") or s.get("description", "")
            notes = s.get("notes", "")
            line = f"  [{icon}] Step {s.get('id', '?')}: {title}"
            if notes:
                line += f"\n      notes: {notes}"
            lines.append(line)
            if s.get("status") in ("pending", "doing") and current_step is None:
                current_step = s
        lines.append("</active-plan>")
        # Add explicit advancement instruction
        if current_step:
            step_id = current_step.get("id", "?")
            lines.append(
                f"\n→ Focus on Step {step_id}. When done, call: "
                f"plan_update(action='step_done', step_id={step_id})"
            )
        return "\n".join(lines)

    # ── Startup ─────────────────────────────────────────────────────────────

    def _startup_hints(self) -> list[str]:
        hints = []
        sessions = find_resumable_sessions(self._sessions_root)
        if sessions:
            hints.append(f"{len(sessions)} resumable session(s) - use /resume to restore")
        # Surface unreviewed harness-improvement proposals at startup (not in the
        # dashboard) so a long-pending one is visible before the next wrap-up.
        try:
            n_open = self.proposals.open_count()
        except Exception:
            n_open = 0
        if n_open:
            hints.append(
                f"{n_open} open improvement proposal(s) awaiting review - "
                "will be re-reported at wrap-up"
            )
        return hints

    def _check_proxy(self):
        proxy_vars = [v for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY") if os.environ.get(v)]
        if proxy_vars:
            print(display.dim(f"Proxy detected: {', '.join(proxy_vars)}"))

    # ── Auto-continue ──────────────────────────────────────────────────────

    def _get_last_assistant_text(self) -> str:
        for msg in reversed(self.history.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    return "".join(texts)
        return ""

    # ── User path confirmation ────────────────────────────────────────────

    # ── React loop ──────────────────────────────────────────────────────────

    def _react_loop(self):
        """Kernel-based react loop."""
        self.turn_count += 1
        self._turn_start = time.time()
        self._interrupted = False
        self._turn_iteration_count = 0
        self._context_pressure_warned = False

        result = self._kernel.run_turn()

        self._interrupted = result.interrupted
        self._session_input_tokens += result.input_tokens
        self._session_output_tokens += result.output_tokens
        self._turn_iteration_count = result.iterations
        display.turn_summary(self.turn_count, result.elapsed, result.input_tokens, result.output_tokens,
                             session_input_tokens=self._session_input_tokens,
                             session_output_tokens=self._session_output_tokens)

    def _on_kernel_response(self, response: dict):
        """Called by Kernel after LLM response is appended to history."""
        pass

    def _on_kernel_tool_results(self, tool_calls: list, results: list):
        """Called by Kernel after tool execution and guard checks."""
        # Track tool calls
        for tc in tool_calls:
            self._last_tool_calls_deque.append(tc["name"])
        self._total_iterations += 1

        # Count evicted messages (dashboard gauge: shows the agent how much
        # context it has already swapped out this session).
        for tc in tool_calls:
            if tc["name"] == "evict":
                n = (tc.get("arguments") or {}).get("indexes") or []
                try:
                    self._evict_count += len(n)
                except TypeError:
                    pass

        # Rebuild the system prompt ONLY on plan-tool usage (turn boundaries
        # already refresh it via _inject_context). The dashboard lives at the
        # tail of the system prompt, AFTER the provider's cache breakpoint on
        # the static section — an intra-turn rebuild changes those bytes on
        # every request and invalidates the entire cached conversation prefix
        # (observed: cache_read collapsing to the static section alone).
        # Degrade silently on failure: a dashboard hiccup must never crash
        # the turn.
        try:
            if any(tc["name"] in ("plan_create", "plan_update", "plan_status")
                   for tc in tool_calls):
                plan_context = self._build_plan_context()
                self._refresh_system_prompt(plan_context=plan_context)
        except Exception:
            pass

        # Context pressure warning is handled by ContextPressureGuard - no duplicate check here

        self._tool_call_cache = {}
        print()



    def _inject_message(self, msg: str):
        """Inject a guard block/escalate message into conversation history.

        v4: This is now ONLY called for block/escalate verdicts.
        Soft inject goes through _append_advisory instead.
        """
        self.history.append({"role": "user", "content": msg})

    def _append_advisory(self, msg: str):
        """Append a soft guard advisory to the last tool_result in history.

        v4: Instead of creating an independent user message (which pollutes
        conversation history and hijacks LLM attention), advisory messages are
        appended to the most recent tool_result. This makes them:
        - Lower priority (tool output, not user instruction)
        - Non-persistent (scroll out with tool results naturally)
        - Non-confusing (LLM treats as supplementary info, not a command)

        Handles both OpenAI format (role=tool, content=str) and
        Anthropic format (role=user, content=[{type: tool_result, ...}]).
        Falls back to a lightweight user message if no tool_result exists.
        """
        # NOTE: Do NOT call display.guard_inject(msg) here - the caller
        # (kernel.py _handle_guard_verdict) already displays it. Calling here
        # would produce duplicate display on terminal.
        
        advisory_suffix = (
            f"\n\n---\n"
            f"[Guard Advisory - note but do not respond to this, prioritize tool results and user requests]\n"
            f"{msg}"
        )
        messages = self.history.get_messages()

        # Search backwards for the most recent tool_result message
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]

            # OpenAI format: {"role": "tool", "content": "..."}
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                m["content"] += advisory_suffix
                return

            # Anthropic format: {"role": "user", "content": [{"type": "tool_result", ...}]}
            if m.get("role") == "user" and isinstance(m.get("content"), list):
                # Find last tool_result block in the content list
                for j in range(len(m["content"]) - 1, -1, -1):
                    block = m["content"][j]
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        # Append to content field of the tool_result
                        if isinstance(block.get("content"), str):
                            block["content"] += advisory_suffix
                        elif isinstance(block.get("content"), list):
                            # Content is a list of blocks, add a text block
                            block["content"].append({
                                "type": "text",
                                "text": advisory_suffix,
                            })
                        else:
                            block["content"] = advisory_suffix
                        return
                # If this user message has no tool_result blocks, keep searching
                continue

            # Stop searching if we hit an assistant message (do not go past the
            # current turn boundary)
            if m.get("role") == "assistant":
                break

        # Fallback: no tool_result found (e.g., pre-guard at turn start).
        # Use a lightweight system-scoped message that will not be confused with
        # user instructions.
        self.history.append({
            "role": "user",
            "content": f"[GUARD ADVISORY - note but do not respond to this, "
                       f"prioritize tool results and user requests]\n{msg}",
        })

    def _get_all_schemas(self) -> list[dict]:
        """Return all tool schemas (no phase filtering)."""
        return self.tool_registry.to_schemas(self.provider.schema_format)


    # ── LLM streaming ──────────────────────────────────────────────────────

    def _call_llm_stream(self, messages, schemas):
        content_parts = []
        tool_calls = []
        tool_calls_by_id = {}
        current_tool = None
        stream_truncated = False
        reasoning_only = False
        thinking_capped = False
        usage = {}
        self._streaming_in_code_block = False

        # ── Runtime thinking cap (see thinking_cap.py) ──
        # Abort a monolithic thinking stream once it reaches
        # config.thinking_budget estimated tokens. cap_budget <= 0 disables
        # the cap entirely (default config → behavior identical to before).
        # Only trips while the response is PURE thinking: once text or tool
        # calls exist, the response already has visible output and aborting
        # would only truncate genuine work.
        from flagscale_agent.react.thinking_cap import ThinkingCapCounter
        cap_budget = getattr(self.config, "thinking_budget", 0) or 0
        thinking_counter = ThinkingCapCounter()

        # ── Intra-stream degenerate-repeat breaker (thinking channel) ──
        # The kernel's windowed repeat detector sees ONE fingerprint per
        # RESPONSE (assistant content), so it is blind to a model that
        # degenerates INSIDE a single response. Measured on a real trial: the
        # visible log reached 9382 lines whose tail 1000 held `OK.` x250 /
        # `Go.` x250 / `Let me run.` x249 / `Running.` x125 with ZERO tool
        # calls and ZERO turn boundaries — one unbounded THINKING stream
        # emitting a 4-line cycle. The kernel saw a single response, nothing
        # fired, and the wall burned to the deadline with no deliverable.
        # Calibrated on that log: the degenerate region shows <=6 distinct
        # non-empty lines in a 16-line window (longest line 18 chars), while
        # genuine reasoning shows 16 distinct in the same window (longest line
        # 583 chars). 0 disables.
        _intra_window = self.INTRA_STREAM_REPEAT_WINDOW
        _intra_distinct_max = self.INTRA_STREAM_REPEAT_DISTINCT_MOST
        _intra_lines = []
        _intra_buf = ""
        intra_degenerate = False

        stream = retry_with_backoff(
            lambda: self.provider.chat_stream(messages, schemas),
            max_retries=3,
        )

        thinking_cleared = False
        streaming_trailing_newlines = 0
        streaming_started = False
        sentinel_stripper = display.SentinelStripper()
        thinking_text = ""
        thinking_signature = ""
        # In-stream wall clock for the pure-thinking cap (IN_STREAM_THINKING_WALL_SEC)
        # Started at stream creation, NOT at the first thinking delta: an
        # endpoint that accepts the request and then returns nothing (or only
        # keep-alive chunks) would otherwise never start the clock, so the cap
        # could not fire on the exact case it exists for. Measured on a real
        # trial, the last delta landed at 20:05:38 and the next byte at
        # 20:11:03 — 325s of silence — so silence is the dominant failure mode.
        _stream_t0 = time.time()

        def compress_newlines(text, trailing_from_prev, is_first):
            if not text:
                return text, trailing_from_prev
            if is_first:
                text = text.lstrip('\n')
                if not text:
                    return text, 0
            if trailing_from_prev > 0:
                leading = 0
                for ch in text:
                    if ch == '\n':
                        leading += 1
                    else:
                        break
                total_trailing = trailing_from_prev + leading
                if total_trailing > 2:
                    text = '\n\n' + text[leading:]
            new_trailing = 0
            for ch in reversed(text):
                if ch == '\n':
                    new_trailing += 1
                else:
                    break
            if new_trailing > 2:
                text = text[:len(text) - new_trailing + 2]
                new_trailing = 2
            return text, new_trailing

        max_stream_retries = 2

        # ── Wall-clock watchdog for a SILENT stream ──
        # The event loop below cannot police connection silence: the SSE
        # decoder discards keep-alive/comment lines, so a socket that stops
        # producing data frames delivers ZERO events and the loop body never
        # runs. Only a watchdog that closes the stream from another thread can
        # break a blocking read. It fires only while the response is still
        # pure-thinking (no visible text, no tool calls) so genuine long
        # generations are never touched.
        _wd_done = threading.Event()
        # Self-terminating: the thread must never outlive the call, including on
        # exception paths that skip the explicit stop below. It therefore holds
        # only a WEAK reference to the stream when the iterator supports one and
        # exits as soon as the stream is garbage-collected — a lingering thread
        # would make a later multiprocessing test fork a multi-threaded process.
        # Plain iterators (list_iterator etc.) do not support weak references;
        # the explicit stop on the return path covers them.
        try:
            _stream_ref = weakref.ref(stream)
        except TypeError:
            def _stream_ref():
                return stream
        _silence_state = {"tripped_at": None}

        def _silence_watchdog():
            limit = self.IN_STREAM_THINKING_WALL_SEC
            if limit <= 0:
                return
            while not _wd_done.wait(1.0):
                target = _stream_ref()
                if target is None:
                    return  # stream released -> the consuming call is over
                if (time.time() - _stream_t0) < limit:
                    continue
                if content_parts or tool_calls or thinking_capped:
                    return
                _silence_state["tripped_at"] = time.time() - _stream_t0
                try:
                    target.close()
                except Exception:
                    pass
                return

        if self.IN_STREAM_THINKING_WALL_SEC > 0:
            _wd = threading.Thread(
                target=_silence_watchdog, name="stream-silence-watch",
                daemon=True)
            _wd.start()

        for _stream_attempt in range(1 + max_stream_retries):
            try:
                for event in stream:
                    if not thinking_cleared:
                        display.thinking_done()
                        thinking_cleared = True
                    # ── In-stream wall-clock cap, evaluated on EVERY event ──
                    # The same cap is also checked inside the thinking branch,
                    # but that placement is structurally blind to the failure
                    # it targets: the provider loop above is a blocking socket
                    # read, so when the endpoint goes SILENT no thinking delta
                    # is dispatched and the per-delta check never runs. Measured
                    # on a real trial: timestamps stop at 20:05:38 and the next
                    # byte is 20:11:03 -- 325s of silence, with the per-delta
                    # wall-clock branch (240s) never firing.
                    #
                    # This event-level check covers the case where the endpoint
                    # keeps sending REAL chunks (usage/empty deltas) but no
                    # visible output. It does NOT cover pure connection silence:
                    # the SSE decoder discards comment/keep-alive lines (`:`
                    # prefixed) and emits nothing for them, so a silent socket
                    # delivers zero events and this loop never iterates. The
                    # watchdog thread started above closes the stream to break
                    # that state.
                    if (not thinking_capped and not content_parts
                            and not tool_calls and _stream_t0 is not None
                            and self.IN_STREAM_THINKING_WALL_SEC > 0
                            and time.time() - _stream_t0
                            >= self.IN_STREAM_THINKING_WALL_SEC):
                        capped_now = True
                        thinking_capped = True
                        reasoning_only = True
                        if display.thinking_enabled():
                            display._write(
                                display.dim(
                                    f"\n[thinking stream ran "
                                    f"{time.time() - _stream_t0:.0f}s of wall clock "
                                    f"with no visible output — aborting this stream]"
                                )
                            )
                        try:
                            stream.close()
                        except Exception:
                            pass
                        break
                    if event["type"] == "text":
                        # Keep the RAW text for the kernel's completion gate...
                        content_parts.append(event["content"])
                        # ...but strip completion sentinels from what reaches the
                        # screen, so a gate-blocked [TASK_COMPLETE] never shows up
                        # with authority before the gate runs (kernel prints the
                        # authoritative marker only once completion is accepted).
                        text = sentinel_stripper.feed(event["content"])
                        if not text:
                            continue
                        text, streaming_trailing_newlines = compress_newlines(
                            text, streaming_trailing_newlines, not streaming_started)
                        if text:
                            streaming_started = True
                        if display._use_color():
                            fence_count = text.count("```")
                            if self._streaming_in_code_block:
                                text = display.cyan(text)
                            elif "```" in text:
                                text = display.render_markdown(text)
                            else:
                                text = display.blue(text)
                            if fence_count % 2 == 1:
                                self._streaming_in_code_block = not self._streaming_in_code_block
                        display._write(text)
                    elif event["type"] == "tool_start":
                        # Clear thinking spinner on first tool call
                        if not thinking_cleared:
                            display.thinking_clear()
                            thinking_cleared = True
                        current_tool = {
                            "id": event["id"],
                            "name": event["name"],
                            "arguments_json": "",
                        }
                        tool_calls.append(current_tool)
                        if event["id"]:
                            tool_calls_by_id[event["id"]] = current_tool
                    elif event["type"] == "tool_delta":
                        delta_id = event.get("id", "")
                        target = tool_calls_by_id.get(delta_id, current_tool) if delta_id else current_tool
                        if target:
                            target["arguments_json"] += event["arguments_delta"]
                    elif event["type"] == "reasoning_only":
                        if not content_parts and not tool_calls:
                            reasoning_only = True
                    elif event["type"] == "thinking":
                        thinking_text += event["content"]
                        if display.thinking_enabled():
                            if display._use_color():
                                display._write(display.dim(event["content"]))
                            else:
                                display._write(event["content"])
                        # ── Runtime cap check ──
                        if not thinking_capped and not content_parts and not tool_calls:
                            thinking_counter.add(event["content"])
                            capped_now, _capped_at = thinking_counter.state(cap_budget)
                            # ── In-stream wall-clock cap (see
                            # IN_STREAM_THINKING_WALL_SEC) ──
                            # Token-based caps cannot fire on slow endpoints
                            # (~4.6 tok/s served: a 16384-token budget is
                            # ~2000s of wall). Judge TIME since the stream
                            # began emitting pure thinking; abort at the cap.
                            if not capped_now:
                                if _stream_t0 is None:
                                    _stream_t0 = time.time()
                                if (self.IN_STREAM_THINKING_WALL_SEC > 0
                                        and time.time() - _stream_t0
                                        >= self.IN_STREAM_THINKING_WALL_SEC):
                                    capped_now = True
                            # ── Intra-stream degenerate-repeat breaker ──
                            # Detect an unbounded filler cycle inside ONE
                            # thinking stream, per completed line, and abort it
                            # instead of letting it consume the wall clock.
                            if _intra_window > 0:
                                _intra_buf, _fired = _intra_stream_update(
                                    _intra_buf, event["content"],
                                    _intra_lines,
                                    _intra_window, _intra_distinct_max,
                                    self.INTRA_STREAM_CHAR_COLLAPSE_MIN_RUN,
                                    self.INTRA_STREAM_CYCLE_TAIL_CHARS,
                                    self.INTRA_STREAM_CYCLE_DISTINCT_MAX,
                                )
                                if _fired == 1:
                                    intra_degenerate = True
                                    if display.thinking_enabled():
                                        display._write(
                                            display.dim(
                                                "\n[char-collapse in thinking "
                                                f"stream — run of one byte >= "
                                                f"{self.INTRA_STREAM_CHAR_COLLAPSE_MIN_RUN}"
                                                " — aborting this stream]"
                                            )
                                        )
                                elif _fired == 2:
                                    intra_degenerate = True
                                    if display.thinking_enabled():
                                        display._write(
                                            display.dim(
                                                "\n[intra-stream filler cycle in "
                                                "thinking stream — single line, "
                                                f"{self.INTRA_STREAM_CYCLE_TAIL_CHARS} "
                                                f"chars with <= "
                                                f"{self.INTRA_STREAM_CYCLE_DISTINCT_MAX}"
                                                " distinct bytes — aborting this stream]"
                                            )
                                        )
                                elif _fired == 3:
                                    intra_degenerate = True
                                    if display.thinking_enabled():
                                        display._write(
                                            display.dim(
                                                "\n[intra-stream degenerate repeat "
                                                f"detected ({_intra_window} lines, "
                                                f"{len(set(_intra_lines))} distinct)"
                                                " — aborting this stream]"
                                            )
                                        )
                                if intra_degenerate:
                                    try:
                                        stream.close()
                                    except Exception:
                                        pass
                                    break
                            if capped_now:
                                thinking_capped = True
                                reasoning_only = True
                                if display.thinking_enabled():
                                    display._write(
                                        display.dim(
                                            f"\n[thinking reached {cap_budget} token cap — forcing convergence]"
                                        )
                                    )
                                # Abort the stream. The kernel re-injects the
                                # accumulated thinking as a user message with a
                                # convergence directive, so the reasoning is
                                # segmented, not lost.
                                break
                    elif event["type"] == "thinking_start":
                        if not thinking_cleared:
                            display.thinking_done()
                            thinking_cleared = True
                        if display.thinking_enabled():
                            display._write(display.dim("💭 thinking...\n"))
                    elif event["type"] == "block_stop":
                        if thinking_text and display.thinking_enabled():
                            display._write(display.dim("\n"))
                    elif event["type"] == "signature":
                        thinking_signature += event["content"]
                    elif event["type"] == "usage":
                        usage = {
                            "input_tokens": event.get("input_tokens"),
                            "output_tokens": event.get("output_tokens"),
                            "cache_read_input_tokens": event.get("cache_read_input_tokens"),
                            "cache_creation_input_tokens": event.get("cache_creation_input_tokens"),
                        }
                    elif event["type"] == "done":
                        break
                if thinking_capped:
                    # Release the underlying HTTP stream promptly instead of
                    # waiting for GC (we abandoned the generator mid-iteration).
                    try:
                        stream.close()
                    except Exception:
                        pass
                break
            except KeyboardInterrupt:
                if not thinking_cleared:
                    display.thinking_clear()
                raise
            except Exception as e:
                if not thinking_cleared:
                    display.thinking_clear()
                    thinking_cleared = True
                if content_parts or tool_calls:
                    stream_truncated = True
                    break
                # Non-retryable 400 errors (e.g., tool_use/tool_result pairing)
                # should not be retried - they are permanent request errors.
                from flagscale_agent.react.retry import _extract_status
                _status = _extract_status(e)
                if _status == 400:
                    raise
                if _stream_attempt < max_stream_retries:
                    wait = 2 ** _stream_attempt
                    display.warn(f"Stream interrupted, retrying in {wait}s...")
                    time.sleep(wait)
                    stream = retry_with_backoff(
                        lambda: self.provider.chat_stream(messages, schemas),
                        max_retries=3,
                    )
                    continue
                raise

        # Flush any residual held-back text (a partial that never completed into
        # a sentinel, e.g. from a truncated stream) — it is genuine text.
        residual = sentinel_stripper.flush()
        if residual:
            residual, streaming_trailing_newlines = compress_newlines(
                residual, streaming_trailing_newlines, not streaming_started)
            if residual:
                streaming_started = True
                display._write(display.blue(residual) if display._use_color() else residual)

        if content_parts:
            if streaming_trailing_newlines > 1 and display._use_color():
                up = streaming_trailing_newlines - 1
                display._write(f"\033[{up}A\033[J")
                print()
            elif streaming_trailing_newlines == 0:
                print()

        # Stop the silence watchdog: the stream is finished, so it must not
        # close anything later. If it already tripped, honour the abort with
        # the same "capped" semantics as the token cap — the kernel reinjects
        # the accumulated reasoning with a convergence directive, so nothing
        # reasoned during that window is lost.
        _wd_done.set()
        if _silence_state["tripped_at"] is not None and not content_parts:
            thinking_capped = True
            reasoning_only = True

        parsed_tool_calls = None
        if tool_calls:
            parsed_tool_calls = []
            for tc in tool_calls:
                try:
                    arguments = json.loads(tc["arguments_json"]) if tc["arguments_json"] else {}
                except json.JSONDecodeError:
                    # Incomplete JSON from truncated stream - skip this tool call
                    continue
                parsed_tool_calls.append({"id": tc["id"], "name": tc["name"], "arguments": arguments})
            if not parsed_tool_calls:
                parsed_tool_calls = None

        return {"content": "".join(content_parts) or None, "tool_calls": parsed_tool_calls, "truncated": stream_truncated, "reasoning_only": reasoning_only, "thinking_capped": thinking_capped, "intra_degenerate": intra_degenerate, "thinking": thinking_text or None, "signature": thinking_signature or None}, usage

    # ── Tool execution (delegated to ToolExecutor) ──────────────────────────

    def _execute_tools(self, tool_calls):
        """Execute tools, intercepting evict/recall for special handling."""
        # Separate evict/recall from normal tools
        normal_calls = []
        special_results = {}  # index in tool_calls -> result string

        for i, tc in enumerate(tool_calls):
            if tc["name"] == "evict":
                special_results[i] = self.context_manager.handle_evict(tc.get("arguments", {}))
            elif tc["name"] == "recall":
                special_results[i] = self.context_manager.handle_recall(tc.get("arguments", {}))
            else:
                normal_calls.append((i, tc))

        # Execute normal tools
        if normal_calls:
            normal_tc_list = [tc for _, tc in normal_calls]
            try:
                normal_results = self._tool_executor.execute_batch(normal_tc_list)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                normal_results = [
                    f"Error executing tool: {e}\n\n[Tool: {tc['name']}]\n[Args: {tc.get('arguments', {})}]\n[Traceback]\n{tb}"
                    for tc in normal_tc_list
                ]
            for (orig_idx, _), result in zip(normal_calls, normal_results):
                special_results[orig_idx] = result

        # Reassemble in original order
        return [special_results[i] for i in range(len(tool_calls))]

    def _execute_tool(self, tool_call):
        return self._tool_executor.execute_single(tool_call)

    def _append_tool_results(self, tool_results: list[dict]):
        """Append tool results, merging them into a single user message.

        Anthropic API requires all tool_results for a batch of tool_uses
        to be in the SAME user message (immediately following the assistant message).
        """
        if not tool_results:
            return
        if len(tool_results) == 1:
            self.history.append(tool_results[0])
        else:
            # Merge all tool_result blocks into one user message
            merged_content = []
            for tr in tool_results:
                content = tr.get("content", [])
                if isinstance(content, list):
                    merged_content.extend(content)
                elif isinstance(content, str):
                    merged_content.append({"type": "text", "text": content})
            self.history.append({"role": "user", "content": merged_content})

    @staticmethod
    def _tool_display_summary(tool_name: str, arguments: dict) -> str:
        return tool_display_summary(tool_name, arguments)

    @staticmethod
    def _is_context_limit_error(e) -> bool:
        return _is_context_limit_error(e)


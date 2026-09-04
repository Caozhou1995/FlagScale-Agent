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
import time
import uuid
from collections import deque
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
    append_session_index, get_recent_sessions,
)
from flagscale_agent.react.skills import SkillManager
from flagscale_agent.react.tools import ToolRegistry
from flagscale_agent.react.tools.edit_file import EditFileTool
from flagscale_agent.react.tools.load_knowledge import LoadKnowledgeTool
from flagscale_agent.react.tools.load_skill import LoadSkillTool
from flagscale_agent.react.tools.read_file import ReadFileTool
from flagscale_agent.react.tools.shell import ShellTool, ShellJobsTool
from flagscale_agent.react.tools.write_file import WriteFileTool
from flagscale_agent.react.tools.web_fetch import WebFetchTool
# find_log removed - merged into monitor

from flagscale_agent.react.memory import Memory
from flagscale_agent.react.tools.memory_write import MemoryWriteTool
from flagscale_agent.react.tools.memory_read import MemoryReadTool
from flagscale_agent.react.tools.memory_list import MemoryListTool
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

from flagscale_agent.react.guard.unit_test import UnitTestGuard
from flagscale_agent.react.guard.memory_discipline import MemoryDisciplineGuard
from flagscale_agent.react.guard.post_evict_recovery import PostEvictRecoveryGuard
from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard
from flagscale_agent.react.guard.arg_type import ArgTypeGuard
from flagscale_agent.react.guard.verification import VerificationGuard

from flagscale_agent.react.prompt_builder import PromptBuilder
from flagscale_agent.react.tool_executor import ToolExecutor, tool_display_summary

from flagscale_agent.react.judge import Judge
from flagscale_agent.react.commands import CommandHandler


# ── WorkerAgent ──────────────────────────────────────────────────────────────

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

        # ── Infrastructure ──
        self.skill_manager = _skill_manager or SkillManager(config.skill_dirs)
        self.tool_registry = _tool_registry or ToolRegistry()

        # Knowledge system
        from flagscale_agent.knowledge import KnowledgeManager
        self._knowledge_manager = KnowledgeManager()

        self._session_id = uuid.uuid4().hex[:8]
        from flagscale_agent.react.paths import get_sessions_root, get_memory_dir
        sessions_root = config.session_dir or get_sessions_root()
        session_dir = os.path.join(sessions_root, self._session_id)
        os.makedirs(session_dir, exist_ok=True)
        self._session_dir = session_dir
        self._sessions_root = sessions_root


        # Provider and history must be created before ContextManager
        if not config.api_key:
            raise ValueError(
                "API key not found. Set ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
            )
        self.provider = _provider or get_provider(
            config.provider, config.model, config.api_key,
            config.base_url, config.max_output_tokens,
            thinking_budget=config.thinking_budget,
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

        guard_registry.register(UnitTestGuard())
        # Memory discipline guard (always active)
        guard_registry.register(MemoryDisciplineGuard())
        # Post-evict recovery guard (always active)
        guard_registry.register(PostEvictRecoveryGuard())
        # Knowledge-first guard (always active, inject-only)
        self._knowledge_guard = KnowledgeSkillGuard()
        guard_registry.register(self._knowledge_guard)
        # Verification discipline guard (always active, block on step_done without evidence)
        guard_registry.register(VerificationGuard(plan=self.task_plan))

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
        self.tool_registry.register(
            ShellTool(
                remind_interval=self.config.shell_remind_interval,
                env=self.config.shell_env,
                health_judge_fn=self._health_judge,
            )
        )
        # Background-job control tool: poll/wait/list/kill jobs started with
        # shell(background=true) or auto-detached from the monitor loop.
        self.tool_registry.register(ShellJobsTool())
        
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
        self.tool_registry.register(MemoryWriteTool(self.memory, self._session_id, task_plan=self.task_plan))
        self.tool_registry.register(MemoryReadTool(self.memory))
        self.tool_registry.register(MemoryListTool(self.memory))
        self.tool_registry.register(PlanCreateTool(self.task_plan, self._session_id))
        self.tool_registry.register(PlanUpdateTool(self.task_plan))
        self.tool_registry.register(PlanStatusTool(self.task_plan))
        
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

    def _build_proxies(self) -> dict[str, str]:
        proxies = {}
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            val = os.environ.get(var)
            if val:
                proxies[var.lower()] = val
        return proxies

    # ── System prompt ────────────────────────────────────────────────────────

    def _refresh_system_prompt(self, memory_context: str = "", plan_context: str = ""):
        tool_names = [t.name for t in self.tool_registry.all_tools()]
        self._prompt_builder.refresh(
            history=self.history,
            active_skill_content={},
            shared_storage_paths=getattr(self, "_shared_storage_paths", []),
            memory_context=memory_context,
            plan_context=plan_context,
            tool_names=tool_names,
            session_dir=self._session_dir,
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

    def _task_budget_summary(self) -> str:
        """Summarize whole-task cumulative wall-clock vs the total time budget.

        Returns "" when no budget is configured (env FLAGSCALE_AGENT_TIME_BUDGET_SEC
        unset or non-positive), so the health prompt stays byte-identical to the
        no-budget version. When set, reports elapsed-since-turn-start and the
        total budget so the judge can reason about the task-level allowance.

        Elapsed is measured from the CURRENT turn's start (self._turn_start,
        re-stamped each _react_loop), not from session start: an interactive
        session may span many turns over hours, and only the active turn's time
        is meaningful against a per-task budget. In single-shot mode a turn is
        the whole session, so the two coincide.
        """
        raw = os.environ.get("FLAGSCALE_AGENT_TIME_BUDGET_SEC", "").strip()
        if not raw:
            return ""
        try:
            budget = float(raw)
        except (TypeError, ValueError):
            return ""
        if budget <= 0:
            return ""
        elapsed = max(0.0, time.time() - self._turn_start)
        remaining = budget - elapsed
        pct = (elapsed / budget) * 100.0 if budget else 0.0

        def _fmt(sec: float) -> str:
            sec = int(max(0.0, sec))
            m, s = divmod(sec, 60)
            h, m = divmod(m, 60)
            if h:
                return f"{h}h{m:02d}m"
            return f"{m}m{s:02d}s"

        return (
            f"cumulative task time elapsed {_fmt(elapsed)} of total budget "
            f"{_fmt(budget)} ({pct:.0f}% used, ~{_fmt(remaining)} remaining)"
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

    def _build_continuation_message(self, summary: str) -> str:
        """Build the full continuation message to inject after hard reset."""
        reset_count = self.history._reset_count
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

        return f"{header}\n{summary}\n{footer}"

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

        session = PromptSession(
            history=FileHistory(history_file),
            completer=completer,
            multiline=True,
            key_bindings=kb,
            style=PromptStyle.from_dict({
                "prompt": "#87d787 bold",
                "": "#e4e4e4",
            }),
        )

        while True:
            try:
                user_input = session.prompt([("class:prompt", "> ")]).strip()
            except (EOFError, KeyboardInterrupt):
                self._exit()
                break

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
        self.history.append({"role": "user", "content": query})
        try:
            self._react_loop()
        except Exception:
            display.warn("WorkerAgent._run_single_shot() react loop failed")
        finally:
            # Headless single-shot has no per-turn REPL save (unlike the
            # interactive loop), so persist explicitly here to guarantee a
            # normally-completed run is durable. A harness timeout that kills
            # the process mid-run is handled separately by the SIGTERM handler
            # installed in _install_signal_handlers().
            self._auto_save()
            try:
                from flagscale_agent.react.tools.shell import _JOB_REGISTRY
                _JOB_REGISTRY.cleanup_all()
            except Exception:
                pass

    def _restore_session(self, data: dict, session_dir: str):
        """Restore a previous session - take over its session_id and dir."""
        # Take over the old session identity
        old_session_dir = self._session_dir
        self._session_id = data.get("session_id", self._session_id)
        self._session_dir = session_dir

        # Re-point plan to old session's dirs
        self.task_plan._dir = os.path.join(session_dir, "plans")
        # Re-point swap store to restored session's dir
        from flagscale_agent.react.swap_store import SwapStore
        from flagscale_agent.react.context_manager import ContextManager
        self.context_manager = ContextManager(
            history=self.history,
            swap_store=SwapStore(os.path.join(session_dir, "swap_store")),
        )

        # Clean up the empty new session dir if it's different
        if old_session_dir != session_dir:
            try:
                import shutil
                if os.path.isdir(old_session_dir) and not os.listdir(old_session_dir):
                    shutil.rmtree(old_session_dir, ignore_errors=True)
            except Exception:
                pass

        # Restore _full_log from conversation_full.json if it exists.
        # This preserves the complete audit trail across /reload and hard resets.
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
                    # Restore hard reset state
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
                # Only append to _messages; _full_log already has the complete history
                self.history._messages.append(msg)
                # Tag with ext_idx from full_log length (messages were already recorded there)
                msg["_ext_idx"] = msg.get("_ext_idx", len(self.history._full_log))
            else:
                self.history.append(msg)
        # Restore turn count and session input history
        self._session_input_history = data.get("session_input_history", [])
        self.turn_count = data.get("turn_count", len(self._session_input_history))
        loaded = data.get("loaded_skills", [])
        # Restore session token counts for cumulative tracking across resume/reload
        self._session_input_tokens = data.get("session_input_tokens", 0)
        self._session_output_tokens = data.get("session_output_tokens", 0)
        for skill_name in loaded:
            try:
                content = self.skill_manager.load(skill_name)
                if content:
                    self._loaded_skills.add(skill_name)
            except Exception:
                pass
        # Refresh system prompt with restored context
        self._refresh_system_prompt()

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

        # Refresh system prompt if plan tools were used
        if any(tc["name"] in ("plan_create", "plan_update", "plan_status")
               for tc in tool_calls):
            self._refresh_system_prompt()

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
        usage = {}
        self._streaming_in_code_block = False

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
        for _stream_attempt in range(1 + max_stream_retries):
            try:
                for event in stream:
                    if not thinking_cleared:
                        display.thinking_done()
                        thinking_cleared = True
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
                        if display._use_color():
                            display._write(display.dim(event["content"]))
                        else:
                            display._write(event["content"])
                    elif event["type"] == "thinking_start":
                        if not thinking_cleared:
                            display.thinking_done()
                            thinking_cleared = True
                        display._write(display.dim("💭 thinking...\n"))
                    elif event["type"] == "block_stop":
                        if thinking_text:
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

        return {"content": "".join(content_parts) or None, "tool_calls": parsed_tool_calls, "truncated": stream_truncated, "reasoning_only": reasoning_only, "thinking": thinking_text or None, "signature": thinking_signature or None}, usage

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


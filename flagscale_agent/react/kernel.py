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

"""AgentKernel — minimal event loop core.

Replaces the monolithic _react_loop() in agent.py.

Responsibilities:
- LLM call + retry on context overflow
- Guard pre/post checks
- Tool execution dispatch
- State machine transitions
- Token accounting

Everything else (session, history, tools, prompts) is injected via dependencies.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from flagscale_agent.react.guard import GuardContext, GuardRegistry, GuardVerdict
from flagscale_agent.react import display


@dataclass
class KernelDeps:
    """All external dependencies the Kernel needs — injected, not imported."""

    provider: Any                          # LLM provider
    history: Any                           # HistoryManager
    tool_registry: Any                     # ToolRegistry
    judge: Any                             # Judge
    guard_registry: GuardRegistry
    config: Any                            # AgentConfig
    display: Any                           # display module
    get_schemas_fn: Callable               # () -> list[dict]
    inject_message_fn: Callable            # (msg: str) -> None  [block/escalate only]
    append_tool_results_fn: Callable       # (results: list) -> None
    format_tool_result_fn: Callable        # (id, result) -> dict
    execute_tools_fn: Callable             # (tool_calls) -> list[str]
    is_context_limit_error_fn: Callable    # (exc) -> bool
    append_advisory_fn: Callable | None = None  # (msg: str) -> None [soft inject → tool_result]
    call_llm_fn: Callable | None = None    # (messages, schemas) -> (response, usage)
    task_plan: Any = None                  # TaskPlan (optional)
    on_response_fn: Callable | None = None  # (response) -> None, called after LLM response appended
    on_tool_results_fn: Callable | None = None  # (tool_calls, results) -> None, called after tool exec


@dataclass
class KernelResult:
    """Result of one kernel run (one user turn)."""

    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed: float = 0.0
    interrupted: bool = False
    stop_reason: str = ""


class AgentKernel:
    """Minimal event loop. < 200 lines of logic.

    One instance per agent. Call run_turn() for each user message.
    """

    def __init__(self, deps: KernelDeps):
        self.deps = deps
        self._interrupted = False
        self._continuation_count = 0

    def run_turn(self) -> KernelResult:
        """Run one ReAct turn (one user message → completion).

        Returns KernelResult with token stats and stop reason.
        """
        result = KernelResult()
        d = self.deps
        max_iter = d.config.max_iterations
        turn_start = time.time()

        self._interrupted = False
        self._continuation_count = 0  # Reset per turn
        d.judge.reset_turn()
        d.guard_registry.reset_turn()

        _prev_handler = signal.getsignal(signal.SIGINT)

        def _sigint(signum, frame):
            if self._interrupted:
                signal.signal(signal.SIGINT, _prev_handler)
                raise KeyboardInterrupt
            self._interrupted = True
            d.display.interrupted()

        signal.signal(signal.SIGINT, _sigint)

        try:
            for iteration in range(max_iter):
                if self._interrupted:
                    break

                # Guard reset happens per-turn (at line 101), not per-iteration
                d.judge.reset_turn()

                schemas = d.get_schemas_fn()

                # ── Pre-guard checks ──
                ctx = self._build_ctx(tool_name="", tool_args={}, tool_result=None)
                verdict = d.guard_registry.check_pre(ctx)
                if verdict is not None:
                    blocked = self._apply_verdict(verdict, pre=True)
                    if blocked:
                        # Don't break — re-prompt LLM with the guard message in history
                        # Guard message was injected by _apply_verdict, now give LLM a chance to respond
                        if iteration >= max_iter - 1:
                            # Safety: if we're at max_iter, stop to avoid infinite loop
                            result.stop_reason = f"blocked_by_guard_at_max_iter: {verdict.reason}"
                            break
                        # Continue to next iteration — LLM will see the guard message
                        continue

                # ── LLM call ──
                d.display.thinking()
                messages = d.history.get_messages()
                self._t0 = time.time()

                try:
                    _call = d.call_llm_fn or (lambda m, s: d.provider.chat_stream(m, s))
                    response, usage = _call(messages, schemas)
                except KeyboardInterrupt:
                    d.display.interrupted()
                    self._interrupted = True
                    break
                except Exception as e:
                    if d.is_context_limit_error_fn(e):
                        response, usage = self._recover_context_overflow(e, schemas)
                        if response is None:
                            result.stop_reason = "context_overflow_unrecoverable"
                            break
                    else:
                        d.display.thinking_clear()
                        display.warn(f"LLM call failed: {e}")
                        result.stop_reason = f"llm_error: {e}"
                        break

                elapsed = time.time() - getattr(self, "_t0", time.time())
                in_tok = usage.get("input_tokens") or 0
                out_tok = usage.get("output_tokens") or 0
                cache_read = usage.get("cache_read_input_tokens") or 0
                cache_create = usage.get("cache_creation_input_tokens") or 0
                total_in_tok = in_tok + cache_read + cache_create
                result.input_tokens += total_in_tok
                result.output_tokens += out_tok
                if total_in_tok:
                    d.history.report_actual_tokens(total_in_tok)

                d.display.llm_done(elapsed, total_in_tok, out_tok,
                                   cache_read_tokens=cache_read or None,
                                   cache_creation_tokens=cache_create or None)

                if self._interrupted:
                    break

                d.history.append(d.provider.format_assistant_message(response))

                if d.on_response_fn:
                    d.on_response_fn(response)

                # ── No tool calls → done ──
                if not response.get("tool_calls"):
                    result.iterations = iteration + 1
                    # Check for explicit stop signals in assistant response
                    assistant_text = ""
                    if isinstance(response.get("content"), str):
                        assistant_text = response["content"]
                    elif isinstance(response.get("content"), list):
                        assistant_text = "".join(
                            b.get("text", "") for b in response["content"]
                            if isinstance(b, dict) and b.get("type") == "text"
                        )

                    # ── Empty output defense: auto-retry up to 3 times ──
                    if not assistant_text.strip():
                        empty_retries = getattr(self, "_empty_output_retries", 0)
                        if empty_retries < 3:
                            self._empty_output_retries = empty_retries + 1
                            d.display.warn(f"Empty LLM output (retry {empty_retries + 1}/3), auto-continuing...")
                            # Remove the empty assistant message we just appended
                            msgs = d.history.get_messages()
                            if msgs and msgs[-1].get("role") == "assistant":
                                msgs.pop()
                            # Inject a nudge
                            d.history.append({"role": "user", "content": "[system: empty response detected, please continue your work]"})
                            continue
                        else:
                            self._empty_output_retries = 0
                            result.stop_reason = "empty_output_max_retries"
                            break
                    else:
                        self._empty_output_retries = 0

                    if "[TASK_COMPLETE]" in assistant_text or "[NEED_USER_INPUT]" in assistant_text:
                        result.stop_reason = "explicit_signal"
                        break

                    # ── Continuation: LLM didn't stop, keep going ──
                    # Short output retry (truncated/confused)
                    if (len(assistant_text.strip()) < 10 and iteration > 0
                            and getattr(self, "_last_turn_had_tools", False)):
                        short_retries = getattr(self, "_short_output_retries", 0)
                        if short_retries < 2:
                            self._short_output_retries = short_retries + 1
                            d.display.warn(f"Short output without tools after active turn, auto-continuing...")
                            d.history.append({"role": "user", "content": "[system: please continue your work]"})
                            continue
                        else:
                            self._short_output_retries = 0

                    # Context pressure check
                    pressure = d.history.get_context_pressure() if hasattr(d.history, 'get_context_pressure') else 0
                    if pressure >= 0.85:
                        result.stop_reason = "context_pressure"
                        break

                    # Continuation count limit (hard ceiling)
                    self._continuation_count += 1
                    if self._continuation_count > d.config.max_continuations:
                        result.stop_reason = "max_continuations"
                        break

                    continuation = self._generate_continuation()
                    d.history.append({"role": "user", "content": continuation})
                    continue

                self._continuation_count = 0

                # ── Execute tools ──
                _pre_guard_verdicts = []
                try:
                    tool_calls = response["tool_calls"]

                    # ── Per-tool pre-guard checks ──
                    # Give guards a chance to block individual tool calls before execution.
                    # IMPORTANT: Do NOT inject messages here (before tool_results are appended)
                    # because that would break tool_use/tool_result pairing required by the API.
                    # Collect verdicts and apply them AFTER tool_results are in history.
                    blocked_indices = set()
                    _pre_guard_verdicts = []  # (verdict, blocked) pairs to apply after tool_results
                    _seen_injects = set()  # Deduplicate inject across multiple tool_calls
                    for i, tc in enumerate(tool_calls):
                        ctx = self._build_ctx(
                            tool_name=tc["name"],
                            tool_args=tc.get("arguments", {}),
                            tool_result=None,
                        )
                        verdict = d.guard_registry.check_pre(ctx)
                        if verdict is not None:
                            if verdict.action in ("block", "escalate"):
                                blocked_indices.add(i)
                                # Deduplicate block/escalate across tool_calls
                                msg_key = verdict.message[:120]
                                if msg_key not in _seen_injects:
                                    _seen_injects.add(msg_key)
                                    _pre_guard_verdicts.append(verdict)
                                # Display is handled later in _apply_verdict — don't display here
                            elif verdict.action == "inject":
                                # Soft advisory — defer until after tool_results are appended
                                # Deduplicate: same message from same guard across tool_calls
                                msg_key = verdict.message[:120]
                                if msg_key not in _seen_injects:
                                    _seen_injects.add(msg_key)
                                    _pre_guard_verdicts.append(verdict)

                    # Execute tools (skip blocked ones)
                    if blocked_indices:
                        blocked_msg = (
                            "[BLOCKED BY GUARD] This tool call was prevented. "
                            "Read the guard message below carefully and retry with a corrected tool call. "
                            "Do NOT respond with plain text only — you MUST make a tool call in your next response."
                        )
                        if len(blocked_indices) == len(tool_calls):
                            # All blocked
                            results = [blocked_msg] * len(tool_calls)
                        else:
                            # Partial block: execute non-blocked, merge results
                            exec_calls = [tc for i, tc in enumerate(tool_calls) if i not in blocked_indices]
                            exec_results = d.execute_tools_fn(exec_calls)
                            results = []
                            exec_idx = 0
                            for i in range(len(tool_calls)):
                                if i in blocked_indices:
                                    results.append(blocked_msg)
                                else:
                                    results.append(exec_results[exec_idx])
                                    exec_idx += 1
                    else:
                        results = d.execute_tools_fn(tool_calls)
                except KeyboardInterrupt:
                    d.display.interrupted()
                    self._interrupted = True
                    break
                except Exception as e:
                    display.warn(f"Tool execution failed: {e}")
                    # Create error results for all tool calls so the LLM can see what happened
                    tool_calls = response["tool_calls"]
                    results = [f"Error executing tool: {e}"] * len(tool_calls)

                # ── Post-guard checks (per tool) ──
                post_verdicts = []
                _seen_post_injects = set()  # Deduplicate inject across tool_calls
                for tc, tool_result in zip(tool_calls, results):
                    ctx = self._build_ctx(
                        tool_name=tc["name"],
                        tool_args=tc.get("arguments", {}),
                        tool_result=tool_result,
                    )
                    verdict = d.guard_registry.check_post(ctx)
                    if verdict is not None:
                        # Deduplicate all verdict types across multiple tool_calls
                        msg_key = verdict.message[:120]
                        if msg_key in _seen_post_injects:
                            continue
                        _seen_post_injects.add(msg_key)
                        post_verdicts.append(verdict)

                tool_results = [
                    d.format_tool_result_fn(tc["id"], r)
                    for tc, r in zip(tool_calls, results)
                ]
                d.append_tool_results_fn(tool_results)

                # Apply deferred pre-guard verdicts AFTER tool_results are in history,
                # so inject/block messages don't break tool_use → tool_result pairing.
                for verdict in _pre_guard_verdicts:
                    self._apply_verdict(verdict, pre=True)

                # Apply post-guard verdicts AFTER tool results are appended,
                # so inject messages don't break tool_call → tool_result pairing.
                # v4: post verdicts are advisory/escalate only (never block),
                # so they don't stop the turn.
                for verdict in post_verdicts:
                    self._apply_verdict(verdict, pre=False)

                if d.on_tool_results_fn:
                    d.on_tool_results_fn(tool_calls, results)

                self._last_turn_had_tools = True
                result.iterations = iteration + 1

                # ── Self-modification detection ──
                # If any file tool modified flagscale_agent/ source, stop and ask for /reload
                if self._detect_self_modification(tool_calls):
                    # Inject a notice to the assistant so it knows to stop
                    d.history.append({"role": "user", "content": (
                        "[system: You just modified FlagScale Agent's own source code "
                        "(flagscale_agent/). These changes require /reload to take effect. "
                        "STOP here and tell the user to run /reload. Do NOT continue other work.]"
                    )})
                    # Do one more LLM call to let it produce the stop message
                    response, usage = d.stream_fn(d.history.get_messages())
                    if d.append_response_fn:
                        d.append_response_fn(response)
                    result.stop_reason = "self_modification_reload_needed"
                    break

        finally:
            signal.signal(signal.SIGINT, _prev_handler)

        result.interrupted = self._interrupted
        result.elapsed = time.time() - turn_start
        return result

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_last_assistant_text(self) -> str:
        """Extract text from last assistant message in history."""
        d = self.deps
        if not d.history:
            return ""
        for msg in reversed(d.history.messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [b.get("text", "") for b in content
                             if isinstance(b, dict) and b.get("type") == "text"]
                    return "".join(texts)
        return ""

    def _build_ctx(self, tool_name: str, tool_args: dict, tool_result: str | None) -> GuardContext:
        d = self.deps
        history = d.history
        # Resolve tool effects from registry
        try:
            tool = d.tool_registry.get(tool_name)
        except (KeyError, AttributeError):
            pass
        # Extract override_reason from tool_args (LLM declares why a blocked call is justified)
        # Use .get() + conditional del to avoid mutating the original dict unexpectedly
        override_reason = ""
        if tool_args and "_override_reason" in tool_args:
            override_reason = tool_args["_override_reason"]
            tool_args = {k: v for k, v in tool_args.items() if k != "_override_reason"}
        # Get last assistant text for guards that need to scan LLM responses
        assistant_text = self._get_last_assistant_text()
        return GuardContext(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            
            turn_count=getattr(d.config, "_turn_count", 0),
            context_pressure=history.get_context_pressure() if history else 0.0,
            evictable_indexes=history.get_evictable_indexes() if history else [],
            messages=history.get_messages() if history else [],
            classify_fn=d.judge.classify,
            override_reason=override_reason,
            assistant_text=assistant_text,
        )

    def _apply_verdict(self, verdict: GuardVerdict, pre: bool) -> bool:
        """Apply a guard verdict. Returns True if this tool call should be blocked.

        v6 semantics (escalation chain: inject → block → escalate):
        - inject: soft advisory appended to tool_result, turn continues
        - block: prevent tool execution, LLM can override with reason
        - escalate: prevent tool execution + independent message, NOT overridable
        """
        d = self.deps
        if verdict.action == "block":
            d.inject_message_fn(verdict.message)
            display.guard_block(verdict.message)
            return True
        elif verdict.action == "escalate":
            # Ultimate enforcement — block tool + inject message, no override path.
            d.inject_message_fn(verdict.message)
            display.guard_escalate(verdict.message)
            return True
        elif verdict.action == "inject":
            # v4: Soft advisory — append to last tool_result instead of
            # creating a new user message. This avoids conversation pollution.
            if d.append_advisory_fn:
                d.append_advisory_fn(verdict.message)
            else:
                d.inject_message_fn(verdict.message)
            display.guard_inject(verdict.message)
        return False

    def _recover_context_overflow(self, exc, schemas):
        """Evict old messages on context overflow, then retry LLM call once."""
        d = self.deps

        # Save recovery state to plan before eviction
        self._save_recovery_state()

        d.display.thinking_clear()
        display.warn("Context overflow, evicting old messages...")

        evictable = d.history.get_evictable_indexes()
        if not evictable:
            display.warn("No evictable messages — cannot recover.")
            return None, {}

        # Evict 50% of evictable messages (aggressive, single pass)
        count = max(1, len(evictable) * 50 // 100)
        for idx in evictable[:count]:
            d.history.evict_message(idx)

        messages = d.history.get_messages()
        _call = d.call_llm_fn or (lambda m, s: d.provider.chat_stream(m, s))
        try:
            d.display.thinking()
            return _call(messages, schemas)
        except Exception as e2:
            d.display.thinking_clear()
            display.warn(f"LLM call failed after eviction: {e2}")
            return None, {}

    def _detect_self_modification(self, tool_calls: list) -> bool:
        """Check if any tool call modified flagscale_agent/ source files.
        
        Detects write_file/edit_file operations targeting the agent's own code,
        which require a /reload to take effect.
        """
        SELF_PATHS = ("flagscale_agent/", "flagscale_agent\\")
        FILE_TOOLS = ("write_file", "edit_file")
        
        for tc in tool_calls:
            name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            if name not in FILE_TOOLS:
                continue
            # Extract path from tool call input
            inp = tc.get("input", {}) if isinstance(tc, dict) else getattr(tc, "input", {})
            if isinstance(inp, str):
                try:
                    import json
                    inp = json.loads(inp)
                except (json.JSONDecodeError, TypeError):
                    continue
            path = inp.get("path", "") if isinstance(inp, dict) else ""
            # Check if path touches agent source
            if any(seg in path for seg in SELF_PATHS):
                return True
        return False

    def _save_recovery_state(self):
        """Save current progress to plan notes before compaction.

        This ensures the agent can recover its working state after context
        is compacted, preventing the post-compaction death loop.
        """
        d = self.deps
        task_plan = getattr(d, "task_plan", None)
        if not task_plan:
            return

        active = task_plan.get_active()
        if not active:
            return

        # Find current "doing" step
        steps = active.get("steps", [])
        doing_steps = [s for s in steps if s.get("status") == "doing"]
        if not doing_steps:
            return

        step = doing_steps[0]
        step_id = step.get("id")

        # Build recovery context from recent history
        recent_msgs = d.history.get_messages()[-6:]  # Last 3 exchanges
        recovery_lines = []
        for msg in recent_msgs:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    for line in content.split("\n"):
                        if line.strip() and not line.startswith("["):
                            recovery_lines.append(line.strip()[:200])
                            break

        if recovery_lines:
            recovery_note = "RECOVERY: " + " | ".join(recovery_lines[-3:])
            try:
                task_plan.update_step(step_id, "doing", recovery_note)
            except Exception:
                pass

    def _get_last_assistant_text(self) -> str:
        """Get the text content of the last assistant message."""
        d = self.deps
        for msg in reversed(d.history.get_messages()):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = [b.get("text", "") for b in content 
                             if isinstance(b, dict) and b.get("type") == "text"]
                    return "".join(texts)
        return ""

    def _generate_continuation(self) -> str:
        task_plan = getattr(self.deps, "task_plan", None)
        if task_plan:
            active = task_plan.get_active()
            if active:
                pending = [
                    s for s in active.get("steps", [])
                    if s.get("status") not in ("done", "skipped")
                ]
                if pending:
                    step = pending[0]
                    return f"Continue with step: {step.get('title', step.get('description', ''))}"
        return "Continue with the next step."

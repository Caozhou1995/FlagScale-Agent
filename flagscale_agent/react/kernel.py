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

import re
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from flagscale_agent.react.guard import GuardContext, GuardRegistry, GuardVerdict
from flagscale_agent.react import display
# Bind the sentinel helpers into the kernel namespace: tests monkeypatch the
# display MODULE for guard_inject/completion_signal routing, and calling them
# through that mock would break the completion detection itself.
from flagscale_agent.react.display import has_override_completion, trailing_sentinel


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
    time_budget_stats_fn: Callable | None = None  # () -> {elapsed, budget, remaining, pct} | None
    # Hard wall-clock stop: when the injected budget stats report remaining <= 0,
    # the loop stops BEFORE the next LLM call, prints a timeout notice and sets
    # stop_reason="time_budget_exhausted". None (or a None return) = no hard stop,
    # preserving the old soft-warning-only behavior.


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

    # Max consecutive completion-gate blocks with zero intervening progress
    # before the loop gives up (livelock breaker). Small: a genuinely diligent
    # agent creates a plan or supplies an override on the FIRST block, so any
    # value >1 gives it room to react while still catching a stuck re-emit loop
    # long before max_iter.
    MAX_CONSECUTIVE_COMPLETION_BLOCKS = 5
    # In-stream thinking wall: mirrors agent's IN_STREAM_THINKING_WALL_SEC
    # (240s, agent.py L1279). Used by the endgame arming arithmetic so the
    # forced-call window is guaranteed to contain a stream boundary.
    STREAM_THINKING_WALL_SEC = 240
    # ── Endgame force_tool cadence: 1 armed round, then 1 released round ──
    # Arming the endgame window PERMANENTLY (every request from the arm point
    # to the wall carries tool_choice="required") makes the turn's own exit
    # structurally unreachable, because that constraint declares a prose-only
    # reply illegal (providers/base.py docstring) while 10.5 of the prompt
    # orders the agent to "EMIT THE MARKER IN A TEXT-ONLY RESPONSE, no tool
    # call in it at all". Measured on kv-store-grpc trial UNKgJ4N
    # (jobs/2026-11-05__06-48-27): 900.18s budget -> arm_thr 58.34% -> forced
    # for the last 375.0s; 126 assistant rounds, ALL 126 carried tool_use,
    # text-only rounds = 0, literal [TASK_COMPLETE] in content = 0 — while 29
    # rounds' thinking said "emit the marker now, text-only" (all 29 after the
    # first TimeBudget injection; 0 of the 27 before it). Control:
    # pypi-server (182.39s, never reached the arm point) closed with a
    # single ['text'] block ending in [TASK_COMPLETE] — the only trial in the
    # batch that emitted the marker. So the marker was not withheld by the
    # model; it was unavailable at the request layer.
    # The original permanent arming was not wrong in intent: prose alone was
    # measured to burn the wall to zero bytes on disk. The fix keeps that
    # pressure but makes it SELF-LIMITING — force one round, release the next.
    # Worst case the agent still gets an action every two rounds (the anti-
    # burn property), and within two rounds it always has a legal opening to
    # close the turn. 1 = alternate; 0 disables the release (permanent arming,
    # the pre-fix behaviour).
    ENDGAME_RELEASE_EVERY = 2
    # Saturation nudge: from this consecutive-saturated-call count on, the
    # kernel DROPS the saturated filler message from history and injects a
    # directive that FORBIDS continuing the text stream (hard convergence).
    # 1 = first saturated text-only call already triggers it: a saturated
    # call can cost several minutes of wall clock, and its filler is re-sent
    # as input on every later call. 0 disables the mechanism entirely.
    OUTPUT_SATURATION_NUDGE_AT = 1
    # A call is "saturated" when its output tokens reach this fraction of the
    # configured max_output_tokens (finish_reason == length).
    OUTPUT_SATURATION_FRACTION = 0.95
    # Degenerate-repeat loop breaker. The counters above bound the number of
    # CONTINUATIONS, but they are reset on several paths (a tool call, a
    # cap-retry, an empty-output retry), so a stream that keeps REGENERATING
    # THE SAME FEW TEXTS can outlive every counter: observed on a real trial as
    # a PERIOD-4 cycle ("Let me write the file." / "Let me write the content." /
    # "Let me write it." / "OK.") repeated 405 times — 1600+ log lines, zero
    # tool calls, no deliverable written. A count-based cap cannot see that, and
    # neither can an adjacency check (a period-4 cycle never repeats back to
    # back — verified: an e2e run with adjacency-only detection ran to
    # max_continuations without firing). What DOES see it is the DIVERSITY of
    # the recent window: W consecutive tool-free responses drawn from at most
    # DISTINCT_MOST distinct texts is a dead loop. Measured on the real
    # trajectory (tee'd log grouped into responses by blank-line separation:
    # 9907 responses, 627 distinct overall, but the degenerate region drawing
    # from only 8 filler phrases — "Let me write it." x2320, "Let me write the
    # content." x2310, "Let me write the file." x2309, "OK." x1153, plus four
    # "OK, writing…" variants): DISTINCT_MOST=6 over a 12-response window fires
    # at response #470 of that log (~5% in), while genuine work introduces new
    # distinct texts every few steps and stays far above the bound. Calibrated
    # on the real log AND on a uniform-cycle stress case (a balanced 8-phrase
    # cycle is the same dead loop but a 12/6 window sees 8 distinct and misses
    # it): 16/8 fires on both, fires on the real log at response #482 of 9907
    # (~4.9%), and still never fires on genuine work with 16+ distinct texts.
    # 0 disables.
    DEGENERATE_REPEAT_WINDOW_CALLS = 16
    DEGENERATE_REPEAT_DISTINCT_MOST = 8
    # Bytes of the assistant text used for the fingerprint. Short filler
    # phrases ("OK.", "Let me write it.") are fully covered; for a long
    # response the TAIL is the part that repeats.
    DEGENERATE_REPEAT_WINDOW = 400
    # Text-only-streak convergence: a filler loop need NOT saturate a single
    # call — the model can emit many SHORT text-only calls in a row ("Let me
    # go." / "OK." / "Writing."), none near the token ceiling, and never
    # saturating (observed: 25 API calls, only 1 saturated, the rest <4k
    # tokens, filling the whole wall). From this consecutive-text-only count
    # on (with NO tool executed in between), the kernel DROPS the accumulated
    # no-action text from history and injects the same HARD convergence
    # directive the saturation path uses. 0 disables it.
    TEXT_ONLY_STREAK_HARD_AT = 6
    # Auto-append fallback (honest hand-back): the whole ladder above only
    # re-prompts, so a model that NEVER self-corrects burns the entire wall
    # (observed: every round ends "no stop signal" and the loop continues).
    # From this many consecutive tool-free, signal-free responses on, the
    # kernel ends the turn ITSELF by appending a harness-side
    # [NEED_USER_INPUT] — deliberately NOT a fabricated [TASK_COMPLETE],
    # which would claim a verified deliverable the model never asserted.
    # The appended message is the kernel's own history entry, so the model's
    # text is never rewritten. 0 disables.
    AUTO_APPEND_NEED_INPUT_AT = 12

    # Edit-churn stop-rule (the FINAL ladder rung, before the text-only
    # fallback). After this many CONSECUTIVE edit-family calls with NO
    # successful non-edit execution interleaved, inject a directive naming
    # the churn and (when a plan with a forbidden list is active) the actual
    # prohibition(s), then END the turn with an honest [NEED_USER_INPUT]
    # hand-back — never a fabricated [TASK_COMPLETE]. The detector requires
    # BOTH conditions so a legitimate TDD cadence (edit, passing test, edit,
    # passing test) never triggers. Fires once per streak (the counter
    # resets after firing); a model that answers the directive with real
    # evidence keeps working. 0 disables.
    FORBIDDEN_STOP_AFTER = 30
    # Tool names treated as "edit" by the churn detector.
    FORBIDDEN_EDIT_TOOLS = frozenset({"edit_file", "write_file"})
    # How many times the rung may fire on consecutive churn streaks before
    # it stays quiet (directive-fatigue guard: a model ignoring the directive
    # twice will not be re-nudged forever).
    FORBIDDEN_STOP_MAX_FIRES = 2

    def __init__(self, deps: KernelDeps):
        self.deps = deps
        self._interrupted = False
        self._continuation_count = 0
        self._consecutive_completion_blocks = 0
        # M5: consecutive saturated-call streak (output hit the token ceiling
        # with no tool call and no stop signal). Reset by a tool turn, a
        # stop-signal turn, or an unsaturated text-only turn.
        self._consecutive_saturated = 0
        # Auto-append fallback state: set the moment the kernel ends a turn
        # harness-side (see AUTO_APPEND_NEED_INPUT_AT). Checked at the end of
        # run_turn to avoid re-appending on every later turn of a truly
        # stuck stream — one hand-back per incident.
        self._auto_appended_this_turn = False
        # Edit-churn stop-rule state (FORBIDDEN_STOP_AFTER). Counted across
        # the whole turn; reset by any successful non-edit execution.
        self._edit_churn_streak = 0
        self._forbidden_stop_fires = 0
        # Structured execution ledger for the current turn: one entry per tool
        # call that ACTUALLY ran (blocked calls excluded), each carrying the
        # `exec_error` flag so an evidence gate can tell "something executed"
        # from "something executed successfully". Surfaced to guards as
        # ctx.recent_tool_history. Reset at the start of every run_turn().
        self._recent_tool_history: list[dict] = []
        # calls a guard PREVENTED from running, kept as their own
        # record. Deliberately NOT mixed into _recent_tool_history — that ledger
        # means "this executed" and first_e2e_run.py reads it as evidence, so a
        # blocked entry there would forge progress. This list exists so an
        # auditor can distinguish "never attempted" from "attempted and blocked":
        # Audit #83's central finding rested on one shell call that a guard
        # swallowed, visible only because the guard banner happened to be the
        # record. Reset per turn with the execution ledger.
        self._blocked_tool_calls: list[dict] = []
        # Live per-turn iteration counter, updated at the TOP of every loop
        # iteration so a consumer INSIDE the turn (a guard asking "how many
        # actions have I already spent?") sees a real number. KernelResult only
        # reports the total AFTER the turn ends, which is too late for any
        # in-turn admission decision.
        self.iterations_live: int = 0

    # Cumulative idle think-interval pacing (see run_turn gate): if this much
    # wall-clock accumulates across consecutive think round-trips with NO tool
    # call between them, the kernel aborts the stream and injects a convergence
    # nudge. Keyed on TIME since the last tool call — never on one stream's
    # token size, which is how six individually-legal streams spent ~620s with
    # zero state produced (write-compressor v11/v12). 240s ≈ the elapsed gap
    # between the observed rungs (159s + 112s) so the intervention lands after
    # the second legal stream, not the first.
    # Measured across 7 trials (job 12-14-17): per-round latency p50 is 3-16s, and
    # the agent produces many SHORT idle gaps (3-8 round streaks, 36-141s total)
    # rather than one long absence. A 240s/3-round floor therefore left the gate
    # silent in 6 of 7 runs even though no-tool round-trips were 49% of all LLM
    # time (2018s of 4122s). 90s over the 2-round floor matches the observed
    # median shape; the cap budget plus the exhausted escape valve keep it bounded.
    IDLE_THINK_INTERVAL_SEC = 90
    # Minimum consecutive no-tool round-trips before the idle gate may fire.
    IDLE_THINK_MIN_ROUNDS = 2

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
        # Circuit breaker for the completion gate: counts consecutive
        # completion-gate blocks with no intervening progress (no plan created,
        # no normal tool executed). A single-shot agent that keeps re-emitting a
        # bare [TASK_COMPLETE] with neither a plan nor an _override_reason would
        # otherwise livelock — the gate correctly blocks every time and the loop
        # `continue`s without incrementing _continuation_count, so it spins all
        # the way to max_iter (thousands of identical blocks). This breaks out
        # once the agent has clearly stalled on the gate.
        self._consecutive_completion_blocks = 0
        self._recent_tool_history = []  # reset the per-turn execution ledger
        self._blocked_tool_calls = []   # reset the per-turn blocked-call record
        # Auto-append fallback: one hand-back per incident. A new user turn
        # re-arms it (a genuinely stuck model can hit it again next turn,
        # which is correct — each turn then costs at most the threshold).
        self._auto_appended_this_turn = False
        # NOTE: _consecutive_text_only deliberately does NOT reset here — the
        # threshold counts tool-free responses, and consecutive run_turn()
        # calls with no tool call in between are the same degenerate stream.
        # Idle think-interval accounting: when did a tool last actually run, and
        # how many consecutive think round-trips have passed since. Each
        # run_turn() is one user turn, so the debt does not carry across turns;
        # the nudge/retry loop iterates INSIDE this call and is not affected.
        self._last_toolcall_ts = None
        self._idle_think_rounds = 0
        # Cap-escalation streak: consecutive runtime-cap trips in THIS turn that
        # were answered with more pure thinking. Reset per turn.
        self._cap_text_streak = 0
        # Squeeze override (user directive, 2026-10-30): when set to a positive
        # token count, the agent's next stream(s) run with the thinking cap
        # SQUEEZED to this budget instead of the configured thinking_budget.
        # Set by the HARD STOP rung (below); cleared when a write/edit finally
        # produces state on disk (progress evidence) — that restores the
        # normal budget.
        self._thinking_cap_override = None
        self._turn_start_ts = time.time()
        self.iterations_live = 0
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

        # Counts loop iterations spent inside the endgame force_tool window,
        # so the arming can alternate (see ENDGAME_RELEASE_EVERY). Initialized
        # per TURN, not per iteration: it is the cadence phase, not a counter
        # of work.
        _endgame_rounds = 0

        try:
            for iteration in range(max_iter):
                # Publish the live count FIRST: every in-turn consumer (a guard
                # deciding whether the remaining wall can still cover a wrap-up)
                # must see how many action rounds THIS turn has already spent.
                self.iterations_live = iteration + 1
                if self._interrupted:
                    break

                # ── Hard wall-clock timeout ──
                # Checked at the top of every iteration, BEFORE the next LLM
                # call: when the injected per-turn budget is exhausted, print
                # a timeout notice and leave the turn. Already-completed tool
                # calls and their results stay in history; the wrap-up / user
                # turn the model would have produced is simply not emitted.
                # Note: once this fires at <=0 the soft TimeBudgetGuard's
                # at/after-100% "clear path to wrap up" is pre-empted — the
                # 90% block is the last chance the model gets to wrap up.
                _tb_stats = None
                if d.time_budget_stats_fn is not None:
                    try:
                        _tb_stats = d.time_budget_stats_fn()
                    except Exception:
                        _tb_stats = None
                # Only a dict carrying a NUMERIC remaining counts. A non-dict
                # return, a missing key, or a None/str/int-incompatible value
                # must never raise here — it just means "no deadline known",
                # so the loop runs unchanged (same as a None return / raise).
                _tb_remaining = _tb_stats.get("remaining") if isinstance(_tb_stats, dict) else None
                if isinstance(_tb_remaining, (int, float)) and _tb_remaining <= 0:
                    display.warn(
                        f"TIME BUDGET EXHAUSTED: per-turn wall-clock budget "
                        f"({_tb_stats.get('budget', 0):.0f}s) used up "
                        f"(elapsed {_tb_stats.get('elapsed', 0):.0f}s). "
                        f"Stopping this turn (hard timeout).")
                    result.stop_reason = "time_budget_exhausted"
                    break

                # ── Endgame posture (structural, not prose) ──
                # Measured failure (trial 4CHXYAh, 2026-10-30): a 900s budget,
                # 5 tool calls total, and in the final 19 seconds the model
                # wrote "I must bank NOW" and spent them deliberating HOW to
                # call write_file vs shell — the turn died at 100% with zero
                # bytes on disk. Prose checkpoints (the soft guard's 90%
                # block) only gate OBSERVATION calls; nothing structurally
                # constrains pure reasoning, and reasoning is exactly what
                # the endgame burns. So from >= 85% of budget spent to the
                # end of the turn, EVERY remaining request is constrained
                # the same way the cap-escalation ladder constrains a
                # runaway thinker: the next stream must carry a tool call
                # (prose-only is not a legal completion) and its thinking
                # budget is squeezed so server-forced thinking cannot eat
                # the round before the tool call is reached. This is
                # budget-driven, not cap-driven, so it stays armed — the
                # wall itself is the exit, and a turn that ends with only
                # reasoning on record scores zero no matter how good the
                # reasoning was.
                #
                # Stream-boundary arithmetic (measured, dna-assembly trial
                # srrzyo4 2026-10-31): the arm point is checked HERE, at the
                # top of the loop, but the 85% crossing usually happens while
                # a thinking stream is ALREADY in flight — that stream is
                # unforced and only ends when its own in-stream wall
                # (240s) or the budget wall kills it. For a 900s budget the
                # endgame window after 85% is 135s < 240s, so NO stream
                # boundary ever lands inside the window: the forced call is
                # attached to a stream that dies at the wall with the tool
                # call never reached (observed: 🔧=0 across the whole trial,
                # 92% nudge printed 9s before death). The fix is to arm
                # EARLY ENOUGH that a stream boundary is guaranteed before
                # the wall: the arm threshold must be pulled earlier by the
                # stream wall as a share of budget, i.e.
                #   threshold = max(50, 85 - (stream_wall / budget) * 100)
                # so the worst case is one stream that starts at the arm
                # point and ends at most stream_wall seconds later — with
                # (threshold/100)*budget - stream_wall seconds of window
                # left for the forced call to run. For 900s that is
                # 85 - 26.7 = 58.3% (forced call gets ~135s guaranteed);
                # for 3600s it is 85 - 6.7 = 78.3%.
                _tb_pct = (
                    _tb_stats.get("pct") if isinstance(_tb_stats, dict) else None)
                _tb_budget = (
                    _tb_stats.get("budget") if isinstance(_tb_stats, dict) else None)
                _arm_thr = 85.0
                if (isinstance(_tb_budget, (int, float)) and _tb_budget > 0):
                    _arm_thr = max(
                        50.0,
                        85.0 - (self.STREAM_THINKING_WALL_SEC
                                / float(_tb_budget)) * 100.0)
                if isinstance(_tb_pct, (int, float)) and _tb_pct >= _arm_thr:
                    # SELF-LIMITING arming (see ENDGAME_RELEASE_EVERY): force
                    # one round, then release one, so the turn always retains a
                    # legal text-only opening in which the marker can be
                    # emitted. Permanent arming made the exit unreachable —
                    # measured, kv-store-grpc UNKgJ4N: 126/126 rounds carried
                    # tool_use, 0 text-only, 0 literal [TASK_COMPLETE], while
                    # 29 rounds' thinking said "emit the marker now, text-only".
                    _endgame_rounds += 1
                    _release = (
                        self.ENDGAME_RELEASE_EVERY > 0
                        and _endgame_rounds % self.ENDGAME_RELEASE_EVERY == 0
                    )
                    if not _release:
                        self._force_tool_next = True
                        self._thinking_cap_override = 1024
                    else:
                        d.display.warn(
                            "Endgame force_tool RELEASED for this round — a "
                            "text-only reply (marker) is a legal completion "
                            "again; use it if the work is done."
                        )

                # Guard reset happens per-turn (at line 101), not per-iteration
                d.judge.reset_turn()

                schemas = d.get_schemas_fn()

                # ── Pre-guard checks ──
                ctx = self._build_ctx(tool_name="", tool_args={}, tool_result=None)
                verdict = d.guard_registry.check_pre(ctx)
                if verdict is not None:
                    blocked = self._apply_verdict(verdict, pre=True)
                    if blocked:
                        # Livelock breaker: the completion gate reads the LAST
                        # assistant message via _get_last_assistant_text(). Once a
                        # bare [TASK_COMPLETE] (no plan / no override) lands in
                        # history, this top-of-loop pre-guard re-fires the gate
                        # EVERY iteration on that stale text — and `continue`s
                        # before the LLM is ever called again, so the agent can
                        # never react. Left unchecked it spins to max_iter
                        # (~2000 identical blocks). Count consecutive completion-
                        # gate blocks and give up once clearly stalled.
                        if verdict.reason == "single_shot_completion_without_plan":
                            self._consecutive_completion_blocks += 1
                            if (self._consecutive_completion_blocks
                                    >= self.MAX_CONSECUTIVE_COMPLETION_BLOCKS):
                                result.stop_reason = (
                                    f"completion_gate_livelock: {verdict.reason}")
                                break
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
                    # ── Structural channel enforcement ──
                    # Text nudges ("MUST emit a tool call") have been measured
                    # to produce compliance TALK and then another full stream
                    # of reasoning: the directive is acknowledged and ignored
                    # because prose is still a legal completion. When the
                    # escalation ladder has given up on wording, it sets this
                    # flag, and the NEXT request is sent with a provider-level
                    # must-call-a-tool constraint, so a prose-only reply is
                    # not a legal answer at all.
                    _force = bool(getattr(self, "_force_tool_next", False))
                    self._force_tool_next = False
                    # The squeeze override must ride the SAME channel as
                    # force_tool: the kernel sets it on ITSELF (L711) but the
                    # agent reads getattr on the AGENT instance (L2418) — two
                    # different objects, so an instance attribute never
                    # crosses. Measured (round 6 regex-chess trial SbvVjVR):
                    # cap#3 printed "0 token cap" (cap_budget=0 branch) even
                    # though the HARD STOP rung had fired — the override was
                    # set on the kernel and lost. Forwarding it as a call
                    # kwarg makes the agent receive the squeezed budget.
                    _squeeze = getattr(self, "_thinking_cap_override", None)
                    self._thinking_cap_override = None
                    # The fallback lambda must FORWARD kwargs: a 2-arg lambda
                    # makes the 3-arg forced call raise TypeError and the
                    # degrade branch silently drops the constraint (measured:
                    # flag set, request still prose-only). provider.chat_stream
                    # accepts force_tool, so forwarding keeps both paths live.
                    _call = d.call_llm_fn or (
                        lambda m, s, **kw: d.provider.chat_stream(m, s, **kw))
                    if _force:
                        _kw = {"force_tool": True}
                        if _squeeze:
                            _kw["thinking_cap_override"] = _squeeze
                        try:
                            response, usage = _call(messages, schemas, **_kw)
                        except TypeError:
                            # Injected callable / provider without the flag:
                            # degrade to the plain call rather than crashing.
                            response, usage = _call(messages, schemas)
                    else:
                        response, usage = _call(messages, schemas)
                except KeyboardInterrupt:
                    d.display.interrupted()
                    self._interrupted = True
                    break
                except Exception as e:
                    # Context overflow should be prevented by ContextPressureGuard
                    # which prompts LLM to call evict/hard_reset before reaching this point
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
                    # ── Idle think-interval gate ──
                    # Shared with the tool-call branch below (a round whose
                    # every call was BLOCKED is just as unproductive as a round
                    # with no calls at all), so the decision lives in
                    # _idle_nudge() rather than inline here.
                    _idle_state = self._idle_nudge(response, d, result)
                    if _idle_state == "exhausted":
                        break
                    if _idle_state == "fired":
                        iteration += 1
                        continue
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
                        # Provider-agnostic reasoning detection: BOTH providers
                        # normalize reasoning into response["thinking"]
                        # (anthropic: thinking blocks; openai: reasoning_content).
                        # Do NOT rely on response["reasoning_only"] — that flag is
                        # only ever set by openai_provider, so an anthropic-compat
                        # endpoint (e.g. GLM) would never take the reasoning path.
                        thinking_text = response.get("thinking") or ""
                        was_reasoning_only = bool(thinking_text.strip())
                        was_thinking_capped = bool(response.get("thinking_capped"))
                        # The stream was aborted BY US for an intra-stream
                        # degenerate repeat. That is not a pause and not an
                        # empty output: the reasoning text IS the filler. Treat
                        # it as its own event so we neither re-inject the
                        # filler (which re-feeds the loop) nor charge the
                        # empty-output budget that ends the turn.
                        was_intra_degenerate = bool(response.get("intra_degenerate"))
                        if was_intra_degenerate:
                            was_thinking_capped = False
                        # ── Idle think-interval gate (cumulative, not per-stream) ──
                        # The convergence machinery above is keyed on ONE stream's
                        # TOKEN size (cap budget / saturation). Measured failure
                        # (write-compressor, v11 job 02-33-28 and v12 03-12-08):
                        # 73% of the log and ~620s of wall clock were spent in SIX
                        # separate think round-trips, each under the 16384-token
                        # cap, with ZERO tool calls between them — the first file
                        # write landed at 76% of the budget. Every stream was
                        # individually legal, so no threshold fired; the kill was
                        # cumulative time with no state produced. The guards cannot
                        # catch this (check_pre returns early with no tool call to
                        # intercept), so the pacing decision has to live here.
                        # Judge by time+round-trips since the last tool call, never
                        # by token count.
                        # NOTE: the idle think-interval decision is made in the
                        # tool-free block ABOVE, before this empty-output branch.
                        # It lives there because a tool-less response WITH visible
                        # text must also be judged (measured: 218.4s / 30,284
                        # tokens in one such round-trip, gate 100% inert). The
                        # counter (_idle_think_rounds), the re-arm
                        # (_last_toolcall_ts) and the budget (_cap_retries) are
                        # therefore owned exclusively by that block; duplicating
                        # the increment here double-counts each round-trip and
                        # trips the gate a round early.
                        # ── Output saturation with no visible text ──
                        # A reasoning model can burn the ENTIRE output budget on
                        # one monolithic thinking stream, then be cut off before
                        # emitting any visible text. Structurally this is the SAME
                        # event as a runtime thinking cap (a stream hit the
                        # ceiling), but `thinking_capped` is only ever set by
                        # providers that surface it -- an anthropic-compat
                        # endpoint does not (observed: 0 cap trips across a
                        # whole trial whose single call reported ↓65,536).
                        # Without this, the saturation fell into the
                        # "reasoning produced content" branch, which re-injects
                        # the FULL ~20k-token reasoning with a RESUME directive
                        # ("continue where you left off") -- actively feeding the
                        # monolithic stream that exhausted the budget, and the
                        # M5 saturation guard is unreachable from that path (it
                        # sits after the `continue`). Detect it by the only
                        # signal that is always available: the token counter.
                        _sat_max = getattr(d.config, "max_output_tokens", 0) or 0
                        was_saturated = bool(
                            _sat_max > 0
                            and out_tok >= 0.95 * _sat_max
                            and not assistant_text.strip()
                        )
                        if was_saturated:
                            was_thinking_capped = True
                        # SEPARATE BUDGETS: a cap-trip is a PACING event, not a
                        # failure — the model still produced (thinking) output
                        # and each retry is already bounded by the cap itself
                        # (~cap_budget thinking tokens per call). Budgeting it
                        # like a genuine empty output killed real runs: e.g.
                        # Observed: cap fired 4x in one turn,
                        # retries 1/3..3/3 exhausted the shared budget, and the
                        # 4th cap-trip stopped the trial (reward 0) with the
                        # full solution design sitting in the discarded
                        # reasoning. Cap-retries get their own generous counter
                        # (per turn) instead of the 3-strike empty budget.
                        cap_retries = getattr(self, "_cap_retries", 0)
                        empty_retries = getattr(self, "_empty_output_retries", 0)
                        intra_retries = getattr(self, "_intra_degen_retries", 0)
                        if was_intra_degenerate:
                            # Own generous-but-bounded counter: the abort is a
                            # healthy intervention, so it must NOT consume the
                            # empty-output budget. Only a model that degenerates
                            # over and over ends the turn.
                            if intra_retries >= 4:
                                # A staircase exhausted by pure degeneration must
                                # still LAND what was already derived. Measured
                                # (polyglot-c-py, 2026-11-10 job 00-11-22): the
                                # four recoveries fired 1/4..4/4 inside 8 minutes
                                # — far under the 900s wall — and the turn ended
                                # at 3 tool calls with the deliverable never
                                # written; reward 0 despite a 0.955 historical win
                                # rate. So the last rung is not "give up": it
                                # forces ONE write-through round (same mechanism
                                # the cap branch uses at L721), and only ends the
                                # turn if that round degenerates too.
                                if not getattr(self, "_degen_flush_armed", False):
                                    self._degen_flush_armed = True
                                    self._intra_degen_retries = 0
                                    self._force_tool_next = True
                                    d.display.warn(
                                        "Degenerate staircase exhausted — forcing "
                                        "one write-through round before ending the "
                                        "turn."
                                    )
                                else:
                                    self._degen_flush_armed = False
                                    self._intra_degen_retries = 0
                                    result.stop_reason = "intra_degenerate_max_retries"
                                    break
                            self._intra_degen_retries = intra_retries + 1
                            d.display.warn(
                                "Degenerate repeat inside the reasoning stream "
                                f"(recovery {self._intra_degen_retries}/4) — "
                                "demanding a tool call..."
                            )
                        elif was_thinking_capped:
                            if cap_retries >= 50:
                                self._cap_retries = 0
                                result.stop_reason = "thinking_cap_max_retries"
                                break
                            self._cap_retries = cap_retries + 1
                            # Escalating cap recovery. Measured failure
                            # (write-compressor v12.2, 2026-10-26 04-03-20,
                            # trial EK83dj8): the in-stream wall-clock cap DID
                            # fire (two 325.3s/316.6s pure-thinking streams cut),
                            # but the ONE-SHOT convergence nudge was ignored —
                            # the model re-read the same 53KB reasoning and spent
                            # another 316.6s thinking, ending the turn with ZERO
                            # tool calls and no deliverable (verifier 3 failed).
                            # The saturation path below already carries the fix
                            # (bounded recap + repeat counter + HARD STOP); this
                            # endpoint never trips `was_saturated` (out_tok stays
                            # below 95% of max_output_tokens), so the escalation
                            # has to be driven by the cap count itself.
                            self._cap_text_streak = getattr(self, "_cap_text_streak", 0) + 1
                            # ── ARM THE STRUCTURAL CHANNEL HERE TOO ─────────
                            # Measured failure (write-compressor, 2026-11-09
                            # 22:56 batch, trial H8KxFia): this branch ran 7
                            # times ("cap-retry 1/4/5/7" + idle-retry 2/3/6)
                            # and EVERY one of them produced ↓0 output tokens:
                            # the warning alone re-ran a full stream that was
                            # cut at the same wall, then produced nothing.
                            # Cause: the escalation that arms force_tool lives
                            # in the NUDGE chain further down, which this
                            # early-return branch never reaches — so although
                            # the model kept being told "forcing convergence",
                            # `_force_tool_next` was never set and the next
                            # request stayed a legal prose-only completion.
                            # Arm it on the FIRST cap of a pure-thinking stream
                            # (this branch is reached only when assistant_text
                            # was empty). The squeeze rides the same call kwarg
                            # as force_tool (see the _squeeze forwarding at the
                            # call site), so a small floor cuts the next stream
                            # early enough for the forced call to land.
                            self._force_tool_next = True
                            self._thinking_cap_override = 1024
                            d.display.warn(f"Thinking reached runtime cap (cap-retry {self._cap_retries}), forcing convergence...")
                        elif empty_retries < 3:
                            self._empty_output_retries = empty_retries + 1
                            if was_reasoning_only:
                                d.display.warn(f"Reasoning produced content but no visible output (retry {empty_retries + 1}/3), continuing...")
                            else:
                                d.display.warn(f"Empty LLM output (retry {empty_retries + 1}/3), auto-continuing...")
                        else:
                            # Exhausted empty-output budget (genuine empty /
                            # reasoning-only response, nothing to re-inject
                            # into a working retry loop).
                            self._empty_output_retries = 0
                            result.stop_reason = "empty_output_max_retries"
                            break

                        # ── Retry budget passed: remove the dead assistant
                        # message, re-inject reasoning, continue the turn ──
                        # NOTE: get_messages() returns a shallow copy — popping
                        # that copy is a no-op on the real history (the
                        # thinking block would stay in the LLM prompt). Use
                        # pop_last_assistant() which mutates _messages for real.
                        d.history.pop_last_assistant()
                        # Inject a targeted nudge
                        if was_intra_degenerate:
                            # Do NOT reproduce the degenerate text: it is pure
                            # filler and re-sending it invites the same loop.
                            # Demand one concrete tool call instead.
                            # Measured failure (write-compressor, 2026-10-25
                            # 12-16-04 batch): the model had ALREADY drafted the
                            # full encoder (~2000 chars) inside the aborted
                            # thinking stream, then spent 4300+ lines in a 4-line
                            # cycle ("Let me write / OK / I'll write the file /
                            # Let me go") — the generic "one tool call" menu
                            # left it PLANNING to write instead of writing. The
                            # nudge must name the ONE action that converts
                            # thought into state: put the drafted file on disk
                            # NOW, unconditionally, before any further design.
                            _intra_n = getattr(self, "_intra_degen_retries", 1)
                            nudge = (
                                "[system: your reasoning stream entered a degenerate "
                                "repeat loop and was aborted mid-stream. The text you "
                                "produced was filler (short phrases repeated with no "
                                "progress), so it has been discarded.\n"
                                f"This is occurrence #{_intra_n} in this turn.\n"
                                "STOP designing. Your next message MUST contain ONE tool "
                                "call, and the ONLY acceptable first call is `write_file`: "
                                "take the solution you have ALREADY drafted in your "
                                "reasoning (it is complete enough — you were about to "
                                "write it), paste it in full, and put it on disk at the "
                                "delivery path the task names (the exact filename under "
                                "/app the task text specifies). Do not re-derive, do not "
                                "improve it first, do not explain — WRITE THE FILE as "
                                "your very next tool call. If it later proves wrong, "
                                "edit the file on disk; an imperfect file that exists "
                                "beats a perfect draft that never lands.\n"
                                "Only if you truly have nothing to write yet, the "
                                "fallback is ONE `shell` command that tests your "
                                "strongest remaining hypothesis — but prefer the write."
                            )
                        elif was_thinking_capped:
                            # The runtime cap aborted a monolithic thinking
                            # stream. Same re-injection as the resume path
                            # (assistant thinking blocks are dropped by the
                            # GLM endpoint and the aborted stream has no
                            # valid signature), but the directive is
                            # CONVERGENCE, not resume: the model must land
                            # its next concrete step now — segmented
                            # reasoning beats one 20-minute thought.
                            #
                            # Saturation case: the reasoning ALREADY ate the
                            # whole output budget once. Re-injecting all of it
                            # spends the next call's input budget re-reading the
                            # same dead stream and invites an identical
                            # continuation, so keep only a bounded tail as a
                            # recap and count the repeat.
                            if was_saturated:
                                self._saturated_text_streak = (
                                    getattr(self, "_saturated_text_streak", 0) + 1
                                )
                                _recap = thinking_text[-4000:] if len(thinking_text) > 4000 else thinking_text
                                _repeat = getattr(self, "_saturated_text_streak", 0)
                                nudge = (
                                    "[system: your reasoning consumed the ENTIRE output budget "
                                    f"({out_tok} of {_sat_max} tokens) and was cut off before producing "
                                    "any visible text or tool call. This is not a pause -- the stream "
                                    "hit the hard ceiling and every further attempt to continue it will "
                                    "be cut at the same point.\n"
                                    f"This is occurrence #{_repeat} in this turn.\n"
                                    "DO NOT reason further about the problem. Do exactly ONE of these NOW, "
                                    "in your very next output:\n"
                                    "  1. Call the single tool that tests your strongest remaining "
                                    "hypothesis (write_file / shell / web_fetch), or\n"
                                    "  2. Write the deliverable with what you already know, or\n"
                                    "  3. Output [NEED_USER_INPUT] with the specific question blocking you.\n"
                                    "Any output that consists only of reasoning will be discarded again.\n\n"
                                    f"<recap_of_discarded_reasoning_tail>\n{_recap}\n"
                                    "</recap_of_discarded_reasoning_tail>"
                                )
                                if _repeat >= 2:
                                    nudge = (
                                        "[system: HARD STOP. You have now saturated the output budget "
                                        f"{_repeat} times in this turn without emitting a single tool call "
                                        "or any visible text. Reasoning cannot solve this -- it is "
                                        "mechanically cut off. Your next message MUST contain a tool call. "
                                        "Start with the cheapest useful action: `shell` to inspect the "
                                        "environment, or `write_file` to persist what you already derived. "
                                        "Do not explain, do not plan, do not reason -- emit the tool call."
                                    )
                            else:
                                _cap_n = getattr(self, "_cap_text_streak", 1)
                                _cap_tail = thinking_text[-4000:] if len(thinking_text) > 4000 else thinking_text
                                # Pure-thinking streams (no visible text) are the
                                # one population the soft rung never converted:
                                # round 7 (regex-chess RK7GQoK, 2026-10-29) —
                                # cap-retry 1 soft nudge -> 172.5s more pure
                                # thinking (idle-retry 2 fired only after 372s
                                # tool-free) -> 240.3s more thinking; 612s = 68%
                                # of the 900s budget burned in 3 zero-output
                                # retries before the structural rung arrived.
                                # This branch is reached ONLY when the stream
                                # carried no visible text, so upgrading the FIRST
                                # cap here costs the cooperative-model headroom
                                # nothing (a model that responds to a nudge emits
                                # visible text and never re-enters this branch).
                                # Wording alone was already judged insufficient
                                # by the measured-failure note in the rung-1
                                # comment itself; arm the structural channel now.
                                self._force_tool_next = True
                                self._thinking_cap_override = 1024
                                if _cap_n >= 2:
                                    # Second (and later) cap in the SAME turn:
                                    # the model has now been told once to land
                                    # a concrete step and instead spent another
                                    # full stream thinking. Reasoning is not
                                    # converging -- cut it off and demand the
                                    # cheapest state-producing action.
                                    # The wording alone has been measured to
                                    # fail: the model acknowledges "I must act"
                                    # and then emits another full stream. So
                                    # the wording is paired with a STRUCTURAL
                                    # constraint on the next request: prose is
                                    # no longer a legal completion, only a tool
                                    # call is.
                                    self._force_tool_next = True
                                    # Squeeze the thinking budget (user
                                    # directive, 2026-10-30): the structural
                                    # tool_choice constraint has been measured
                                    # toothless on this endpoint — the server
                                    # forces thinking ON, and the client-side
                                    # caps abort the stream during that forced
                                    # thinking before the tool call is ever
                                    # reached. So instead of (only) forcing the
                                    # channel, SQUEEZE the thinking budget
                                    # itself: the next stream(s) get a small
                                    # token floor, so the reasoning is cut
                                    # early and the turn actually ends with a
                                    # tool call. Cleared when a write/edit
                                    # produces state on disk (below).
                                    self._thinking_cap_override = 1024
                                    nudge = (
                                        "[system: HARD STOP. Your reasoning has now hit the "
                                        f"runtime cap {_cap_n} times in this turn and each time "
                                        "you resumed thinking instead of acting. Further "
                                        "reasoning will be cut at the same wall every time, so "
                                        "it cannot finish the job — only a tool call can.\n"
                                        "This request has therefore been sent with a structural "
                                        "constraint: a prose answer is NOT accepted. Your next "
                                        "message MUST contain exactly ONE tool call, and it must "
                                        "produce state on disk. CONVERGE = WRITE THE FILE: take "
                                        "the artifact you have already drafted in the reasoning "
                                        "above and `write_file` it to the delivery path the task "
                                        "names now — even partial, even a skeleton, even "
                                        "imperfect. Refining it in later rounds is expected; "
                                        "landing nothing is the one outcome that cannot be "
                                        "rescued. The only other accepted call is `shell` ONE "
                                        "command that inspects the environment or tests your "
                                        "strongest hypothesis.\n"
                                        "If this reasoning already fixed the delivery contract "
                                        "(path/format/constraints), `memory_write` it first so it "
                                        "survives, then write the file.\n"
                                        "Do not explain, do not plan, do not summarize, do not "
                                        "reason. Emit the tool call immediately.\n\n"
                                        f"<previous_reasoning>\n{_cap_tail}\n"
                                        "</previous_reasoning>"
                                    )
                                else:
                                    # First cap this turn: bounded recap (not the
                                    # whole stream) so the next call spends its
                                    # input budget on acting, not re-reading.
                                    #
                                    # A menu ("either call a tool ... or write
                                    # your answer") leaves the model free to keep
                                    # deriving, and measured trajectories show it
                                    # does: a cap that merely ADVISES action is
                                    # followed by another full stream of the same
                                    # reasoning. The directive therefore names ONE
                                    # mandatory class of action and makes the
                                    # artifact-on-disk the default, not an option.
                                    nudge = (
                                        "[system: your reasoning reached its runtime length cap "
                                        "and was stopped. Your reasoning so far is reproduced "
                                        "BELOW AS A BOUNDED TAIL — do NOT continue it; continuing "
                                        "it will be cut at exactly the same wall again.\n"
                                        "CONVERGE means LAND A FILE, not 'think a shorter "
                                        "thought': take the solution you have ALREADY derived "
                                        "in the reasoning above and put it on disk at the "
                                        "delivery path the task names, THIS message. Even "
                                        "partial, even imperfect, even a skeleton you will "
                                        "refine in the next rounds — an imperfect file on disk "
                                        "can be edited, a perfect derivation that never lands "
                                        "cannot, and it scores exactly like no work at all.\n"
                                        "Your next message MUST contain exactly ONE tool "
                                        "call, and the default is `write_file` with whatever "
                                        "you already have. Only if truly nothing is writable "
                                        "yet, `shell` ONE command that tests your single "
                                        "strongest remaining hypothesis and prints its "
                                        "result.\n"
                                        "If your reasoning already fixed the delivery contract "
                                        "(path, format, constraints), `memory_write` it first so it "
                                        "survives, then write that file. Do not explain, do not "
                                        "plan, do not summarize, do not reason further: emit the "
                                        "tool call immediately. Then iterate from what that call "
                                        "returns — a wrong concrete answer you can see beats a "
                                        "right one you are still deriving.\n\n"
                                        "<previous_reasoning>\n"
                                        f"{_cap_tail}\n"
                                        "</previous_reasoning>"
                                    )
                        elif was_reasoning_only:
                            # Many providers (e.g. GLM Anthropic-compat endpoint)
                            # DROP assistant thinking blocks from the input, so
                            # the model cannot see its own reasoning on retry.
                            # Re-inject the FULL thinking content as a user
                            # message so the model resumes from its prior
                            # reasoning instead of starting over.
                            nudge = (
                                "[system: your previous response generated reasoning but no visible output, "
                                "and the reasoning was not preserved in the conversation. Your full reasoning "
                                "is reproduced below — resume from where it left off and output your answer "
                                "or next action directly. Do NOT repeat the reasoning.]\n\n"
                                "<previous_reasoning>\n"
                                f"{thinking_text}\n"
                                "</previous_reasoning>"
                            )
                        else:
                            nudge = "[system: empty response detected, please continue your work]"
                        d.history.append({"role": "user", "content": nudge})
                        continue
                    else:
                        self._empty_output_retries = 0
                        self._cap_retries = 0
                        self._intra_degen_retries = 0
                        # CAP TEXT STREAK deliberately does NOT reset here.
                        # Measured (round-2 rerun 2026-10-29 19-12-23,
                        # regex-chess__nqvTfjB): cap #1 -> first-rung bounded
                        # recap -> the model ran ONE `ls` (visible output) and
                        # resumed deriving; cap #2 then found the counter reset
                        # by that `ls`, so it served the SAME first-rung recap
                        # again and the HARD STOP rung was unreachable. Net:
                        # two caps, two stale recaps, action count pinned at 2,
                        # deliverable never written. One inspected command is
                        # not convergence, so only EVIDENCE OF PROGRESS clears
                        # the streak (keyed on tool kind, see _cap_progress).

                    # Detect completion signals. Use end-anchored matching to
                    # distinguish a real sentinel (trailing the text) from a
                    # mention (e.g. the agent quoting "[TASK_COMPLETE]" while
                    # explaining guard logic). A sentinel is a line that is
                    # (almost) just the marker, optionally followed by an
                    # _override_reason on the same or next line. The trailing
                    # check tolerates markdown decoration ("**[TASK_COMPLETE]**")
                    # — models routinely bold the marker, and a strict
                    # endswith() would reject it and loop on "no stop signal".
                    _text_stripped = assistant_text.rstrip()
                    _trailing = trailing_sentinel(_text_stripped)
                    _override_completion = has_override_completion(_text_stripped)
                    _is_completion = (
                        _trailing == "[TASK_COMPLETE]" or _override_completion
                    )
                    _is_need_input = _trailing == "[NEED_USER_INPUT]"
                    if _is_completion or _is_need_input:
                        # ── Completion-path guard consultation ──
                        # The break below is otherwise unconditional, so guards that
                        # gate completion (e.g. PlanGuard single-shot completion gate)
                        # get no say on a text-only [TASK_COMPLETE]. Consult them here:
                        # a block re-prompts the LLM (continue) instead of breaking, so
                        # the agent must address the gate (or override) before completing.
                        # Text-only [TASK_COMPLETE] has no tool_args to carry an
                        # _override_reason, so a completion-gate block would be
                        # unoverridable unless we give it a text channel. Parse an
                        # override reason from the assistant text itself: the agent
                        # declares it inline as  _override_reason: <reason>  (same
                        # keyword as the tool-arg form). This lets a genuinely-trivial
                        # task be released deliberately, while a bare re-emit of
                        # [TASK_COMPLETE] (no reason) stays blocked.
                        comp_override = self._extract_text_override(assistant_text)
                        comp_ctx = self._build_ctx(
                            tool_name="",
                            tool_args=({"_override_reason": comp_override}
                                       if comp_override else {}),
                            tool_result=None,
                            # This consultation runs right after the LLM produced
                            # the sentinel THIS iteration — the completion text is
                            # fresh, not stale history. The top-of-loop check_pre
                            # (llm_responded defaults False) must NOT be treated as
                            # a completion signal even when it scans a prior turn's
                            # trailing [TASK_COMPLETE] out of history.
                            llm_responded=True,
                        )
                        comp_verdict = d.guard_registry.check_pre(comp_ctx)
                        if comp_verdict is not None:
                            comp_blocked = self._apply_verdict(comp_verdict, pre=True)
                            if comp_blocked:
                                # Livelock breaker: a blocked completion `continue`s
                                # without touching _continuation_count, so a bare
                                # [TASK_COMPLETE] re-emitted with no plan / no override
                                # would spin to max_iter. Count consecutive blocks and
                                # give up once the agent has clearly stalled on the gate.
                                self._consecutive_completion_blocks += 1
                                if (self._consecutive_completion_blocks
                                        >= self.MAX_CONSECUTIVE_COMPLETION_BLOCKS):
                                    result.stop_reason = (
                                        "completion_gate_livelock: "
                                        f"{comp_verdict.reason}"
                                    )
                                    break
                                if iteration < max_iter - 1:
                                    # Re-prompt: LLM sees the guard message and must act
                                    continue
                        # Not blocked (released via plan/override) — reset the breaker.
                        self._consecutive_completion_blocks = 0
                        # The model produced a proper stop signal: the tool-free
                        # streak ends here too, otherwise the residue would make
                        # the NEXT turn's auto-append threshold fire early.
                        self._consecutive_text_only = 0
                        # Print the authoritative completion marker HERE — only now
                        # that the gate has passed. The raw sentinel was stripped
                        # from the live stream (SentinelStripper), so this is the
                        # single, gate-approved place it reaches the screen.
                        # By convention the terminating sentinel is the LAST one in
                        # the text; an earlier mention (e.g. the agent quoting
                        # "[TASK_COMPLETE]" while explaining code) must NOT override
                        # the actual trailing signal. Prefer the trailing sentinel
                        # already detected (decoration-tolerant); fall back to the
                        # last bare occurrence for the override form, where text
                        # follows the marker on the next line.
                        if _trailing is not None:
                            _last_sentinel = _trailing
                        else:
                            _last_sentinel = max(
                                ("[TASK_COMPLETE]", "[NEED_USER_INPUT]"),
                                key=lambda s: assistant_text.rfind(s),
                            )
                        if _last_sentinel in assistant_text:
                            d.display.completion_signal(_last_sentinel)
                        result.stop_reason = "explicit_signal"
                        break

                    # ── Degenerate-repeat loop breaker ──
                    # Independent of every counter above: those bound the NUMBER
                    # of continuations and are reset on tool calls / retries, so
                    # a stream that regenerates the same FEW texts outlives all
                    # of them. An ADJACENCY check is not enough either: the real
                    # failure was a PERIOD-4 cycle whose neighbours always
                    # differ (verified — adjacency-only ran to max_continuations
                    # without firing). What does catch it is the DIVERSITY of a
                    # window of recent tool-free responses.
                    if self.DEGENERATE_REPEAT_WINDOW_CALLS > 0:
                        _fp_src = (assistant_text or "").strip()
                        _fp = " ".join(
                            _fp_src[-self.DEGENERATE_REPEAT_WINDOW:].split()
                        )
                        _win = getattr(self, "_recent_assistant_fps", None)
                        if _win is None:
                            _win = self._recent_assistant_fps = []
                        if _fp:
                            _win.append(_fp)
                        _n = self.DEGENERATE_REPEAT_WINDOW_CALLS
                        if len(_win) > _n:
                            del _win[:-_n]
                        if (len(_win) >= _n
                                and len(set(_win)) <= self.DEGENERATE_REPEAT_DISTINCT_MOST):
                            result.stop_reason = "degenerate_repeat_loop"
                            result.error = (
                                f"the last {len(_win)} responses contained only "
                                f"{len(set(_win))} distinct texts with no tool call; "
                                "aborting to protect the wall clock"
                            )
                            d.display.warn(
                                "Degenerate repeat loop detected "
                                f"({len(_win)} responses, only {len(set(_win))} "
                                "distinct, no tool call) — stopping this turn."
                            )
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

                    # Track consecutive text-only responses (no tools, no stop signal)
                    self._consecutive_text_only = getattr(self, "_consecutive_text_only", 0) + 1

                    # M5: a text-only response that ALSO saturated the output
                    # token budget is a degenerate stream (the model emitted
                    # filler until the ceiling), not a paused-but-working
                    # turn. Track the streak and let the continuation name
                    # the concrete escape instead of a generic "please
                    # continue" (which re-fuels the stream).
                    _max_out = getattr(d.config, "max_output_tokens", 0) or 0
                    _saturated = bool(
                        _max_out > 0
                        and out_tok >= int(_max_out * self.OUTPUT_SATURATION_FRACTION)
                    )
                    if _saturated:
                        self._consecutive_saturated = (
                            getattr(self, "_consecutive_saturated", 0) + 1
                        )
                        # Drop the saturated filler from history: it would
                        # otherwise be re-sent as input on every later call
                        # (65k wasted input tokens per call) and invites the
                        # model to continue the same stream.
                        d.history.pop_last_assistant()
                    else:
                        self._consecutive_saturated = 0

                    # M5b: a long text-only streak (many short no-action calls)
                    # is the same degenerate stream even without a saturated
                    # call. Past the hard threshold, drop the accumulated
                    # no-action narration from history too — otherwise it is
                    # re-sent as input on every later call and invites the
                    # model to continue it.
                    _long_text_only = (
                        self.TEXT_ONLY_STREAK_HARD_AT > 0
                        and self._consecutive_text_only >= self.TEXT_ONLY_STREAK_HARD_AT
                    )
                    if _long_text_only and not _saturated:
                        d.history.pop_last_assistant()

                    # ── Auto-append fallback (honest hand-back) ──
                    # The escalation ladder above only re-prompts. If the model
                    # STILL produces tool-free, signal-free text past the
                    # fallback threshold, end the turn harness-side: append
                    # the kernel's OWN [NEED_USER_INPUT] (never rewrite the
                    # model's message, never fabricate [TASK_COMPLETE]) and
                    # stop. One hand-back per turn; the next run_turn() re-arms.
                    if (self.AUTO_APPEND_NEED_INPUT_AT > 0
                            and not self._auto_appended_this_turn
                            and self._consecutive_text_only
                            >= self.AUTO_APPEND_NEED_INPUT_AT):
                        self._auto_appended_this_turn = True
                        d.history.append({
                            "role": "user",
                            "content": (
                                "[system: auto_appended [NEED_USER_INPUT] — "
                                f"{self._consecutive_text_only} consecutive "
                                "responses contained no tool call and no stop "
                                "signal. The harness ended this turn on your "
                                "behalf; no deliverable was verified. Start the "
                                "next turn with a concrete tool call, or close "
                                "the task yourself.]"
                            ),
                        })
                        d.display.completion_signal("[NEED_USER_INPUT]")
                        result.stop_reason = "auto_appended_need_input"
                        break

                    continuation = self._generate_continuation(
                        text_only=True, saturated=_saturated
                    )
                    d.history.append({"role": "user", "content": continuation})
                    continue

                self._continuation_count = 0
                self._consecutive_text_only = 0  # Reset: tools were executed
                self._consecutive_saturated = 0  # M5: a real action resets the streak
                self._recent_assistant_fps = []  # tools ran: not a dead loop
                self._intra_degen_retries = 0  # tools ran: recovery succeeded
                self._degen_flush_armed = False  # tools ran: the write-through landed
                self._consecutive_completion_blocks = 0  # progress made; reset breaker

                # ── Execute tools ──
                _pre_guard_verdicts = []
                blocked_indices = set()
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
                                # a prevented call leaves a durable
                                # record OUTSIDE the execution ledger. Without it,
                                # "attempted and blocked" and "never attempted"
                                # are indistinguishable downstream (Audit #83's
                                # central finding rested entirely on one blocked
                                # shell call that happened to survive as a banner).
                                self._blocked_tool_calls.append({
                                    "name": tc.get("name", ""),
                                    "guard": getattr(verdict, "guard_name", "") or "",
                                    "reason": verdict.reason or "",
                                    "action": verdict.action,
                                })
                                # Deduplicate block/escalate across tool_calls
                                msg_key = verdict.message
                                if msg_key not in _seen_injects:
                                    _seen_injects.add(msg_key)
                                    _pre_guard_verdicts.append(verdict)
                                # Display is handled later in _apply_verdict — don't display here
                            elif verdict.action == "inject":
                                # Soft advisory — defer until after tool_results are appended
                                # Deduplicate exact repeats: full-text key (inject payloads
                                # may share a constant prefix yet differ later)
                                msg_key = verdict.message
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

                # ── Record the structured execution ledger ──
                # One entry per tool call that ACTUALLY ran this cycle; blocked
                # calls are excluded (they never executed). `exec_error` marks a
                # call whose result text is the executor's own error prefix, so
                # an evidence gate can tell execution from REAL execution.
                self._record_tool_ledger(tool_calls, results, blocked_indices)

                # Edit-churn stop-rule (FORBIDDEN_STOP_AFTER): a run of edit-
                # family calls with NO successful non-edit execution between
                # them. Count only EXECUTED calls (blocked ones never ran).
                # A successful non-edit execution resets the streak — that is
                # the calibration that spares a legitimate TDD cadence.
                if self.FORBIDDEN_STOP_AFTER > 0:
                    for i, tc in enumerate(tool_calls):
                        if i in blocked_indices:
                            continue
                        _name = tc.get("name", "")
                        _err = (results[i] if i < len(results) else "") or ""
                        if _name in self.FORBIDDEN_EDIT_TOOLS:
                            self._edit_churn_streak += 1
                        elif not str(_err).startswith("Error executing tool:"):
                            self._edit_churn_streak = 0
                        # Cap-escalation streak: cleared ONLY by evidence of
                        # progress — a successful state-producing call — not by
                        # any visible output (see the note at the cap reset).
                        # Measured: one inspection command between two caps is
                        # what kept the HARD STOP rung unreachable.
                        if (
                            _name in self.FORBIDDEN_EDIT_TOOLS
                            and not str(_err).startswith("Error executing tool:")
                        ):
                            self._cap_text_streak = 0
                            # Squeeze release (user directive): a write/edit
                            # has finally produced state on disk — restore the
                            # normal thinking budget so later reasoning is not
                            # artificially starved.
                            self._thinking_cap_override = None
                    if (self._edit_churn_streak >= self.FORBIDDEN_STOP_AFTER
                            and self._forbidden_stop_fires
                            < self.FORBIDDEN_STOP_MAX_FIRES):
                        self._forbidden_stop_fires += 1
                        self._edit_churn_streak = 0
                        _directive = self._generate_forbidden_stop_directive()
                        d.history.append({"role": "user", "content": _directive})
                        # Make the stop VISIBLE to the operator: the bare
                        # [NEED_USER_INPUT] sentinel alone gives the human no
                        # reason ("what fired?"). Echo a short explanation of
                        # the rule and the hand-back semantics. Model sees the
                        # full directive; this terminal echo is for the user.
                        d.display.guard_block(
                            "Edit-churn circuit breaker fired: "
                            f"{self.FORBIDDEN_STOP_AFTER} consecutive "
                            "edit_file/write_file calls with no successful "
                            "non-edit execution in between. The turn is ended "
                            "harness-side with [NEED_USER_INPUT] — no "
                            "deliverable was verified. The model was given a "
                            "directive naming the churn and the recovery "
                            "options (run evidence / different method / "
                            "honest hand-back). To clear it, the run must "
                            "execute a non-edit tool successfully.")
                        d.display.completion_signal("[NEED_USER_INPUT]")
                        result.stop_reason = "forbidden_edit_churn_stop"
                        break

                # M6: reset the repeat-block convergence streak ONLY when a tool
                # actually EXECUTED this cycle (real progress past whatever blocked
                # before). This MUST NOT fire when every call was blocked — a
                # repeated block requires the agent to re-issue an identical call,
                # which is exactly an all-blocked turn; resetting there zeroes the
                # streak at the top of every iteration before it can accumulate
                # (the counter would stay pinned at 1 and never reach the
                # threshold). So gate on at least one non-blocked call.
                if len(blocked_indices) < len(tool_calls):
                    d.guard_registry.reset_repeat_convergence()

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
                        # Deduplicate exact repeats across tool_calls (full-text key)
                        msg_key = verdict.message
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

                # A round whose every call was BLOCKED produced no state, so it
                # is an idle round-trip like any other: run the same gate the
                # tool-free branch uses. Without this, a run of blocked
                # round-trips is invisible to the gate (measured: 8 round-trips,
                # 1 executed call, ~300s in one turn, zero nudges).
                if blocked_indices and len(blocked_indices) == len(tool_calls):
                    _idle_state = self._idle_nudge(response, d, result)
                    if _idle_state == "exhausted":
                        break
                else:
                    # A tool really ran: the idle clock and round floor restart.
                    self._last_toolcall_ts = time.time()
                    self._idle_think_rounds = 0

                self._last_turn_had_tools = True
                result.iterations = iteration + 1

        finally:
            signal.signal(signal.SIGINT, _prev_handler)

        result.interrupted = self._interrupted
        result.elapsed = time.time() - turn_start
        return result

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_text_override(text: str) -> str:
        """Parse an inline override reason from assistant text.

        A text-only completion signal ([TASK_COMPLETE]) carries no tool_args, so
        an overridable completion-gate block needs a text channel. The agent
        declares an override inline using the same keyword as the tool-arg form:

            _override_reason: <reason text to end of line>

        Returns the reason (stripped) or "" if absent. Only a reason longer than
        the guard's minimum (checked downstream by accept_override) actually
        releases a block, so a bare keyword with no substance still gets blocked.
        """
        if not text:
            return ""
        # Match  _override_reason: ...  or  _override_reason = ...  (quotes optional)
        m = re.search(
            r'_override_reason\s*[:=]\s*["\']?(.+?)["\']?\s*$',
            text,
            re.MULTILINE,
        )
        return m.group(1).strip() if m else ""

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

    def _idle_nudge(self, response, d, result) -> str:
        """The cumulative idle think-interval gate.

        Judges ACCUMULATED TIME with no state produced since the last tool that
        ACTUALLY RAN — never one stream's token count. Individual responses stay
        under every length cap, so no threshold fires, yet the sum is what kills
        the wall clock (measured: six separate think round-trips, ~620s, first
        file write at 76% of budget).

        Returns:
          "fired"     — a nudge was injected; the caller should `continue`
          "exhausted" — the retry budget ran out; the caller should `break`
          "silent"    — below the round floor or the time interval
          "stop"      — the response carries an explicit stop signal; never
                        pre-empt a decision the model already made

        Shared by BOTH branches of the loop: a round whose every tool call was
        BLOCKED is as unproductive as a round with no tool calls at all, and
        leaving the gate only on the tool-free branch meant a run of blocked
        round-trips (measured: 8 round-trips, 1 executed call, 3 fully blocked,
        ~300s in one turn) never tripped it.
        """
        text = ""
        if isinstance(response.get("content"), str):
            text = response["content"].strip()
        elif isinstance(response.get("content"), list):
            text = "".join(
                b.get("text", "") for b in response["content"]
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
        if text and (trailing_sentinel(text)
                     in ("[TASK_COMPLETE]", "[NEED_USER_INPUT]")
                     or has_override_completion(text)):
            return "stop"

        # Persist the count: each round-trip is one loop iteration, so a
        # non-persisted count can never reach the floor.
        self._idle_think_rounds = getattr(self, "_idle_think_rounds", 0) + 1
        _idle_secs = time.time() - (
            getattr(self, "_last_toolcall_ts", None)
            or getattr(self, "_turn_start_ts", time.time())
        )
        if (self._idle_think_rounds < self.IDLE_THINK_MIN_ROUNDS
                or _idle_secs < self.IDLE_THINK_INTERVAL_SEC):
            return "silent"

        # Charge the SHARED cap budget, never a private counter: the exhausted
        # escape valve (and every white-box test that pre-sets it) keys on
        # _cap_retries, so a private counter makes exhaustion unreachable.
        _retries = getattr(self, "_cap_retries", 0)
        if _retries >= 50:
            self._cap_retries = 0
            result.stop_reason = "idle_interval_max_retries"
            return "exhausted"
        self._cap_retries = _retries + 1

        # RE-ARM BOTH, so the cadence is "another full interval of silence earns
        # another nudge" rather than one nudge per round-trip (noise, not pacing).
        self._last_toolcall_ts = time.time()
        self._idle_think_rounds = 0

        # BOUNDED recap, matching the cap path: re-injecting the full thinking
        # text re-feeds the very reasoning we are asking it to stop.
        _think = response.get("thinking") or ""
        _tail = _think[-4000:] if len(_think) > 4000 else _think
        # Second and later idle nudge: wording alone has been measured to
        # produce "I must act" followed by another full stream of reasoning, so
        # pair the wording with a STRUCTURAL constraint on the next request --
        # prose stops being a legal completion. The first nudge stays advisory
        # so a genuinely stuck-but-cooperative model is not jammed.
        _structural = _retries >= 1
        if _structural:
            self._force_tool_next = True
            # Pair the structural constraint with a SQUEEZE, same as the
            # cap ladder: force_tool alone was measured toothless on this
            # endpoint (the server forces thinking ON and the client-side
            # caps abort the stream during that forced thinking before any
            # tool call is reached) — round 7 regex-chess idle-retry 2 set
            # force_tool and the next stream still burned 240.3s of pure
            # thinking. Without the budget squeeze the structural wording
            # never reaches the model as an enforced constraint.
            self._thinking_cap_override = 1024
        d.display.warn(
            f"Idle think-interval ({_idle_secs:.0f}s with no tool state) — "
            f"forcing convergence (idle-retry {_retries + 1}"
            f"{', structural' if _structural else ''})..."
        )
        d.history.append({
            "role": "user",
            "content": (
                "[system: Idle think-interval — "
                f"{_idle_secs:.0f}s of wall-clock has accumulated since your "
                "last tool call that actually ran. Each response was "
                "individually legal; no length cap fired. The cost is "
                "CUMULATIVE TIME WITH NO STATE PRODUCED.\n"
                "STOP reasoning. Land the next concrete step NOW, in your very "
                "next output. CONVERGE = PUT A FILE ON DISK, not a shorter "
                "thought: take whatever you have already derived and `write_file` "
                "it at the delivery path the task names — even partial, even a "
                "skeleton, even imperfect. A file on disk can be refined; a "
                "derivation that never lands scores like no work. If nothing is "
                "writable yet, run the ONE probe that can kill your strongest "
                "current hypothesis before you build on it: the checker the task "
                "ships (`shell` `python3 check.py` or the test command), a call "
                "into the named library, a minimal-input reproduction. A probe "
                "you did not run is a hypothesis you cannot rule out; 30 seconds "
                "of observation outscores 5 minutes of derivation. Or output "
                "[NEED_USER_INPUT] with the specific question blocking you. "
                "If a guard is blocking you, re-issue the SAME call once "
                "with an `_override_reason` that addresses that guard's "
                "objection instead of re-deriving a new plan. An observed "
                "hypothesis beats a perfect derivation that never touched "
                "the world.\n"
                + (f"\n<previous_reasoning>\n{_tail}\n</previous_reasoning>"
                   if _tail.strip() else "")
            ),
        })
        return "fired"

    def _record_tool_ledger(self, tool_calls, results, blocked_indices) -> None:
        """Append one ledger entry per tool call that ACTUALLY ran.

        Blocked calls never executed, so they are excluded. Each entry carries
        `exec_error`: True when the result text is the executor's own error
        prefix (a per-call error or a batch-level failure that replaced the
        result), so an evidence gate can distinguish execution from REAL
        execution. `result` is coerced to a string (None -> "") and truncated
        to 500 chars.
        """
        err_prefix = "Error executing tool:"
        for i, tc in enumerate(tool_calls):
            if i in blocked_indices:
                continue
            raw = results[i] if i < len(results) else None
            text = "" if raw is None else str(raw)
            self._recent_tool_history.append({
                "name": tc.get("name", ""),
                "result": text[:500],
                "exec_error": text.startswith(err_prefix),
            })

    def _build_ctx(self, tool_name: str, tool_args: dict, tool_result: str | None,
                   llm_responded: bool = False) -> GuardContext:
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
            judge=d.judge,
            override_reason=override_reason,
            assistant_text=assistant_text,
            llm_responded=llm_responded,
            recent_tool_history=list(self._recent_tool_history),
            blocked_tool_calls=list(self._blocked_tool_calls),
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

    def _generate_forbidden_stop_directive(self) -> str:
        """Directive for the edit-churn stop-rule: name the churn, quote the
        plan's relevant prohibition(s) when available, and force a binary
        choice. Never claims completion — the terminal option is an honest
        [NEED_USER_INPUT] hand-back."""
        plan = None
        task_plan = getattr(self.deps, "task_plan", None)
        if task_plan:
            try:
                plan = task_plan.get_active() or {}
            except Exception:
                plan = {}
        quoted = []
        if plan:
            for item in (plan.get("forbidden") or []):
                action = str(item.get("action", "")) if isinstance(item, dict) else str(item)
                if action.strip():
                    quoted.append(action.strip())
        lines = [
            "[system: FORBIDDEN — you have now made "
            f"{self.FORBIDDEN_STOP_AFTER} consecutive edit-family calls with "
            "no successful execution in between. Your own plan's forbidden list "
            "exists for exactly this state.",
        ]
        if quoted:
            lines.append("Items you wrote that this pattern most likely violates:")
            for q in quoted[:3]:
                lines.append(f"  ✗ {q}")
        lines.append(
            "Repeating the same edit again is NOT one of your options. "
            "Your next response must contain EXACTLY ONE of:\n"
            "1. The new EVIDENCE that justifies another attempt — a command "
            "you ran and its actual output (not another edit).\n"
            "2. A DIFFERENT method entirely, executed as a tool call now.\n"
            "3. The single line [NEED_USER_INPUT] — hand the task back honestly; "
            "no deliverable was verified."
        )
        lines.append("Do not re-emit the same edit with tweaked arguments.]")
        return "\n".join(lines)

    def _generate_continuation(self, text_only: bool = False,
                               saturated: bool = False) -> str:
        """Generate continuation prompt.

        Args:
            text_only: If True, the previous response had no tool calls and no
                stop signal. Use a more explicit prompt to help LLM self-correct.
            saturated: If True, that text-only response ALSO hit the output
                token ceiling — a degenerate stream, not a paused thought.
                Past a small streak, forbid continuing the stream outright.
        """
        consecutive = getattr(self, "_consecutive_text_only", 0)
        saturated_streak = getattr(self, "_consecutive_saturated", 0)

        # M5: a saturated text-only stream repeating itself is the model
        # filling the token budget with filler. The generic "you must emit a
        # tool call / stop signal" nudge is not enough — it can be appended to
        # and re-saturated. Name the ceiling explicitly, forbid continuation,
        # and require a single concrete action on the very next output.
        # M5b: the SAME hard treatment applies to a long text-only STREAK even
        # when no single call saturated — a filler loop of many short calls is
        # the same degenerate stream, and the soft 3+ nudge repeats forever
        # without dropping the accumulated narration from history.
        _long_text_only = (
            self.TEXT_ONLY_STREAK_HARD_AT > 0
            and consecutive >= self.TEXT_ONLY_STREAK_HARD_AT
        )
        if (saturated and self.OUTPUT_SATURATION_NUDGE_AT > 0
                and saturated_streak >= self.OUTPUT_SATURATION_NUDGE_AT):
            return (
                f"[system: your last {saturated_streak} responses ran to the "
                "OUTPUT TOKEN CEILING while emitting no tool call and no stop "
                "signal — the stream is repeating itself and filling the "
                "budget with filler. STOP generating narration NOW. Do NOT "
                "continue the previous text or repeat a sentence. Your next "
                "response must contain EXACTLY ONE of:\n"
                "1. A single tool call that takes the next concrete action "
                "(run the command you keep describing, write the deliverable "
                "file, run the verifier) — no accompanying prose.\n"
                "2. The single line [TASK_COMPLETE] if the deliverable is "
                "already on disk and verified.\n"
                "3. The single line [NEED_USER_INPUT] if you are blocked.\n"
                "If you cannot decide, take option 1 with the smallest "
                "action that produces an artefact on disk. Length is not "
                "progress.]"
            )

        if _long_text_only:
            return (
                f"[system: your last {consecutive} responses contained NO tool "
                "call and NO stop signal — only narration. Nothing you have "
                "written in those responses changed anything on disk, and no "
                "amount of further narration will. STOP describing the next "
                "step and TAKE it. Do NOT repeat a sentence or continue the "
                "previous text. Your next response must contain EXACTLY ONE of:\n"
                "1. A single tool call that performs the next concrete action "
                "(run the command you keep describing, write the deliverable "
                "file, run the verifier) — with no accompanying prose.\n"
                "2. The single line [TASK_COMPLETE] if the deliverable is "
                "already on disk and verified.\n"
                "3. The single line [NEED_USER_INPUT] if you are blocked.\n"
                "If you cannot decide, take option 1 with the smallest action "
                "that produces an artefact on disk. Length is not progress.]"
            )

        # If LLM has been outputting text without tools repeatedly, escalate clarity
        if text_only and consecutive >= 3:
            return (
                "You have responded 3+ times with text only — no tool calls and no "
                "stop signal ([TASK_COMPLETE] or [NEED_USER_INPUT]).\n\n"
                "You MUST do one of:\n"
                "1. Emit a tool call (shell, readFile, editFile, etc.) to take action\n"
                "2. End with [TASK_COMPLETE] if the task is done\n"
                "3. End with [NEED_USER_INPUT] if you need user input\n\n"
                "Do NOT output only narration or intentions. Act or stop."
            )

        if text_only:
            plan_hint = ""
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
                        plan_hint = f" Current step: {step.get('title', step.get('description', ''))}"
            return (
                "Your previous response contained text but no tool calls and no stop signal. "
                "The ONLY valid stop signals are [TASK_COMPLETE] and [NEED_USER_INPUT] "
                "— they must appear as the last line of your response. "
                "Simply stopping output is NOT a stop signal.\n\n"
                "You MUST choose exactly one:\n"
                "1. If the task is fully done: end with [TASK_COMPLETE] as the last line, "
                "no tool calls in the same response.\n"
                "2. If you need user input: end with [NEED_USER_INPUT] as the last line, "
                "no tool calls in the same response.\n"
                "3. If you need to take action: emit ONLY tool call(s), no stop signal.\n\n"
                "Do NOT mix tool calls and stop signals in the same response."
                f"{plan_hint}"
            )

        # Normal continuation (after tool execution, LLM just didn't stop)
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

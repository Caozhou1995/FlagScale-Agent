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

"""Tests for the M5 output-saturation convergence nudge.

A text-only LLM response (no tool call, no stop signal) that ALSO saturated
the output token budget is a degenerate stream — the model emitted filler
until the ceiling. On a repeating streak the kernel must inject a directive
that FORBIDS continuing the stream, instead of the generic "please continue"
nudge (which re-fuels it).
"""

from unittest.mock import MagicMock
import types

from flagscale_agent.react.kernel import AgentKernel, KernelDeps
from flagscale_agent.react.guard import GuardRegistry


class _History:
    def __init__(self):
        self.messages = [{"role": "user", "content": "do the task"}]

    def append(self, msg):
        self.messages.append(msg)

    def get_messages(self):
        return self.messages

    def get_context_pressure(self):
        return 0.1

    def get_evictable_indexes(self):
        return []

    def report_actual_tokens(self, n):
        pass

    def pop_last_assistant(self):
        # drop the last assistant message if present
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                self.messages.pop(i)
                return


def _make_kernel(responses, max_output_tokens=100):
    """Build a kernel that replays a fixed list of (response, usage) pairs.

    responses: list of dicts -> each is the unified response dict. Usage is
    derived: output_tokens = max_output_tokens when the response is text-only
    and saturated (flag), else a small number.
    """
    history = _History()
    config = types.SimpleNamespace(
        max_iterations=50, max_continuations=200, mode="auto",
        max_output_tokens=max_output_tokens,
    )
    state = {"i": 0}

    def call_llm_fn(messages, schemas):
        i = state["i"]
        state["i"] += 1
        if i >= len(responses):
            # Stop the loop cleanly
            return ({"content": "done [TASK_COMPLETE]", "tool_calls": None},
                    {"input_tokens": 1, "output_tokens": 5})
        r = responses[i]
        out = r.pop("_out_tokens", 5)
        return r, {"input_tokens": 1, "output_tokens": out}

    provider = MagicMock()
    provider.format_assistant_message.side_effect = (
        lambda r: {"role": "assistant", "content": r.get("content", "")}
    )

    deps = KernelDeps(
        provider=provider,
        history=history,
        tool_registry=MagicMock(),
        judge=MagicMock(),
        guard_registry=GuardRegistry(),
        config=config,
        display=MagicMock(),
        get_schemas_fn=lambda: [],
        inject_message_fn=lambda m: None,
        append_tool_results_fn=lambda r: None,
        format_tool_result_fn=lambda t, r: {},
        execute_tools_fn=lambda tcs: ["ok"] * len(tcs),
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=call_llm_fn,
    )
    return AgentKernel(deps), history, state


# ── Unit: _generate_continuation branching ───────────────────────────────────

def test_saturation_directive_at_threshold():
    """saturated streak >= threshold -> a stream-forbidding directive."""
    k, _, _ = _make_kernel([])
    k._consecutive_saturated = k.OUTPUT_SATURATION_NUDGE_AT
    msg = k._generate_continuation(text_only=True, saturated=True)
    assert "OUTPUT TOKEN CEILING" in msg
    assert "[TASK_COMPLETE]" in msg
    # must explicitly forbid continuing the stream
    assert "STOP generating" in msg


def test_saturation_below_threshold_uses_normal_nudge():
    """A single saturated turn is not yet a repeating streak."""
    k, _, _ = _make_kernel([])
    k._consecutive_saturated = k.OUTPUT_SATURATION_NUDGE_AT - 1
    msg = k._generate_continuation(text_only=True, saturated=True)
    assert "OUTPUT TOKEN CEILING" not in msg
    assert "no tool calls and no stop signal" in msg


def test_unsaturated_text_only_never_triggers_directive():
    """A short text-only turn keeps the generic nudge even at a high streak."""
    k, _, _ = _make_kernel([])
    k._consecutive_saturated = 99
    msg = k._generate_continuation(text_only=True, saturated=False)
    assert "OUTPUT TOKEN CEILING" not in msg


def test_disabled_when_threshold_zero():
    """OUTPUT_SATURATION_NUDGE_AT = 0 disables the mechanism entirely."""
    k, _, _ = _make_kernel([])
    saved = k.OUTPUT_SATURATION_NUDGE_AT
    try:
        k.OUTPUT_SATURATION_NUDGE_AT = 0
        k._consecutive_saturated = 100
        msg = k._generate_continuation(text_only=True, saturated=True)
        assert "OUTPUT TOKEN CEILING" not in msg
    finally:
        k.OUTPUT_SATURATION_NUDGE_AT = saved


# ── Integration: streak counting + injection through run_turn ─────────────────

def test_run_turn_counts_saturated_streak_and_injects_directive():
    """Two consecutive saturated text-only turns -> directive in history."""
    # max_output_tokens=100; each text-only response reports output_tokens=100
    resp = [
        {"content": "Let me run.\n\nLet me go.", "tool_calls": None, "_out_tokens": 100},
        {"content": "Let me run.\n\nLet me go.", "tool_calls": None, "_out_tokens": 100},
        {"content": "Let me run.\n\nLet me go.", "tool_calls": None, "_out_tokens": 100},
    ]
    k, history, _ = _make_kernel(resp, max_output_tokens=100)
    result = k.run_turn()
    joined = "\n".join(m.get("content", "") for m in history.messages
                       if isinstance(m.get("content"), str))
    assert "OUTPUT TOKEN CEILING" in joined


def test_unsaturated_text_only_does_not_inject_directive():
    """Text-only turns that do NOT saturate never trigger the directive."""
    resp = [
        {"content": "thinking briefly", "tool_calls": None, "_out_tokens": 5},
        {"content": "thinking briefly", "tool_calls": None, "_out_tokens": 5},
        {"content": "thinking briefly", "tool_calls": None, "_out_tokens": 5},
    ]
    k, history, _ = _make_kernel(resp, max_output_tokens=100)
    k.run_turn()
    joined = "\n".join(m.get("content", "") for m in history.messages
                       if isinstance(m.get("content"), str))
    assert "OUTPUT TOKEN CEILING" not in joined


def test_tool_turn_resets_saturated_streak():
    """A real tool call resets the saturated streak to zero."""
    k, _, _ = _make_kernel([])
    k._consecutive_saturated = 3
    resp = [
        {"content": "", "tool_calls": [{"id": "1", "name": "shell", "arguments": {}}],
         "_out_tokens": 100},
    ]
    k2, _, _ = _make_kernel(resp, max_output_tokens=100)
    k2._consecutive_saturated = 3
    k2.run_turn()
    assert k2._consecutive_saturated == 0


# ── M5b: text-only streak (mode-B filler, no single saturated call) ──────────

def test_text_only_streak_hard_directive_at_threshold():
    """A long text-only streak triggers the hard directive even unsaturated."""
    k, _, _ = _make_kernel([])
    k._consecutive_text_only = k.TEXT_ONLY_STREAK_HARD_AT
    msg = k._generate_continuation(text_only=True, saturated=False)
    assert "NO tool call and NO stop signal" in msg
    assert "[TASK_COMPLETE]" in msg
    assert "Length is not progress" in msg
    # it must NOT be the soft 3+ nudge (which repeats forever without force)
    assert "responded 3+ times with text only" not in msg


def test_text_only_streak_below_threshold_uses_soft_or_generic_nudge():
    """Below the hard threshold, a short streak keeps the weaker nudge."""
    k, _, _ = _make_kernel([])
    k._consecutive_text_only = k.TEXT_ONLY_STREAK_HARD_AT - 1
    msg = k._generate_continuation(text_only=True, saturated=False)
    assert "NO tool call and NO stop signal" not in msg


def test_text_only_streak_hard_disabled_when_zero():
    """TEXT_ONLY_STREAK_HARD_AT = 0 disables the mode-B mechanism."""
    k, _, _ = _make_kernel([])
    saved = k.TEXT_ONLY_STREAK_HARD_AT
    try:
        k.TEXT_ONLY_STREAK_HARD_AT = 0
        k._consecutive_text_only = 999
        msg = k._generate_continuation(text_only=True, saturated=False)
        assert "NO tool call and NO stop signal" not in msg
    finally:
        k.TEXT_ONLY_STREAK_HARD_AT = saved


def test_run_turn_short_text_only_streak_injects_hard_directive():
    """Many SHORT unsaturated text-only turns (mode-B) -> hard directive."""
    resp = [{"content": "Let me go.", "tool_calls": None, "_out_tokens": 5}
            for _ in range(8)]
    k, history, _ = _make_kernel(resp, max_output_tokens=100)
    k.run_turn()
    joined = "\n".join(m.get("content", "") for m in history.messages
                       if isinstance(m.get("content"), str))
    assert "NO tool call and NO stop signal" in joined


# ── M11: saturated REASONING-ONLY response (no visible text at all) ──────────
#
# The real failure (jobs/ds1017_m10/model-extraction, HEAD=1c7ed76): a single
# call reported `632.7s | ↑20,616 | ↓65,536` — the whole output budget went to
# one monolithic thinking stream that was cut off before emitting any visible
# text or tool call. Because `assistant_text` was empty, control went to the
# EMPTY-OUTPUT branch, which treated it as "reasoning produced content" and
# re-injected the full ~20k-token reasoning with a RESUME directive, actively
# feeding the stream; M5's saturation guard sits after that branch's `continue`
# and was therefore unreachable. These tests pin the corrected routing.

def _thinking_only(out_tokens, text="a very long monolithic chain of thought " * 50):
    """A reasoning-only response: thinking content, no visible text, no tools."""
    return {"content": "", "tool_calls": None, "thinking": text,
            "_out_tokens": out_tokens}


def _all_injected(history):
    return "\n".join(m.get("content", "") for m in history.messages
                     if isinstance(m.get("content"), str))


def test_saturated_reasoning_only_routes_to_convergence_not_resume():
    """Saturated thinking-only call must NOT get the resume directive.

    Falsifier for the real defect: before the fix this asserted-failing branch
    was the only one reachable, and its message told the model to "resume from
    where it left off" — the exact behaviour that re-fuels the dead stream.
    """
    resp = [_thinking_only(100), _thinking_only(100)]
    k, history, _ = _make_kernel(resp, max_output_tokens=100)
    k.run_turn()
    joined = _all_injected(history)
    assert "consumed the ENTIRE output budget" in joined
    # the old (wrong) path's directive must be gone
    assert "resume from where it left off" not in joined


def test_saturated_reasoning_only_repeats_escalate_to_hard_directive():
    """A second saturated thinking-only call escalates to the HARD STOP form."""
    resp = [_thinking_only(100) for _ in range(3)]
    k, history, _ = _make_kernel(resp, max_output_tokens=100)
    k.run_turn()
    joined = _all_injected(history)
    assert "HARD STOP" in joined
    assert "MUST contain a tool call" in joined
    assert k._saturated_text_streak >= 2


def test_unsaturated_reasoning_only_keeps_resume_path():
    """Below the ceiling, a reasoning-only response is still a normal retry.

    Guards against over-reach: the convergence directive must be reserved for
    genuine saturation, not for every reasoning-only response.
    """
    resp = [_thinking_only(10), _thinking_only(10)]
    k, history, _ = _make_kernel(resp, max_output_tokens=100)
    k.run_turn()
    joined = _all_injected(history)
    assert "consumed the ENTIRE output budget" not in joined
    assert "reasoning but no visible output" in joined


def test_saturation_leaves_no_full_reasoning_in_context():
    """The dead stream must not be re-injected whole (input budget waste)."""
    big = "REASONING_MARKER " * 2000  # -> truncated to a bounded tail
    resp = [{"content": "", "tool_calls": None, "thinking": big, "_out_tokens": 100},
            _thinking_only(100)]
    k, history, _ = _make_kernel(resp, max_output_tokens=100)
    k.run_turn()
    injected = [m.get("content", "") for m in history.messages
                if isinstance(m.get("content"), str)]
    sat_msgs = [c for c in injected if "consumed the ENTIRE output budget" in c]
    assert sat_msgs, "saturation directive must be injected"
    assert all(len(c) < 6000 for c in sat_msgs), \
        "the discarded reasoning must be re-injected as a bounded recap only"


def test_saturated_reasoning_only_does_not_count_as_text_only_continuation():
    """Saturation is handled before the text-only bookkeeping.

    The two counters must stay independent: saturation escalates through
    _saturated_text_streak, not through the mode-B text-only streak.
    """
    resp = [_thinking_only(100) for _ in range(3)]
    k, history, _ = _make_kernel(resp, max_output_tokens=100)
    k.run_turn()
    assert getattr(k, "_consecutive_text_only", 0) == 0
    joined = _all_injected(history)
    assert "NO tool call and NO stop signal" not in joined


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

"""ActuallyRecheckGuard — nudge when reasoning says "Actually, ...".

What this guard IS:
- post-only, inject-only. Never blocks, never narrows the agent's options.
- LITERAL detection of the word "actually" (case-insensitive) in the CURRENT
  iteration's REASONING (ctx.thinking_text). Deliberately no regex and no
  semantic classification: the harness must not decide whether a reversal was
  right or wrong — that is the agent's own judgement, and a false verdict in
  the context is worse than silence.
- dedup per LLM iteration via len(ctx.messages); several tool calls emitted by
  one assistant message must not produce several copies of the note.

What this guard is NOT:
- it does not read assistant TEXT (that is the visible answer, not the
  reversal trace);
- it does not fire on other words ("actual", "factual", "practically") — the
  token boundary matters, else the note becomes background noise;
- it does not require a tool call to have produced a specific result.
"""

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.actually_recheck import (
    ActuallyRecheckGuard,
    TRIGGER_WORD,
)


def _ctx(thinking="", messages=None, **kw):
    return GuardContext(
        thinking_text=thinking,
        messages=messages if messages is not None else [{"role": "user"}],
        **kw,
    )


def _iter(n):
    return [{"role": "user", "i": i} for i in range(n)]


# ── check_pre never fires ───────────────────────────────────────────────────

def test_check_pre_is_always_none():
    g = ActuallyRecheckGuard()
    assert g.check_pre(_ctx("Actually, no.")) is None


# ── detection ───────────────────────────────────────────────────────────────

def test_fires_on_actually_in_reasoning():
    g = ActuallyRecheckGuard()
    v = g.check_post(_ctx("Actually, the task doesn't require it."))
    assert v is not None
    assert v.action == "inject"
    assert v.category == "actually_recheck"


def test_case_insensitive():
    g = ActuallyRecheckGuard()
    assert g.check_post(_ctx("ACTUALLY, hold on.", messages=_iter(3))) is not None
    g2 = ActuallyRecheckGuard()
    assert g2.check_post(_ctx("actually, hold on.", messages=_iter(3))) is not None


def test_trigger_word_constant_is_actually():
    assert TRIGGER_WORD == "actually"


def test_silent_when_no_thinking_published():
    """Reasoning never becomes a history message; absent text must be silent."""
    g = ActuallyRecheckGuard()
    assert g.check_post(_ctx("")) is None


def test_silent_when_word_absent():
    g = ActuallyRecheckGuard()
    assert g.check_post(_ctx("The peak is at 1580 cm^-1; I will fit a Lorentzian.")) is None


def test_does_not_fire_on_longer_words_containing_the_word():
    """'actual', 'factual', 'practically' must not trigger — otherwise the
    note degrades into background noise inside a long trace."""
    g = ActuallyRecheckGuard()
    assert g.check_post(_ctx("The actual value differs.", messages=_iter(2))) is None
    assert g.check_post(_ctx("This is factual and practical.", messages=_iter(3))) is None


def test_does_not_read_assistant_text():
    """The VISIBLE answer containing the word is not a reasoning reversal."""
    g = ActuallyRecheckGuard()
    ctx = _ctx("", assistant_text="Actually, here is my answer.")
    assert g.check_post(ctx) is None


# ── dedup per LLM iteration ─────────────────────────────────────────────────

def test_one_inject_per_iteration():
    g = ActuallyRecheckGuard()
    first = g.check_post(_ctx("Actually, reconsider.", messages=_iter(5)))
    assert first is not None and first.action == "inject"
    # Same iteration: a second tool call re-runs check_post with the same
    # len(messages) — must be silent.
    second = g.check_post(_ctx("Actually, reconsider.", messages=_iter(5)))
    assert second is None


def test_new_iteration_fires_again():
    g = ActuallyRecheckGuard()
    assert g.check_post(_ctx("Actually, reconsider.", messages=_iter(5))) is not None
    assert g.check_post(_ctx("Actually, reconsider.", messages=_iter(7))) is not None


def test_reset_turn_clears_marker():
    g = ActuallyRecheckGuard()
    assert g.check_post(_ctx("Actually, reconsider.", messages=_iter(5))) is not None
    g.reset_turn()
    assert g.check_post(_ctx("Actually, reconsider.", messages=_iter(5))) is not None


# ── message contract ────────────────────────────────────────────────────────

def test_message_asks_for_evidence_and_leaves_the_decision_to_the_agent():
    g = ActuallyRecheckGuard()
    v = g.check_post(_ctx("Actually, drop that requirement.", messages=_iter(1)))
    assert v is not None
    body = v.message
    # asks for evidence behind the reversal
    assert "evidence" in body.lower()
    # explicitly leaves the verdict to the agent (no answer smuggled in)
    assert "You decide" in body
    # must not carry a tool-specific or task-specific instruction
    assert "assert" not in body.lower()


def test_verdict_is_inject_and_overridable_never_block():
    g = ActuallyRecheckGuard()
    v = g.check_post(_ctx("Actually, no.", messages=_iter(1)))
    assert v is not None
    assert v.action == "inject"
    assert v.overridable is True

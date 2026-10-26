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

"""The 75% rung must not be able to kill the run it is trying to protect.

Observed failure (2026-10-26, terminal-bench, two independent trials):

  gpt2-codegolf: the agent's FIRST write landed a 1200-byte skeleton at the
  delivery path. Every APPEND that would have completed the file was then
  blocked by the 75% cold-consumer rung — 9 consecutive identical blocks,
  ending in "🛑 CONVERGENCE (HARD): 9 identical blocks". The agent diagnosed
  it itself ("I need to split into 2 calls. But guards keep blocking
  appends"), and the run died at 91% of budget with the skeleton still the
  deliverable (reward 0). The sibling trial showed 8 blocks.

Two defects, both closed here:

  1. CLOSED LOOP. The 75% rung asks "have you RUN the artifact you are about
     to ship?" — but appending the bytes that make the artifact runnable is
     itself the blocked call. A cold-consumer question cannot gate the
     construction of the thing it wants run. Any rung now releases on real
     landing evidence (a writer that actually executed, or this very call
     being that write).

  2. NO ESCAPE VALVE. A rung the environment makes structurally unsatisfiable
     re-armed forever, converting a speed bump into the cause of death. After
     _MAX_DENIALS rejections the rung stops blocking and speaks once more as
     a non-blocking demand naming the honest exit (land what you have / state
     the failed command and its real output).
"""

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.first_e2e_run import FirstE2eRunGuard


def make_ctx(tool_name="shell", tool_args=None, override_reason="",
             history=None):
    ctx = GuardContext()
    ctx.tool_name = tool_name
    ctx.tool_args = tool_args or {}
    ctx.override_reason = override_reason
    if history is not None:
        ctx.recent_tool_history = history
    return ctx


class _Stats:
    def __init__(self, pct=0.0):
        self.pct = pct

    def __call__(self):
        budget = 1200.0
        elapsed = budget * self.pct / 100.0
        return {"elapsed": elapsed, "budget": budget,
                "remaining": budget - elapsed, "pct": self.pct}


def _write_history():
    return [{"name": "write_file",
             "result": "Wrote 2533 chars to /app/gpt2.c", "exec_error": False}]


def _failed_write_history():
    return [{"name": "write_file", "result": "error", "exec_error": True}]


def _readonly_history():
    return [{"name": "shell", "result": "$ ls -la /app", "exec_error": False}]


def _guard(pct=80.0):
    return FirstE2eRunGuard(stats_fn=_Stats(pct))


# ── defect 1: a landing releases every rung, including 75% ────────────────

class TestLandingReleasesSeventyFive:
    def test_append_after_first_landing_is_not_blocked(self):
        """The exact failure shape: first write landed, the append that
        completes the file must go through."""
        g = _guard(80.0)
        first = g.check_pre(make_ctx(tool_name="write_file",
                                     tool_args={"path": "/app/gpt2.c"}))
        assert first is not None, "75% rung should ask once"
        # The very append that completes the artifact: a landed write exists
        # in the execution ledger, so this rung has been answered.
        ok = g.accept_override(
            "appending part 2 of the file at /app/gpt2.c",
            make_ctx(tool_name="write_file", history=_write_history()))
        assert ok is True
        # And the gate stays quiet for the rest of the turn.
        assert g.check_pre(make_ctx(tool_name="write_file",
                                    history=_write_history())) is None

    def test_failed_write_is_not_landing_evidence(self):
        """A write that errored is not a landing — it must not release."""
        g = _guard(80.0)
        g.check_pre(make_ctx(tool_name="write_file"))
        assert g.accept_override(
            "trying to write the file",
            make_ctx(tool_name="shell", history=_failed_write_history())) is False

    def test_lower_rung_still_requires_evidence(self):
        """The 25% rung keeps its stricter behaviour: prose alone is not a
        write. (Guards against the landing rule swallowing the progress rule.)"""
        g = _guard(30.0)
        g.check_pre(make_ctx(tool_name="shell"))
        assert g.accept_override(
            "I will write the encoder next",
            make_ctx(tool_name="shell", history=_readonly_history())) is False


# ── defect 2: escape valve after repeated denials ────────────────────────

class TestEscapeValve:
    def test_rung_stops_blocking_after_max_denials(self):
        g = _guard(80.0)
        denials = 0
        for _ in range(FirstE2eRunGuard._MAX_DENIALS):
            assert g.check_pre(make_ctx(tool_name="shell")) is not None
            assert g.accept_override(
                "nothing runnable yet", make_ctx(tool_name="shell",
                                                 history=_readonly_history())) is False
            denials += 1
        assert denials == FirstE2eRunGuard._MAX_DENIALS
        # Next crossing must NOT block — it speaks once as a demand instead.
        v = g.check_pre(make_ctx(tool_name="shell"))
        assert v is not None
        assert v.action != "block", "the rung must not block past its denial budget"
        assert "honest exit" in v.message.lower()
        # And it is retired for the turn: no further blocks at all.
        assert g.check_pre(make_ctx(tool_name="shell")) is None

    def test_escape_valve_message_names_the_failed_command_route(self):
        g = _guard(80.0)
        for _ in range(FirstE2eRunGuard._MAX_DENIALS):
            g.check_pre(make_ctx(tool_name="shell"))
            g.accept_override("no", make_ctx(tool_name="shell",
                                             history=_readonly_history()))
        v = g.check_pre(make_ctx(tool_name="shell"))
        msg = v.message.lower()
        # The two legal exits must both be spelled out.
        assert "incomplete" in msg
        assert "error output" in msg

    def test_denial_budget_resets_on_new_turn(self):
        g = _guard(80.0)
        g.check_pre(make_ctx(tool_name="shell"))
        g.accept_override("no", make_ctx(tool_name="shell",
                                         history=_readonly_history()))
        assert g._denials.get(75) == 1
        g.reset_turn()
        assert g._denials == {}
        # A fresh turn gets the full budget back (it may still block).
        assert g.check_pre(make_ctx(tool_name="shell")) is not None

    def test_landing_clears_the_denial_count(self):
        g = _guard(80.0)
        g.check_pre(make_ctx(tool_name="write_file"))
        g.accept_override("no", make_ctx(tool_name="shell",
                                         history=_readonly_history()))
        assert g._denials.get(75) == 1
        g.check_pre(make_ctx(tool_name="write_file"))
        assert g.accept_override(
            "appending", make_ctx(tool_name="write_file",
                                   history=_write_history())) is True
        assert not g._denials.get(75), "an accepted release clears the count"

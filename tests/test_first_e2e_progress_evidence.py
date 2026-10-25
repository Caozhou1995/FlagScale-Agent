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

"""ProgressOrder 25%/50% rungs: PROGRESS-bound release + re-arm on rejection.

Observed failure (2026-10-26, two independent terminal-bench trials): the
25% and 50% rungs released on any prose _override_reason. In both runs the
agent answered with a sentence like "next step: write the encoder" and then
kept reasoning for another 40-45% of the wall-clock WITHOUT A SINGLE TOOL
CALL (measured: 1151 log lines and 1 tool call between the 25% block and the
first write). One run's first delivery landed at 76% of budget, the other's
real implementation only at 60%.

Two defects, both closed here:
  1. A promise to write released the rung. Now the release requires a write
     that actually EXECUTED (kernel execution ledger), or "none apply".
  2. A rung's block marked every crossed threshold spent at fire time, and
     only reset_turn() ever cleared that set. A single-turn task (no new user
     turn) therefore retired each rung FOREVER after one appearance — which is
     why adding prompt clauses did nothing for five consecutive batches.
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
    """A write_file that ACTUALLY ran (module-level writer)."""
    return [{"name": "write_file",
             "result": "Wrote 412 chars to /app/enc.py", "exec_error": False}]


def _shell_write_history():
    """A shell call is NOT a write signal: the harness's own writers are the
    only authoritative landing event (a shell token list would be an
    open-ended detector and a `grep` whose output contains such a token would
    false-pass)."""
    return [{"name": "shell",
             "result": "wrote /app/enc.py (412 bytes)", "exec_error": False}]


def _readonly_history():
    return [{"name": "shell", "result": "$ ls -la /app\ntotal 16",
             "exec_error": False}]


def _errored_write_history():
    return [{"name": "write_file",
             "result": "Error executing tool: permission denied",
             "exec_error": True}]


_PROMISE = ("Design is settled; next step I will write the encoder to "
            "/app/enc.py and iterate. This checkpoint can release.")
_NONE_APPLY = ("none apply — this deliverable has no delivery path; the "
               "artifact is a passive data file read by the consumer.")


class TestProsePromiseDoesNotRelease:
    """The core regression: a promise to write is not a write."""

    def test_promise_with_no_write_is_rejected(self):
        s = _Stats(26.0)
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx())
        assert v is not None and v.reason == "first_e2e_run_25pct"
        ctx = make_ctx(override_reason=_PROMISE, history=_readonly_history())
        assert g.accept_override(_PROMISE, ctx) is False

    def test_rejected_rung_rearms_and_blocks_again(self):
        """Single-turn task: the rung must NOT stay retired after rejection."""
        s = _Stats(26.0)
        g = FirstE2eRunGuard(stats_fn=s)
        assert g.check_pre(make_ctx()) is not None
        assert g.accept_override(_PROMISE,
                                 make_ctx(history=_readonly_history())) is False
        # Still >=25%: the checkpoint must re-block, not fall silent forever.
        v2 = g.check_pre(make_ctx())
        assert v2 is not None and v2.reason == "first_e2e_run_25pct"

    def test_real_write_then_release_then_silent(self):
        s = _Stats(26.0)
        g = FirstE2eRunGuard(stats_fn=s)
        assert g.check_pre(make_ctx()) is not None
        assert g.accept_override(_PROMISE,
                                 make_ctx(history=_write_history())) is True
        assert g.check_pre(make_ctx()) is None  # released, stays spent

    def test_errored_write_does_not_release(self):
        s = _Stats(26.0)
        g = FirstE2eRunGuard(stats_fn=s)
        assert g.check_pre(make_ctx()) is not None
        ctx = make_ctx(history=_errored_write_history())
        assert g.accept_override(_PROMISE, ctx) is False

    def test_shell_output_mentioning_write_is_not_evidence(self):
        """A shell call is not a write rung release — even if its OUTPUT text
        mentions writing (that is the false-pass a token list would create)."""
        s = _Stats(26.0)
        g = FirstE2eRunGuard(stats_fn=s)
        assert g.check_pre(make_ctx()) is not None
        ctx = make_ctx(history=_shell_write_history())
        assert g.accept_override(_PROMISE, ctx) is False

    def test_this_call_being_the_write_releases(self):
        """Option (c): the call being blocked IS the first landing."""
        s = _Stats(26.0)
        g = FirstE2eRunGuard(stats_fn=s)
        assert g.check_pre(make_ctx()) is not None
        ctx = make_ctx(tool_name="write_file",
                       tool_args={"path": "/app/enc.py", "content": "x=1"})
        assert g.accept_override(_PROMISE, ctx) is True

    def test_none_apply_still_allowed(self):
        s = _Stats(26.0)
        g = FirstE2eRunGuard(stats_fn=s)
        assert g.check_pre(make_ctx()) is not None
        assert g.accept_override(_NONE_APPLY, make_ctx()) is True


class TestFiftyRung:
    def test_fifty_promise_rejected_and_rearms(self):
        s = _Stats(52.0)
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx())
        assert v is not None and v.reason == "first_e2e_run_50pct"
        assert g.accept_override(_PROMISE,
                                 make_ctx(history=_readonly_history())) is False
        v2 = g.check_pre(make_ctx())
        assert v2 is not None and v2.reason == "first_e2e_run_50pct"

    def test_fifty_accepts_with_write_evidence(self):
        s = _Stats(52.0)
        g = FirstE2eRunGuard(stats_fn=s)
        assert g.check_pre(make_ctx()) is not None
        assert g.accept_override(_PROMISE,
                                 make_ctx(history=_write_history())) is True


class TestUnreleasedRungsAreNotSubsumed:
    """A jump past several rungs asks the most severe NOW; the earlier rung
    is a question still owed an answer, not a skipped one."""

    def test_jump_to_52_answers_50_and_subsumes_25(self):
        """A real landing at the 50% rung also answers the 25% write-through
        question — the same write satisfies both, so the agent is not blocked
        a second time for evidence it already produced."""
        s = _Stats(52.0)
        g = FirstE2eRunGuard(stats_fn=s)
        v50 = g.check_pre(make_ctx())
        assert v50.reason == "first_e2e_run_50pct"
        assert g.accept_override(_PROMISE,
                                 make_ctx(history=_write_history())) is True
        assert g.check_pre(make_ctx()) is None

    def test_reset_turn_rearms_everything(self):
        s = _Stats(30.0)
        g = FirstE2eRunGuard(stats_fn=s)
        assert g.check_pre(make_ctx()) is not None
        assert g.accept_override(_PROMISE,
                                 make_ctx(history=_write_history())) is True
        assert g.check_pre(make_ctx()) is None
        g.reset_turn()
        assert g.check_pre(make_ctx()) is not None

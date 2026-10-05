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

"""Unit tests for FirstE2eRunGuard — progress-ORDER checkpoints at 25%, 50%
and 75% of the enforced wall-clock budget (write-late / run-late / cold-consumer)."""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.first_e2e_run import (
    FirstE2eRunGuard,
    _MSG_25,
    _MSG_50,
    _MSG_75,
)


def make_ctx(tool_name="shell"):
    ctx = GuardContext()
    ctx.tool_name = tool_name
    return ctx


class _Stats:
    """Mutable stats source: set .pct=None to simulate no injected wall."""

    def __init__(self):
        self.pct = 0.0

    def __call__(self):
        if self.pct is None:
            return None
        budget = 1200.0
        elapsed = budget * self.pct / 100.0
        return {
            "elapsed": elapsed,
            "budget": budget,
            "remaining": budget - elapsed,
            "pct": self.pct,
        }


class TestFirstE2eRunSilence:
    def test_none_stats_is_silent(self):
        s = _Stats()
        s.pct = None
        g = FirstE2eRunGuard(stats_fn=s)
        assert g.check_pre(make_ctx()) is None
        assert g.check_post(make_ctx()) is None

    def test_stats_fn_none_is_silent(self):
        g = FirstE2eRunGuard(stats_fn=None)
        assert g.check_pre(make_ctx()) is None

    def test_stats_fn_exception_is_silent(self):
        def boom():
            raise RuntimeError("stats failed")

        g = FirstE2eRunGuard(stats_fn=boom)
        s_pct_holder = _Stats()
        s_pct_holder.pct = 30.0
        g._stats_fn = boom
        assert g.check_pre(make_ctx()) is None

    def test_no_tool_name_is_silent(self):
        g = FirstE2eRunGuard(stats_fn=_Stats())
        ctx = GuardContext()
        ctx.tool_name = None
        assert g.check_pre(ctx) is None

    def test_below_thresholds_is_silent(self):
        s = _Stats()
        s.pct = 24.9
        g = FirstE2eRunGuard(stats_fn=s)
        assert g.check_pre(make_ctx()) is None

    def test_boundary_25_exactly_blocks(self):
        s = _Stats()
        s.pct = 25.0
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx())
        assert v is not None and v.action == "block"
        assert v.reason == "first_e2e_run_25pct"


class TestFirstE2eRunBlocks:
    def test_25_blocks_once_per_turn(self):
        s = _Stats()
        s.pct = 26.0
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx())
        assert v is not None and v.action == "block"
        assert v.reason == "first_e2e_run_25pct"
        assert v.overridable is True
        # Same turn: silent (each checkpoint fires at most once per turn).
        assert g.check_pre(make_ctx()) is None

    def test_50_blocks_once_per_turn(self):
        s = _Stats()
        s.pct = 60.0
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx())
        assert v is not None and v.reason == "first_e2e_run_50pct"
        assert g.check_pre(make_ctx()) is None

    def test_75_blocks_once_per_turn(self):
        s = _Stats()
        s.pct = 76.0
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx())
        assert v is not None and v.action == "block"
        assert v.reason == "first_e2e_run_75pct"
        assert v.overridable is True
        # Same turn: silent after fired once.
        assert g.check_pre(make_ctx()) is None

    def test_75_boundary_exactly_blocks(self):
        s = _Stats()
        s.pct = 75.0
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx())
        assert v is not None and v.reason == "first_e2e_run_75pct"

    def test_jump_past_all_fires_only_75(self):
        s = _Stats()
        s.pct = 80.0
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx())
        assert v is not None and v.reason == "first_e2e_run_75pct"
        # All three thresholds marked spent; none fire on their own after a jump.
        assert g._fired == {75, 50, 25}
        assert g.check_pre(make_ctx()) is None

    def test_jump_past_both_fires_only_50(self):
        s = _Stats()
        s.pct = 70.0
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx())
        assert v is not None and v.reason == "first_e2e_run_50pct"
        # Both thresholds marked spent; 25 never fires on its own after a jump.
        assert g._fired == {50, 25}
        assert g.check_pre(make_ctx()) is None

    def test_reset_turn_rearms(self):
        s = _Stats()
        s.pct = 30.0
        g = FirstE2eRunGuard(stats_fn=s)
        assert g.check_pre(make_ctx()) is not None
        assert g.check_pre(make_ctx()) is None
        g.reset_turn()
        assert g.check_pre(make_ctx()) is not None


class TestFirstE2eRunMessageContent:
    def test_msg_25_write_through_anchors(self):
        assert "WRITE-THROUGH checkpoint" in _MSG_25
        assert "delivery path" in _MSG_25
        assert "skeleton" in _MSG_25
        assert "THIS STEP" in _MSG_25
        # Override contract: explicit a/b/c options are offered.
        assert "_override_reason" in _MSG_25
        assert "(a)" in _MSG_25 and "(b)" in _MSG_25 and "(c)" in _MSG_25
        # Task-agnostic: no named task or dataset.
        assert "doom" not in _MSG_25.lower()
        assert "mips" not in _MSG_25.lower()

    def test_msg_50_first_exercise_anchors(self):
        assert "FIRST-EXERCISE checkpoint" in _MSG_50
        assert "end-to-end" in _MSG_50
        assert "intermediate" in _MSG_50
        assert "_override_reason" in _MSG_50
        assert "(a)" in _MSG_50 and "(b)" in _MSG_50 and "(c)" in _MSG_50
        assert "90%" in _MSG_50
        # Cross-contamination guard: each message carries only its own face.
        assert "WRITE-THROUGH" not in _MSG_50
        assert "FIRST-EXERCISE" not in _MSG_25

    def test_msg_75_cold_consumer_evidence_binding(self):
        assert "COLD-CONSUMER checkpoint" in _MSG_75
        # the prediction-vs-run distinction is the whole point
        assert "predict" in _MSG_75.lower()
        assert "COMMAND you ran" in _MSG_75 or "output" in _MSG_75.lower()
        # evidence binding: paste command + output, not a bare yes/no
        assert "OUTPUT it produced" in _MSG_75
        assert "_override_reason" in _MSG_75
        assert "(a)" in _MSG_75 and "(b)" in _MSG_75 and "(c)" in _MSG_75
        # Cross-contamination guard: 75% carries its own face only.
        assert "WRITE-THROUGH" not in _MSG_75
        assert "FIRST-EXERCISE" not in _MSG_75
        # Task-agnostic: no named task or dataset.
        assert "doom" not in _MSG_75.lower()
        assert "mips" not in _MSG_75.lower()

    def test_verdict_message_combines_const_and_head(self):
        s = _Stats()
        s.pct = 26.0
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx())
        assert "WRITE-THROUGH checkpoint" in v.message
        assert "used" in v.message  # _fmt head appended

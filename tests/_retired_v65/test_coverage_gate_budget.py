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

"""Regression tests: the constraint-coverage gate must not consume the wall.

Measured failure this locks (four_m19 model-extraction-relu-logits, reward 0;
the SAME task scored 1 in four_m17 and four_m18):
  * the agent's log ends with `[failing check: acceptance_constraint_coverage_missing]`
    repeated 20 times, then
    `[TimeBudget] TIME IS UP — 101% ... (15m11s used)` and
    `TIME BUDGET EXHAUSTED: per-turn wall-clock budget (900s) used up`.
  * the grader then failed on CONTENT (`Failed to match rows: [2, 4, ...]`) because
    the deliverable was never written — the coverage sweep had eaten the turn.

The gate is still right when there IS budget: a per-constraint measurement sweep
is the only defence against banking a deliverable that violates a stated
constraint. What must change is its authority near the wire: past the wrap-up
threshold those tool calls belong to the deliverable.
"""
import pytest

from flagscale_agent.react.guard.verification import VerificationGuard


def _guard(pct=None, stats_fn=None):
    if stats_fn is None:
        def stats_fn():
            if pct is None:
                return None
            return {"elapsed": 1.0, "budget": 100.0, "remaining": 100.0 - pct, "pct": pct}
    return VerificationGuard(stats_fn=stats_fn), stats_fn


def test_gate_is_mandatory_when_budget_is_ample():
    g, _ = _guard(pct=10.0)
    assert g._coverage_gate_affordable() is True


def test_gate_is_mandatory_exactly_at_the_threshold():
    # 25% remaining == the documented floor -> still enforced (boundary is inclusive)
    g, _ = _guard(pct=75.0)
    assert g._coverage_gate_affordable() is True


def test_gate_degrades_when_the_wall_is_nearly_spent():
    g, _ = _guard(pct=90.0)
    assert g._coverage_gate_affordable() is False


def test_gate_degrades_past_the_wall():
    # the measured case: 101% spent when the last block fired
    g, _ = _guard(pct=101.0)
    assert g._coverage_gate_affordable() is False


def test_no_concrete_budget_never_degrades_the_gate():
    # standalone/interactive: no real deadline exists, so the gate keeps its teeth
    g = VerificationGuard(stats_fn=lambda: None)
    assert g._coverage_gate_affordable() is True
    g2 = VerificationGuard()  # stats_fn omitted entirely
    assert g2._coverage_gate_affordable() is True


def test_dead_or_malformed_clock_fails_closed_on_the_gate():
    # an unreadable clock is NOT evidence of time pressure
    def boom():
        raise RuntimeError("clock died")

    assert VerificationGuard(stats_fn=boom)._coverage_gate_affordable() is True
    assert VerificationGuard(stats_fn=lambda: {"pct": "half"})._coverage_gate_affordable() is True
    assert VerificationGuard(stats_fn=lambda: [1, 2, 3])._coverage_gate_affordable() is True


@pytest.mark.parametrize("pct,expected", [(0.0, True), (74.9, True), (75.1, False), (99.9, False)])
def test_threshold_boundary(pct, expected):
    g, _ = _guard(pct=pct)
    assert g._coverage_gate_affordable() is expected

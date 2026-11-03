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

"""a `hit` may not cover a half-miss.

Measured defect: the settlement arrived with
prediction_observed = "Half-hit, half-miss against the compound statement" and
was recorded as `hit`. The plan then counted a success the run did not have, and
the wrong half of the compound prediction became invisible — the pre-registration
instrument (which exists to make earlier claims falsifiable) certified itself.

The fix is in TaskPlan.settle so every caller is covered, and it is
one-directional: it can refuse a `hit`, it never forces one. Refusing costs the
agent one re-settle with honest wording; accepting an inflated hit corrupts the
record permanently.
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.plan import TaskPlan, _observed_reports_a_miss


@pytest.fixture
def plan(tmp_path):
    p = TaskPlan(str(tmp_path / "plans"))
    p.create(title="t", steps=["s1"])
    return p


def _pred(plan):
    plan.predict(statement="x should reach ~10", expected="~10", kind="k")
    preds = plan.get_active().get("predictions") or []
    assert preds, "predict produced no prediction"
    return preds[-1]["id"]


# ── classifier: the direction of the error matters ──────────────────────────

@pytest.mark.parametrize("observed", [
    "Half-hit, half-miss against the compound statement",   # the measured case
    "half-miss on the clamp clause; hit on the fold",
    "The run is inconclusive.",
    "4 of 7 checks failed",
    "the check could not run: not measured",
])
def test_text_reporting_a_miss_is_detected(observed):
    assert _observed_reports_a_miss(observed) is True


@pytest.mark.parametrize("observed", [
    "prompt_len()=14910 <= 15000, 15 passed",
    "132 passed in 1.94s",
    "~3350 passed",
    # 'missing' contains 'miss' — a substring rule would wrongly block this hit,
    # teaching the agent to reword the observation instead of recording the truth.
    "The missing placeholder was restored",
    # A correction narrative mentioning a past failure is not a present miss.
    "The earlier attempt failed; after fixing the root it then held",
    "",
])
def test_honest_hits_are_not_blocked(observed):
    assert _observed_reports_a_miss(observed) is False


# ── settle: the gate itself ─────────────────────────────────────────────────

def test_settle_refuses_hit_over_a_self_reported_miss(plan):
    pid = _pred(plan)
    with pytest.raises(ValueError) as ei:
        plan.settle(pid, "hit", "Half-hit, half-miss against the compound statement")
    assert "cannot be settled" in str(ei.value)
    assert "split the compound prediction" in str(ei.value)
    # The prediction must still be OPEN — a refused settle must not consume it.
    left = [p["id"] for p in plan.unsettled_predictions()]
    assert pid in left, "a refused settle must leave the prediction unsettled"


def test_settle_accepts_miss_with_the_same_text(plan):
    pid = _pred(plan)
    plan.settle(pid, "miss", "Half-hit, half-miss against the compound statement")
    rec = [p for p in plan.get_active()["predictions"] if p["id"] == pid][0]
    assert rec["outcome"] == "miss"


def test_settle_accepts_honest_hit(plan):
    pid = _pred(plan)
    plan.settle(pid, "hit", "measured 10.2 vs expected ~10; 15 passed")
    rec = [p for p in plan.get_active()["predictions"] if p["id"] == pid][0]
    assert rec["outcome"] == "hit"


def test_not_run_is_unaffected(plan):
    # Silencing a run must never be blocked by its own wording.
    pid = _pred(plan)
    plan.settle(pid, "not_run", "never ran — wall hit; not measured")
    rec = [p for p in plan.get_active()["predictions"] if p["id"] == pid][0]
    assert rec["outcome"] == "not_run"


def test_settle_without_observed_text_still_works(plan):
    # Legacy callers pass no observation; the rule must not break them.
    pid = _pred(plan)
    plan.settle(pid, "hit")
    rec = [p for p in plan.get_active()["predictions"] if p["id"] == pid][0]
    assert rec["outcome"] == "hit"

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

"""The reviewer/diverger dispatch is a CONSCIOUS TRADE-OFF, not unconditional.

The plan's fixed Frame step used to order a diverger UNCONDITIONALLY, and the
reviewer demand used to read as if it fired at every step_done, so a trivial
2-line edit was over-provisioned. A later revision priced the dispatch by an
abstract T0/T1/T2 "task tier", but that grading was opaque — a reader could not
tell what tier a task was without memorizing the definitions. The gate now asks
the agent to WEIGH the worker's BENEFIT against its TIME-BUDGET COST and DECIDE,
stating the decision either way — no tier label to memorize. The demand lives in
the two acceptance blocks (tools/plan_create.py) and the two prompt-level demand
sites (react/prompt.py). The reviewer demand is anchored at the plan's FIXED
Review step, NOT at each step_done.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
from flagscale_agent.react.tools.plan_create import (
    _FRAME_STEP_ACCEPTANCE,
    _REVIEW_STEP_ACCEPTANCE,
)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _no_tier_label(low: str) -> None:
    """No abstract T0/T1/T2 grading may survive in the shipped text."""
    assert "task tier" not in low
    for token in ("t0", "t1", "t2"):
        assert not re.search(r"\b" + token + r"\b", low), f"tier token {token} leaked"


def _no_unconditional_both_halves(low: str) -> None:
    """The dispatched half must read as CONDITIONAL, not a 'BOTH halves' mandate.

    Renaming tiers to a trade-off left the old unconditional 'do BOTH halves'
    phrase in place — the two halves then contradict (item 1 mandates, item 2
    prices). The self-half may be unconditional, but the phrase 'BOTH halves'
    must not survive.
    """
    assert "both halves" not in low, "unconditional 'BOTH halves' mandate leaked"


class TestTradeoffGateInAcceptanceBlocks:
    def test_frame_acceptance_names_the_tradeoff(self):
        low = _norm(" ".join(_FRAME_STEP_ACCEPTANCE))
        assert "a deliberate trade-off, not a mandate" in low
        assert "benefit" in low
        assert "time budget" in low
        assert "state the decision and why" in low
        _no_tier_label(low)
        _no_unconditional_both_halves(low)

    def test_review_acceptance_names_the_tradeoff(self):
        low = _norm(" ".join(_REVIEW_STEP_ACCEPTANCE))
        assert "a deliberate trade-off, not a mandate" in low
        assert "benefit" in low
        assert "time budget" in low
        assert "state the decision and why" in low
        _no_tier_label(low)

    def test_benefit_and_cost_are_both_spelled_out(self):
        # the agent must know WHY a worker is worth it, and WHAT it costs
        low = _norm(" ".join(_FRAME_STEP_ACCEPTANCE))
        assert "before any artifact exists" in low  # diverger benefit
        assert "whole deadline" in low  # cost
        lo_r = _norm(" ".join(_REVIEW_STEP_ACCEPTANCE))
        assert "fresh context" in lo_r  # reviewer benefit
        assert "whole deadline" in lo_r  # cost


class TestFrameAcceptanceLinksStillPresent:
    """The trade-off gate must not remove the frame-step discipline beside it."""

    def test_diverger_role_tagging_preserved(self):
        low = _norm(" ".join(_FRAME_STEP_ACCEPTANCE))
        assert 'role="diverger"' in low
        assert "leaf role" in low
        assert "fresh context" in low

    def test_no_task_specific_threshold_leak(self):
        low = _norm(" ".join(_FRAME_STEP_ACCEPTANCE))
        for w in ("minute", "hour", "gb", "second"):
            assert f" {w} " not in low, f"threshold word leaked: {w!r}"
        for w in ("gcode", "qemu", "sam-cell-seg"):
            assert not re.search(r"\b" + re.escape(w) + r"\b", low)


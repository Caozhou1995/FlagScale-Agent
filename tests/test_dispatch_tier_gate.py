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

"""The reviewer/diverger dispatch demand is TIER-PRICED, not unconditional.

The plan's fixed Frame step used to order a diverger UNCONDITIONALLY, and the
reviewer demand used to read as if it fired at every step_done, so a trivial
2-line edit was over-provisioned. The gate now lives in the two acceptance
blocks (tools/plan_create.py) and the two prompt-level demand sites
(react/prompt.py): the agent declares a task tier (T0/T1/T2) and the tier
decides how many workers (if any) are dispatched — a deliberate cost decision,
not template compliance. The reviewer demand is anchored at the plan's FIXED
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


class TestTierGateInAcceptanceBlocks:
    def test_frame_acceptance_carries_task_tier(self):
        low = _norm(" ".join(_FRAME_STEP_ACCEPTANCE))
        assert "task tier" in low
        # all three tiers named
        for token in ("t0", "t1", "t2"):
            assert f" {token} " in f" {low} ", f"tier {token} missing"
        # T0 = no worker; T1 = at most one reviewer; T2 = full
        assert "dispatch no worker" in low
        assert "at most one reviewer" in low
        assert "full diverger + reviewer" in low

    def test_review_acceptance_carries_task_tier(self):
        low = _norm(" ".join(_REVIEW_STEP_ACCEPTANCE))
        assert "task tier" in low
        assert "no reviewer" in low or "no " in low
        assert "at most one reviewer" in low

    def test_frame_tier_prices_the_diverger_not_mandates(self):
        low = _norm(" ".join(_FRAME_STEP_ACCEPTANCE))
        # the old unconditional "spawn a worker" mandate is reframed
        assert "price the dispatch, do not obey it mechanically" in low
        assert "declare one tier in plan notes" in low


class TestTierGateInPrompt:
    def test_reviewer_demand_is_priced_by_tier(self):
        p = _norm(SYSTEM_PROMPT_STATIC)
        assert "priced by task tier, not unconditional" in p
        assert "declare the task's tier in plan notes" in p

    def test_frame_diverger_demand_is_t2_only(self):
        p = _norm(SYSTEM_PROMPT_STATIC)
        assert "for t2, a dispatched diverger worker" in p


class TestFrameAcceptanceLinksStillPresent:
    """The tier gate must not remove the frame-step discipline it lives beside."""

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


class TestRoleOrderSeparation:
    """diverger and reviewer are DIFFERENT roles at DIFFERENT times.

    Frame (pre-artifact) dispatches the DIVERGER — the framing probe; a
    reviewer needs a FROZEN artifact and belongs at the review step. The Frame
    acceptance must not read as if Frame dispatches reviewers.
    """

    def test_frame_step_dispatches_diverger_not_reviewer(self):
        low = _norm(" ".join(_FRAME_STEP_ACCEPTANCE))
        # Frame names what IT dispatches: the diverger (pre-artifact probe)
        assert "what this frame step dispatches is the diverger" in low
        # and says a reviewer needs a FROZEN artifact at the fixed Review step
        assert "a reviewer needs a frozen artifact" in low
        assert "belongs at the review step" in low

    def test_reviewer_line_keeps_diverger_out(self):
        p = _norm(SYSTEM_PROMPT_STATIC)
        # the reviewer-demand line points the diverger elsewhere (Frame),
        # not inline in the reviewer tier text
        assert "the diverger is a frame-time dispatch" in p
        assert "not here" in p

    def test_order_documented_as_frame_then_fixed_review_step(self):
        p = _norm(SYSTEM_PROMPT_STATIC)
        # Frame half sequenced before the reviewer half, and the reviewer is
        # anchored at the plan's FIXED Review step — not at each step_done.
        assert "divergence at the frame step" in p
        assert "reviewer at the plan's fixed review step" in p
        assert "not fired at every step_done" in p

    def test_frame_and_review_roles_are_time_separated(self):
        p = _norm(SYSTEM_PROMPT_STATIC)
        # the two roles are explicitly placed at DIFFERENT times, not conflated
        assert "diverger at frame" in p
        assert "reviewer at the fixed review step" in p

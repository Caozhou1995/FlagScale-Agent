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

"""Only the leak guard survives from the old reviewer/diverger dispatch tests.

The plan's fixed Frame step used to order a diverger UNCONDITIONALLY, and the
reviewer demand used to read as if it fired at every step_done. Both dispatch
guidance blocks were later disabled outright — the Frame diverger and the
Review reviewer prompts are no longer in _FRAME_STEP_ACCEPTANCE /
_REVIEW_STEP_ACCEPTANCE — so the tests that asserted their wording no longer
have a subject. What remains worth guarding is that the frame-step text beside
them never picks up a task-specific threshold.
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


class TestFrameAcceptanceLinksStillPresent:
    """The frame-step discipline beside the dispatch guidance must stay leak-free."""

    def test_no_task_specific_threshold_leak(self):
        low = _norm(" ".join(_FRAME_STEP_ACCEPTANCE))
        for w in ("minute", "hour", "gb", "second"):
            assert f" {w} " not in low, f"threshold word leaked: {w!r}"
        for w in ("gcode", "qemu", "sam-cell-seg"):
            assert not re.search(r"\b" + re.escape(w) + r"\b", low)


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

"""Tests for the premortem discipline carried by the plan's fixed FRAME step.

The feature-keyed premortem inject in PlanGuard (and its workspace-detection
machinery, `_premortem_block`) was removed: the plan-framing discipline now
lives entirely in the fixed FIRST plan step's acceptance criteria, delivered by
plan_create() — one systematic home, no separate inject, no workspace probing.
"""

from flagscale_agent.react.tools.plan_create import (
    _FRAME_STEP_TITLE, _FRAME_STEP_ACCEPTANCE,
)


def _frame_text():
    return " ".join(_FRAME_STEP_ACCEPTANCE)


class TestFrameStepPremortem:
    def test_frame_step_exists(self):
        assert _FRAME_STEP_TITLE
        assert _FRAME_STEP_ACCEPTANCE

    def test_premortem_reference_artifacts(self):
        # The frame step tells the agent to diff its own convention against any
        # reference artifacts present, before inventing one.
        t = _frame_text()
        assert "reference artifacts" in t
        assert "expected_" in t or "golden" in t
        assert "diff" in t.lower()

    def test_premortem_harness_use_it_as_grader(self):
        # A present test/verify harness must be USED as the grader, not shadowed
        # by a hand-rolled check.
        t = _frame_text()
        assert "harness" in t
        assert "use it as the grader" in t.lower()

    def test_premortem_is_unconditional_guidance(self):
        # No workspace detection: the guidance is static text on the frame step.
        t = _frame_text().lower()
        assert "if reference artifacts are present" in t

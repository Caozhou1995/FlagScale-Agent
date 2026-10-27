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

"""``add_forbidden`` must count as real plan progress.

Why this is not cosmetic: the plan-update guard decides whether a call is a
SUBSTANTIVE response or an idle ping. An action that mutates the plan but is
missing from ``_PROGRESS_ACTIONS`` means an agent that appends a hard-won
prohibition — the single most useful thing it can record when it notices it is
stuck in a loop — gets no credit for it, and the guard keeps escalating. The
symmetric failure: an action that only *looks* like progress must not be able to
clear a stall. ``set_thinking`` is deliberately NOT in the progress set for that
reason (it is handled by the separate rebuilt-model path).
"""

import pytest

from flagscale_agent.react.guard.plan_update import PlanUpdateGuard


def test_add_forbidden_is_a_progress_action():
    assert "add_forbidden" in PlanUpdateGuard._PROGRESS_ACTIONS


def test_set_thinking_stays_out_of_the_progress_set():
    # Guard against someone "helpfully" widening the set: set_thinking has its
    # own clearance path and must stay distinct, or a re-framed sentence would
    # clear a stall the agent has not actually broken.
    assert "set_thinking" not in PlanUpdateGuard._PROGRESS_ACTIONS


def test_existing_progress_actions_are_unchanged():
    # A regression here silently disarms the stall detector for a real action.
    expected = {
        "step_done", "step_skip", "complete", "abandon",
        "add_steps", "update_acceptance", "batch", "add_forbidden",
    }
    assert PlanUpdateGuard._PROGRESS_ACTIONS == frozenset(expected)


def test_read_only_actions_are_not_progress():
    for a in ("memory_read", "memory_list", "recall", "plan_status"):
        assert a not in PlanUpdateGuard._PROGRESS_ACTIONS

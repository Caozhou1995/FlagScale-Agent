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

"""PlanGuard — complex task without plan detection.

Two activation modes:
1. Complexity judge fired → hard block at _PLAN_GATE_MAX_EXPLORATORY
2. Independent: warn at dynamic threshold, hard block at dynamic threshold

v2: Configurable thresholds.
exploratory calls before requiring a plan.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class PlanGuard(Guard):
    """Detects complex tasks without a plan and prompts plan creation.

    Uses tool_name to identify exploratory (read-only) calls.
    v2: Configurable thresholds.
    """

    name = "plan"
    priority = 35


    # Base thresholds
    _PLAN_GATE_MAX_EXPLORATORY_BASE = 6
    _PLAN_GATE_INDEPENDENT_WARN_BASE = 8
    _PLAN_GATE_INDEPENDENT_BLOCK_BASE = 12

    def __init__(self, task_plan=None):
        self._task_plan = task_plan
        self._complex_task_no_plan: bool = False
        self._pre_plan_tool_calls: int = 0
        self._consecutive_reads: int = 0
        self._block_count: int = 0  # track repeated blocks for escalation



    @property
    def _threshold_multiplier(self) -> float:
        """Get threshold multiplier. Higher = more tolerant."""
        return 1.0

    @property
    def _plan_gate_max_exploratory(self) -> int:
        return max(4, int(self._PLAN_GATE_MAX_EXPLORATORY_BASE * self._threshold_multiplier))

    @property
    def _plan_gate_independent_warn(self) -> int:
        return max(6, int(self._PLAN_GATE_INDEPENDENT_WARN_BASE * self._threshold_multiplier))

    @property
    def _plan_gate_independent_block(self) -> int:
        return max(8, int(self._PLAN_GATE_INDEPENDENT_BLOCK_BASE * self._threshold_multiplier))

    def _has_active_plan(self) -> bool:
        """Check if a plan already exists (active or paused)."""
        if self._task_plan is None:
            return False
        try:
            return self._task_plan.get_active() is not None
        except Exception:
            return False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        # Plan-related tools are always allowed
        if ctx.tool_name in ("plan_create", "memory_write"):
            return None

        # If a plan already exists, skip all plan-gate logic — the agent is
        # executing under a plan and should not be blocked for reading files.
        if self._has_active_plan():
            return None

        # Use tool_name to classify: read-only = exploratory
        from flagscale_agent.react.guard.utils import READ_ONLY_TOOLS
        if ctx.tool_name in READ_ONLY_TOOLS:
            self._consecutive_reads += 1
        else:
            self._consecutive_reads = 0

        self._pre_plan_tool_calls += 1

        # Mode 1: complexity judge fired → hard block at threshold
        if self._complex_task_no_plan:
            if self._pre_plan_tool_calls > self._plan_gate_max_exploratory:
                self._block_count += 1
                if self._block_count >= 3:
                    return GuardVerdict.escalate(
                        f"[PLAN GATE] Complex task blocked {self._block_count} times "
                        f"without plan creation. Create a plan or ask the user for guidance.",
                        reason="complex task no plan persistent",
                        category="plan_needed",
                    )
                return GuardVerdict.block(
                    f"[PLAN GATE] BLOCKED: {self._pre_plan_tool_calls} exploratory calls without a plan. "
                    f"Create a plan based on what you've gathered so far.",
                    reason="complex task no plan exceeded",
                    category="plan_needed",
                )

        # Mode 2: independent — soft warn, then hard block
        if self._consecutive_reads >= self._plan_gate_independent_block:
            self._block_count += 1
            if self._block_count >= 3:
                return GuardVerdict.escalate(
                    f"[PLAN GATE] Blocked {self._block_count} times without plan creation. "
                    f"Create a plan or ask the user for guidance.",
                    reason="independent plan threshold persistent",
                    category="plan_needed",
                )
            return GuardVerdict.block(
                f"[PLAN GATE] BLOCKED: {self._consecutive_reads} consecutive exploratory calls "
                f"without a plan. Create a plan to organize your approach.",
                reason="independent plan threshold exceeded",
                category="plan_needed",
            )

        if self._consecutive_reads >= self._plan_gate_independent_warn:
            return GuardVerdict.inject(
                f"\n[PLAN REMINDER] You've made {self._consecutive_reads} "
                f"exploratory calls without a plan. Consider calling plan_create "
                f"soon to organize your findings. "
                f"You will be BLOCKED at {self._plan_gate_independent_block} calls.",
                reason="plan independent warn threshold",
                category="plan_needed",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name in ("plan_create",):
            self._complex_task_no_plan = False
            self._pre_plan_tool_calls = 0
            self._consecutive_reads = 0
            self._block_count = 0
        return None

    def reset_turn(self):
        """Reset all counters at the start of a new user turn.

        This prevents state leaking between user messages — a fresh question
        should start with a clean slate for plan-gate detection.
        """
        self._pre_plan_tool_calls = 0
        self._consecutive_reads = 0
        self._block_count = 0
        self._complex_task_no_plan = False

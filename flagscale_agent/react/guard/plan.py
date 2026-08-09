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

"""PlanGuard — reminds agent to create a plan for long tasks."""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class PlanGuard(Guard):
    """Reminds agent to create a plan when working without one.

    After REMIND_THRESHOLD tool calls without an active plan, injects
    a one-time reminder. No blocking, no escalation.
    """

    name = "plan"
    priority = 35

    REMIND_THRESHOLD = 15

    def __init__(self, task_plan=None):
        self._task_plan = task_plan
        self._calls_without_plan = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        # Plan-related tools don't count
        if ctx.tool_name in ("plan_create", "plan_update", "plan_status"):
            return None

        # If plan exists, nothing to do
        if self._task_plan and self._task_plan.get_active():
            return None

        self._calls_without_plan += 1

        # Periodic reminder every REMIND_THRESHOLD calls
        if self._calls_without_plan % self.REMIND_THRESHOLD == 0:
            return GuardVerdict.inject(
                message=(
                    f"[Plan] {self._calls_without_plan} tool calls without a plan. "
                    f"Consider plan_create() to organize a long task."
                ),
                reason="plan_reminder",
                category="plan_needed",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name == "plan_create":
            self._calls_without_plan = 0
        return None

    def reset_turn(self):
        """New user message resets counter."""
        self._calls_without_plan = 0

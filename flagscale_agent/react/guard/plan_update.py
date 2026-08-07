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

"""PlanUpdateGuard — enforces plan updates after step completion."""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class PlanUpdateGuard(Guard):
    """Enforces plan_update after completing plan steps.

    Tracks when a plan exists and whether the agent has updated it recently.
    If the agent completes work without updating the plan, injects a reminder.
    """

    name = "plan_update"
    priority = 50


    REMIND_THRESHOLD = 20  # turns without plan_update before injecting reminder

    def __init__(self, task_plan):
        self._task_plan = task_plan
        self._last_update_turn = 0

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check if plan needs updating after tool execution."""
        active_plan = self._task_plan.get_active()
        if not active_plan:
            return None

        steps = active_plan.get("steps", [])
        if not steps:
            return None

        # Track plan_update/plan_create calls as "plan touched"
        if ctx.tool_name in ("plan_update", "plan_create"):
            self._last_update_turn = ctx.turn_count
            return None

        # Skip plan-related reads
        if ctx.tool_name == "plan_status":
            return None

        turns_elapsed = ctx.turn_count - self._last_update_turn

        if turns_elapsed >= self.REMIND_THRESHOLD:
            doing_steps = [s for s in steps if s.get("status") == "doing"]
            if doing_steps:
                step_id = doing_steps[0].get("id")
                return GuardVerdict.inject(
                    message=(
                        f"[PlanUpdate] Active step {step_id} not updated in {turns_elapsed} turns. "
                        f"Mark it done/skipped, or add notes to preserve context."
                    ),
                    reason="plan_not_updated",
                    category="plan_update",
                )

        return None

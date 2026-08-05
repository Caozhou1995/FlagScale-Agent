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

"""TrainingAttemptGuard — 2-Strike Rule (periodic).

Every 2 attempts within an unfinished experiment, blocks further launches
and requires root cause analysis before continuing.

Trigger: attempt_count % 2 == 0 AND attempt_count > 0 AND experiment not finalized.
Bypass: _override_reason (>=30 chars) OR finalize current experiment.

This guard does NOT use regex/keywords to judge success/failure.
It simply counts attempts — if you keep adding attempts without finalizing,
that itself indicates something needs analysis.
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import _is_flagscale_launch_command
from flagscale_agent.react.state_machine import AgentState


class TrainingAttemptGuard(Guard):
    """2-Strike: every STRIKE_INTERVAL attempts, block and require root cause analysis.

    Triggers at attempt #2, #4, #6, ... within a single unfinalized experiment.
    """

    name = "training_2strike"
    priority = 42  # After ExperimentGuard (40)
    activate_on_states = {AgentState.EXECUTING}
    overridable = True

    STRIKE_INTERVAL = 2  # Block every N attempts
    MIN_OVERRIDE_LENGTH = 30

    def __init__(self):
        super().__init__()
        self._current_experiment = ""
        self._attempt_count = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Block launch if at a strike point (attempt_count % STRIKE_INTERVAL == 0)."""
        # No experiment or no attempts yet — nothing to block
        if not self._current_experiment:
            return None
        if self._attempt_count == 0:
            return None
        # Only block at strike intervals
        if self._attempt_count % self.STRIKE_INTERVAL != 0:
            return None

        # Only block shell launch commands
        if ctx.tool_name != "shell":
            return None
        cmd = ctx.tool_args.get("command", "")
        if not _is_flagscale_launch_command(cmd):
            return None

        return GuardVerdict.block(
            f"[2-Strike] 实验 '{self._current_experiment}' 已连续 "
            f"{self._attempt_count} 次 attempt 未完成。"
            "请先分析根因再继续。\n"
            "解除方式：\n"
            "1. workspace_experiment(finalize) 结束当前实验，开新实验\n"
            "2. 提供 _override_reason（>=30字根因分析 + 修复方案）绕过",
            reason="2strike_blocked",
        )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Track experiment lifecycle via workspace_experiment calls."""
        if ctx.tool_name != "workspace_experiment":
            return None

        action = ctx.tool_args.get("action", "")

        if action == "create":
            # New experiment starts — reset
            name = ctx.tool_args.get("name", "")
            self._current_experiment = name
            self._attempt_count = 0

        elif action == "add_attempt":
            self._attempt_count += 1
            # Inject warning when hitting a strike point
            if (self._attempt_count > 0
                    and self._attempt_count % self.STRIKE_INTERVAL == 0):
                return GuardVerdict.inject(
                    f"[2-Strike] 实验 '{self._current_experiment}' "
                    f"已有 {self._attempt_count} 次 attempt。"
                    "下次 launch 将被 block，请先分析根因。",
                    reason="2strike_warning",
                )

        elif action == "finalize":
            # Experiment done — reset
            self._current_experiment = ""
            self._attempt_count = 0

        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Validate override reason: >=30 chars, not lazy."""
        if not reason or len(reason.strip()) < self.MIN_OVERRIDE_LENGTH:
            return False
        lazy = ("try again", "just do it", "ignore", "skip", "override")
        if reason.strip().lower() in lazy:
            return False
        return True

    def reset_new_turn(self):
        """State persists across turns — never auto-reset."""
        pass

    def reset_state(self):
        """Full reset — only on decay or override."""
        super().reset_state()
        self._current_experiment = ""
        self._attempt_count = 0

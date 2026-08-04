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

"""TrainingAttemptGuard — 2-Strike Rule.

After 2 consecutive training failures (recorded via workspace_experiment),
blocks further launches until agent demonstrates root cause analysis:
1. Read relevant source code (at least 2 read_file calls to .py files)
2. State a hypothesis (via plan_update notes or explicit statement)

This guard does NOT detect launches or failures itself — it relies on
ExperimentGuard's lifecycle enforcement to ensure results are recorded.
It only reacts to workspace_experiment(action='update_last_attempt') results.
"""

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


_FAILURE_KEYWORDS = ("fail", "error", "crash", "oom", "timeout", "abort", "killed")
_SUCCESS_KEYWORDS = ("success", "completed", "converged", "running", "pass")

# Simple launch keywords (shared with ExperimentGuard)
_LAUNCH_KEYWORDS = (
    "torchrun", "deepspeed", "flagscale", "train.py",
    "pretrain.py", "pretrain_", "run.py",
)


def _is_launch_command(cmd: str) -> bool:
    """Simple keyword check for training launch commands."""
    cmd_lower = cmd.lower()
    if cmd_lower.lstrip().startswith(("grep ", "cat ", "echo ", "find ", "ls ", "head ", "tail ")):
        return False
    return any(kw in cmd_lower for kw in _LAUNCH_KEYWORDS)


def _is_failure_result(result: str) -> bool:
    """Check if an update_last_attempt result indicates failure."""
    r = result.lower()
    # If any success keyword present and no failure keyword, it's success
    has_success = any(kw in r for kw in _SUCCESS_KEYWORDS)
    has_failure = any(kw in r for kw in _FAILURE_KEYWORDS)
    if has_failure:
        return True
    if has_success:
        return False
    # Ambiguous — treat as failure to be safe
    return True


class TrainingAttemptGuard(Guard):
    """2-Strike Rule: blocks after 2 consecutive failures until root cause analysis.

    Relies on ExperimentGuard to enforce that results are recorded via
    workspace_experiment(action='update_last_attempt', result='...').
    """

    name = "training_2strike"
    priority = 42  # After ExperimentGuard (40)
    activate_on_states = {AgentState.EXECUTING}
    overridable = True

    STRIKE_THRESHOLD = 2
    SOURCE_READ_REQUIREMENT = 2  # Must read at least 2 source files

    def __init__(self):
        self._consecutive_failures = 0
        self._is_blocked = False
        self._source_reads_since_block = 0
        self._hypothesis_declared = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Block launch if 2-strike triggered and requirements not met."""
        if not self._is_blocked:
            return None

        # Only block shell launch commands
        if ctx.tool_name != "shell":
            return None
        cmd = ctx.tool_args.get("command", "")
        if not _is_launch_command(cmd):
            return None

        # Check if unblock conditions met
        if self._source_reads_since_block >= self.SOURCE_READ_REQUIREMENT:
            if self._hypothesis_declared:
                # Unblock — agent has done analysis
                self._is_blocked = False
                self._source_reads_since_block = 0
                self._hypothesis_declared = False
                return None
            else:
                return GuardVerdict.block(
                    "[2-Strike] Source code read, but no hypothesis stated. "
                    "Record your diagnosis in plan notes or experiment before relaunching.",
                    reason="2strike_no_hypothesis",
                )

        return GuardVerdict.block(
            f"[2-Strike] BLOCKED: {self._consecutive_failures} consecutive failures. "
            f"Read source code to diagnose root cause before retrying "
            f"({self._source_reads_since_block}/{self.SOURCE_READ_REQUIREMENT} reads done). "
            f"Then state your hypothesis.",
            reason="2strike_blocked",
        )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Track failure results and source reads."""
        # Track update_last_attempt results
        if ctx.tool_name == "workspace_experiment":
            action = ctx.tool_args.get("action", "")
            if action == "update_last_attempt":
                result = ctx.tool_args.get("result", "")
                if _is_failure_result(result):
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self.STRIKE_THRESHOLD:
                        self._is_blocked = True
                        self._source_reads_since_block = 0
                        self._hypothesis_declared = False
                        return GuardVerdict.inject(
                            f"[2-Strike] {self._consecutive_failures} consecutive failures. "
                            "Read relevant source code and state your hypothesis before "
                            "launching again.",
                            reason="2strike_warning",
                        )
                else:
                    # Success resets the counter
                    self._consecutive_failures = 0
                    self._is_blocked = False

        # Track source code reads (when blocked)
        if self._is_blocked and ctx.tool_name == "read_file":
            path = ctx.tool_args.get("path", "")
            if path.endswith(".py"):
                self._source_reads_since_block += 1

        # Track hypothesis declaration via plan_update notes
        if self._is_blocked and ctx.tool_name == "plan_update":
            notes = ctx.tool_args.get("notes", "")
            if notes and len(notes) > 20:  # Substantive note
                self._hypothesis_declared = True

        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Reject short/lazy override reasons. Require meaningful explanation."""
        # Minimum length requirement
        if len(reason) < 15:
            return False
        # Reject common lazy phrases
        lazy_patterns = ["try again", "just do it", "ignore this", "override"]
        reason_lower = reason.lower()
        if any(pattern in reason_lower for pattern in lazy_patterns):
            return False
        return True

    def reset_new_turn(self):
        """State persists across turns — never auto-reset."""
        pass

    def reset_state(self):
        """Full reset — only on decay or override."""
        super().reset_state()
        self._consecutive_failures = 0
        self._is_blocked = False
        self._source_reads_since_block = 0
        self._hypothesis_declared = False

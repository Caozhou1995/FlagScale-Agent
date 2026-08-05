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

"""TrainingRuntimeGuard — monitor enforcement after launch (inject-escalation type).

After training launch, injects reminders to call flagscale_train_monitor().
The framework's escalate_after mechanism escalates inject → block if ignored.

Does NOT manage:
- Experiment recording (→ ExperimentGuard)
- Failure counting (→ TrainingAttemptGuard)
- Heartbeat/periodic monitoring (removed in v6 — LLM handles this naturally)
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.guard.utils import _is_flagscale_launch_command
from flagscale_agent.react.state_machine import AgentState


class TrainingRuntimeGuard(Guard):
    """Monitor enforcement after training launch (inject-escalation type).

    The framework's escalate_after mechanism will escalate to block if ignored.
    """

    name = "training_runtime"
    priority = 50
    activate_on_states = {AgentState.EXECUTING}
    overridable = True
    escalate_after = 3  # Framework escalates inject → block after 3 ignores

    def __init__(self):
        super().__init__()
        self._awaiting_monitor: bool = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        if not self._awaiting_monitor:
            return None

        # Monitor call → clear gate
        if ctx.tool_name == "flagscale_train_monitor":
            self._awaiting_monitor = False
            return None

        # Allow diagnostic and recording tools
        if ctx.tool_name in ("plan_update", "workspace_experiment", "read_file",
                             "memory_write", "memory_read", "memory_list",
                             "evict", "evict_list", "recall"):
            return None

        # Allow read-only shell commands
        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "")
            if isinstance(cmd, str) and self._is_read_only_shell(cmd):
                return None

        return GuardVerdict.inject(
            "[TrainingRuntime] Training launched but not monitored. "
            "Call flagscale_train_monitor(output_dir='...'). "
            "Read-only commands (nvidia-smi, ps, cat, ls) are allowed.",
            reason="monitor_required",
        )

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Monitor call clears gate (redundant safety with check_pre)
        if ctx.tool_name == "flagscale_train_monitor":
            self._awaiting_monitor = False

        # Detect training launch → activate gate
        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "")
            if isinstance(cmd, str) and _is_flagscale_launch_command(cmd):
                self._awaiting_monitor = True

        return None

    def reset_new_turn(self):
        """State persists across turns — never auto-reset."""
        pass

    def reset_state(self):
        """Full reset."""
        super().reset_state()
        self._awaiting_monitor = False

    @staticmethod
    def _is_read_only_shell(cmd: str) -> bool:
        """Check if a shell command is read-only (safe during monitor gate)."""
        cmd_stripped = cmd.strip().lower()
        read_only_prefixes = (
            "cat ", "head ", "tail ", "less ", "grep ", "find ", "ls ",
            "wc ", "nvidia-smi", "ps ", "pgrep ", "top ", "df ", "du ",
            "ssh ", "ping ", "curl ", "wget ",
        )
        return cmd_stripped.startswith(read_only_prefixes)

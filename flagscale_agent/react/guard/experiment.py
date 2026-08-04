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

"""ExperimentGuard — enforces experiment recording discipline.

Lifecycle:
  create experiment → add_attempt (with change description) → launch → monitor
  → update_last_attempt (with result) → add_attempt → launch → ...

Rules:
  1. Before first launch: must have create + add_attempt → else BLOCK
  2. Before subsequent launch: must have update_last_attempt for previous run
     AND add_attempt for new run → else BLOCK
  3. No complex regex. Simple keyword detection for launch commands.
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


# Simple launch keywords — if any appears in a shell command, it's likely a launch
_LAUNCH_KEYWORDS = (
    "torchrun",
    "deepspeed",
    "flagscale",
    "train.py",
    "pretrain.py",
    "pretrain_",
    "run.py",
)


def _is_launch_command(cmd: str) -> bool:
    """Simple keyword check for training launch commands."""
    cmd_lower = cmd.lower()
    # Must be a substantive command, not just grep/cat/echo referencing keywords
    if cmd_lower.lstrip().startswith(("grep ", "cat ", "echo ", "find ", "ls ", "head ", "tail ")):
        return False
    return any(kw in cmd_lower for kw in _LAUNCH_KEYWORDS)


class ExperimentGuard(Guard):
    """Enforces experiment recording discipline throughout training lifecycle.

    State machine:
      IDLE → (create+add_attempt) → READY → (launch) → LAUNCHED
      LAUNCHED → (update_last_attempt) → RESULT_RECORDED → (add_attempt) → READY
    """

    name = "experiment_lifecycle"
    priority = 40  # High priority — blocks before other guards
    activate_on_states = {AgentState.EXECUTING}
    overridable = True

    def __init__(self, experiment_manager=None):
        self._experiment_manager = experiment_manager
        # Lifecycle state
        self._experiment_created = False
        self._attempt_added = False
        self._result_pending = False  # True after launch, cleared by update_last_attempt
        self._launched_without_result = False  # True if launched and no result recorded yet

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Block launch commands if experiment lifecycle not followed."""
        # Track workspace_experiment calls (pre-phase catches same-batch scenarios)
        if ctx.tool_name == "workspace_experiment":
            self._handle_experiment_call(ctx.tool_args)
            return None

        # Only check shell commands
        if ctx.tool_name != "shell":
            return None

        cmd = ctx.tool_args.get("command", "")
        if not _is_launch_command(cmd):
            return None

        # --- Enforcement ---

        # Rule 2: Previous run result not recorded
        if self._result_pending:
            return GuardVerdict.block(
                "[Experiment] BLOCKED: Previous training result not recorded. "
                "Call workspace_experiment(action='update_last_attempt', result='...') "
                "to record what happened, then add_attempt for the new run.",
                reason="result_not_recorded",
            )

        # Rule 1: No experiment created
        if not self._experiment_created:
            return GuardVerdict.block(
                "[Experiment] BLOCKED: No experiment record. "
                "Call workspace_experiment(action='create', name='...', purpose='...') "
                "then workspace_experiment(action='add_attempt', ...) before launching.",
                reason="no_experiment",
            )

        # Rule 1: No attempt added
        if not self._attempt_added:
            return GuardVerdict.block(
                "[Experiment] BLOCKED: No attempt recorded for this run. "
                "Call workspace_experiment(action='add_attempt', change='...') "
                "before launching.",
                reason="no_attempt",
            )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Track launch and result events."""
        # Track workspace_experiment calls
        if ctx.tool_name == "workspace_experiment":
            self._handle_experiment_call(ctx.tool_args)
            return None

        # Detect successful launch (shell with launch command that didn't error)
        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "")
            if _is_launch_command(cmd):
                # Mark as launched — next launch requires update_last_attempt
                self._result_pending = True
                self._attempt_added = False  # Consume the attempt

        # Detect training result from monitor
        if ctx.tool_name in ("flagscale_train_monitor", "parse_training_metrics"):
            # Monitor was called — if result_pending, remind to record
            if self._result_pending and ctx.tool_result:
                return GuardVerdict.inject(
                    "[Experiment] Training result observed. Record it with "
                    "workspace_experiment(action='update_last_attempt', result='...').",
                    reason="record_result_reminder",
                )

        return None

    def _handle_experiment_call(self, tool_args: dict):
        """Update lifecycle state based on workspace_experiment calls."""
        action = tool_args.get("action", "")
        if action == "create":
            self._experiment_created = True
        elif action == "add_attempt":
            self._attempt_added = True
        elif action == "update_last_attempt":
            self._result_pending = False

    def reset_new_turn(self):
        """Lifecycle state persists across turns. Never reset."""
        pass

    def reset_state(self):
        """Full reset — only on decay or override."""
        super().reset_state()
        self._experiment_created = False
        self._attempt_added = False
        self._result_pending = False

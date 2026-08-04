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

"""TrainingRuntimeGuard — monitor enforcement and heartbeat.

Responsibilities:
1. After training launch: BLOCK until monitor is called
2. Heartbeat: periodic reminders to check training progress
3. Detect training stopped (from monitor output or GPU idle)
4. GPU zombie detection

Does NOT manage:
- Experiment recording (→ ExperimentGuard)
- Failure counting or source read enforcement (→ TrainingAttemptGuard)
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict
from flagscale_agent.react.state_machine import AgentState


# Simple launch keywords
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


class TrainingRuntimeGuard(Guard):
    """Monitor enforcement and training heartbeat.

    After detecting a training launch, blocks all non-diagnostic actions
    until flagscale_train_monitor is called. Then periodically reminds
    to check training health.
    """

    name = "training_runtime"
    priority = 50
    activate_on_states = {AgentState.EXECUTING}
    overridable = True

    # Thresholds
    _MONITOR_GATE_MAX_BLOCKS = 5
    _HEARTBEAT_MONITOR_INTERVAL = 4  # turns between monitor reminders
    _HEARTBEAT_GPU_CHECK_INTERVAL = 6  # turns between GPU check reminders

    def __init__(self):
        self._awaiting_monitor: bool = False
        self._monitor_gate_block_count: int = 0
        self._training_started: bool = False
        self._turns_since_last_monitor: int = 0
        self._turns_since_last_gpu_check: int = 0
        self._last_launch_output_dir: str = ""

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        # --- Monitor enforcement gate ---
        if self._awaiting_monitor:
            if ctx.tool_name == "flagscale_train_monitor":
                self._awaiting_monitor = False
                self._monitor_gate_block_count = 0
                return None
            # Allow diagnostic and recording tools
            if ctx.tool_name in ("plan_update", "workspace_experiment", "read_file",
                                 "memory_write", "memory_read", "memory_list"):
                return None
            # Allow read-only shell commands
            if ctx.tool_name == "shell":
                cmd = ctx.tool_args.get("command", "")
                if isinstance(cmd, str) and self._is_read_only_shell(cmd):
                    return None

            self._monitor_gate_block_count += 1
            if self._monitor_gate_block_count >= self._MONITOR_GATE_MAX_BLOCKS:
                # Give up blocking after too many attempts
                self._awaiting_monitor = False
                self._monitor_gate_block_count = 0
                return None

            return GuardVerdict.block(
                "[TrainingRuntime] BLOCKED: Monitor training before doing other work. "
                "Call flagscale_train_monitor(output_dir='...'). "
                "Read-only commands (nvidia-smi, ps, cat, ls) are allowed.",
                reason="monitor_required",
            )

        # --- Heartbeat reminders ---
        if self._training_started and not self._awaiting_monitor:
            if self._turns_since_last_monitor >= self._HEARTBEAT_MONITOR_INTERVAL:
                self._turns_since_last_monitor = 0
                return GuardVerdict.inject(
                    "[HEARTBEAT] Training running but not monitored for "
                    f"{self._HEARTBEAT_MONITOR_INTERVAL} turns. "
                    "Check progress with flagscale_train_monitor.",
                    reason="heartbeat_monitor",
                )
            if self._turns_since_last_gpu_check >= self._HEARTBEAT_GPU_CHECK_INTERVAL:
                self._turns_since_last_gpu_check = 0
                return GuardVerdict.inject(
                    "[HEARTBEAT] Check GPU utilization — if 0% with active process, "
                    "training may be hung.",
                    reason="heartbeat_gpu",
                )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # --- Track monitor calls ---
        if ctx.tool_name == "flagscale_train_monitor":
            self._turns_since_last_monitor = 0
            self._turns_since_last_gpu_check = 0
            if ctx.tool_result and self._training_started:
                if self._detect_training_stopped(ctx.tool_result):
                    self._training_started = False

        # --- Track GPU checks ---
        if ctx.tool_name == "shell" and ctx.tool_result and self._training_started:
            cmd = ctx.tool_args.get("command", "")
            if isinstance(cmd, str) and "nvidia-smi" in cmd:
                self._turns_since_last_gpu_check = 0
                if self._detect_gpu_idle(ctx.tool_result):
                    self._training_started = False

        # --- Detect training launch ---
        if ctx.tool_name == "shell":
            cmd = ctx.tool_args.get("command", "")
            if isinstance(cmd, str) and _is_launch_command(cmd):
                self._awaiting_monitor = True
                self._training_started = True
                self._turns_since_last_monitor = 0
                self._turns_since_last_gpu_check = 0

        return None

    def reset_new_turn(self):
        """Increment heartbeat counters. Lifecycle state persists."""
        if self._training_started:
            self._turns_since_last_monitor += 1
            self._turns_since_last_gpu_check += 1

    def reset_state(self):
        """Full reset."""
        super().reset_state()
        self._awaiting_monitor = False
        self._monitor_gate_block_count = 0
        self._training_started = False
        self._turns_since_last_monitor = 0
        self._turns_since_last_gpu_check = 0

    @staticmethod
    def _detect_training_stopped(monitor_output: str) -> bool:
        """Detect if monitor output indicates training has stopped."""
        indicators = (
            "training_started.*false",
            "all.*rank.*error",
            "Process.*exited",
            "No active training",
            "no running process",
        )
        text = monitor_output.lower()
        return any(re.search(p, text, re.I) for p in indicators)

    @staticmethod
    def _detect_gpu_idle(nvidia_smi_output: str) -> bool:
        """Detect if all GPUs show 0% utilization (training likely stopped)."""
        lines = nvidia_smi_output.strip().split("\n")
        util_values = re.findall(r"(\d+)\s*%", nvidia_smi_output)
        if not util_values:
            return False
        # If ALL GPUs at 0%, training stopped
        return all(int(v) == 0 for v in util_values)

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

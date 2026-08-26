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

"""Tests for process health detection."""

import pytest

from flagscale_agent.react.tools.process_health import (
    detect_output_anomalies,
    should_kill_process,
)


class TestDetectOutputAnomalies:
    def test_oom_killed_detected(self):
        output = "make: *** [target] Error 137\nKilled\n"
        result = detect_output_anomalies(output)
        assert result['oom_killed'] is True
        assert result['killed_count'] == 1

    def test_multiple_killed(self):
        output = "Killed\ncompiling...\nKilled\nKilled\n"
        result = detect_output_anomalies(output)
        assert result['killed_count'] == 3

    def test_repeated_errors(self):
        output = "Error: something\nError: something\nError: something\n"
        result = detect_output_anomalies(output)
        assert result['repeated_errors'] is True

    def test_clean_output(self):
        output = "Building project...\ncompiling file.c\n"
        result = detect_output_anomalies(output)
        assert result['oom_killed'] is False
        assert result['killed_count'] == 0
        assert result['repeated_errors'] is False


class TestShouldKillProcess:
    def test_zombie_process(self):
        should_kill, reason = should_kill_process(
            elapsed_seconds=100,
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': True, 'cpu_percent': 0, 'num_children': 0, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=False,
            prev_num_children=0,
        )
        assert should_kill is True
        assert 'zombie' in reason.lower()

    def test_oom_killed_twice(self):
        should_kill, reason = should_kill_process(
            elapsed_seconds=100,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 50, 'num_children': 4, 'children_alive': 4},
            output_anomalies={'oom_killed': True, 'killed_count': 2, 'repeated_errors': False},
            had_children=True,
            prev_num_children=4,
        )
        assert should_kill is True
        assert 'OOM' in reason

    def test_oom_children_disappeared(self):
        # NEW: OOM detected by child count drop (kernel killed 5 coqc processes)
        should_kill, reason = should_kill_process(
            elapsed_seconds=100,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 30, 'num_children': 3, 'children_alive': 3},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=8,  # 8 -> 3: 5 children disappeared suddenly
        )
        assert should_kill is True
        assert 'OOM' in reason
        assert 'child_drop' in reason

    def test_all_children_dead(self):
        should_kill, reason = should_kill_process(
            elapsed_seconds=150,
            output_changed=False,
            stall_count=5,
            proc_health={'zombie': False, 'cpu_percent': 0.1, 'num_children': 4, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=4,
        )
        assert should_kill is True
        assert 'child processes exited' in reason.lower()

    def test_cpu_zero_long_stall(self):
        should_kill, reason = should_kill_process(
            elapsed_seconds=200,
            output_changed=False,
            stall_count=7,  # 7 * 30s = 210s > 3min
            proc_health={'zombie': False, 'cpu_percent': 0.0, 'num_children': 0, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=False,
            prev_num_children=0,
        )
        assert should_kill is True
        assert '0% CPU' in reason

    def test_timeout_20min(self):
        should_kill, reason = should_kill_process(
            elapsed_seconds=1300,  # >20min
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 50, 'num_children': 2, 'children_alive': 2},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=2,
        )
        assert should_kill is True
        assert '20-minute timeout' in reason

    def test_healthy_process(self):
        should_kill, reason = should_kill_process(
            elapsed_seconds=100,
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 50, 'num_children': 2, 'children_alive': 2},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=2,
        )
        assert should_kill is False

    def test_empty_output_triggers_stall_then_kill(self):
        # Regression test for pitfall/flagscale_agent/empty_output_stall_detection_disabled
        # Commands with no streaming output (e.g., `apt-get ... | tail -5`, silent network waits)
        # should accumulate stall_count and trigger hard indicator #4 at 3min
        should_kill, reason = should_kill_process(
            elapsed_seconds=190,  # >180s
            output_changed=False,  # empty output = not changed
            stall_count=7,  # >=6 (7*30s = 3m30s of stalling)
            proc_health={'zombie': False, 'cpu_percent': 0.0, 'num_children': 1, 'children_alive': 1},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=1,
        )
        assert should_kill is True
        assert "0% CPU with no output" in reason
        assert "210s" in reason  # stall_count * 30 = 7*30 = 210s

    def test_oom_killed_once_not_enough(self):
        # Need 2+ kills to trigger
        should_kill, reason = should_kill_process(
            elapsed_seconds=100,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 50, 'num_children': 4, 'children_alive': 4},
            output_anomalies={'oom_killed': True, 'killed_count': 1, 'repeated_errors': False},
            had_children=True,
            prev_num_children=4,
        )
        assert should_kill is False

    def test_children_dead_but_recent_output(self):
        # Output still changing, don't kill yet
        should_kill, reason = should_kill_process(
            elapsed_seconds=150,
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 0.1, 'num_children': 4, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=4,
        )
        assert should_kill is False

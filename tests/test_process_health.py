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

    def test_child_drop_alone_does_not_kill(self):
        # REDESIGN (support B): a child-count drop with NO OOM evidence
        # (killed_count=0, no memory pressure) is normal worker convergence,
        # NOT OOM. This is the caffe-cifar-10 false-kill case: 5→2 workers
        # exiting normally must not be convicted.
        should_kill, reason = should_kill_process(
            elapsed_seconds=120,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 30, 'num_children': 2, 'children_alive': 2},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=5,  # 5 -> 2
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,   # no real memory pressure
        )
        assert should_kill is False

    def test_child_drop_with_oom_evidence_kills(self):
        # support B: drop >= half AND no live children AND real OOM evidence
        # (killed in output) → convict.
        should_kill, reason = should_kill_process(
            elapsed_seconds=120,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 5, 'num_children': 2, 'children_alive': 0},
            output_anomalies={'oom_killed': True, 'killed_count': 1, 'repeated_errors': False},
            had_children=True,
            prev_num_children=8,  # 8 -> 2, dropped more than half
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is True
        assert 'OOM likely' in reason

    def test_child_drop_with_mem_pressure_kills(self):
        # support B: same drop, evidence via system memory pressure instead of output
        should_kill, reason = should_kill_process(
            elapsed_seconds=120,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 5, 'num_children': 2, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=8,
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=True,
        )
        assert should_kill is True
        assert 'OOM likely' in reason

    def test_liveness_veto_blocks_child_drop_kill(self):
        # support A: even with drop + no live children + memory pressure, if the
        # remaining process is burning CPU time, the liveness veto wins.
        should_kill, reason = should_kill_process(
            elapsed_seconds=120,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 5, 'num_children': 2, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=8,
            progress_signals={'cpu_time_delta': 2.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=True,
        )
        assert should_kill is False

    def test_liveness_veto_does_not_block_oom_hard_evidence(self):
        # OOM hard evidence (Killed x2) fires even if cpu_time is rising.
        should_kill, reason = should_kill_process(
            elapsed_seconds=120,
            output_changed=False,
            stall_count=2,
            proc_health={'zombie': False, 'cpu_percent': 40, 'num_children': 4, 'children_alive': 4},
            output_anomalies={'oom_killed': True, 'killed_count': 2, 'repeated_errors': False},
            had_children=True,
            prev_num_children=4,
            progress_signals={'cpu_time_delta': 5.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is True
        assert 'OOM' in reason

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

    def test_soft_timeout_20min_does_not_hard_kill(self):
        # support C: 20min soft cap no longer hard-kills — deferred to LLM judge.
        # A heavy compile making progress must survive past 20min.
        should_kill, reason = should_kill_process(
            elapsed_seconds=1300,  # >20min <60min
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 50, 'num_children': 2, 'children_alive': 2},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=2,
            progress_signals={'cpu_time_delta': 3.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is False

    def test_soft_timeout_no_progress_still_defers(self):
        # 20min<elapsed<60min with no progress: still hand to judge, not hard-kill.
        should_kill, reason = should_kill_process(
            elapsed_seconds=1500,
            output_changed=True,  # output moving, but no cpu/io progress
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 0.2, 'num_children': 1, 'children_alive': 1},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=1,
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is False

    def test_hard_timeout_60min_kills(self):
        # support C: hard cap 60min with no *progress* → kill. Output is moving
        # and children alive (so earlier heuristics #3/#4 skip), but cpu/io/rss
        # deltas are zero — output churns without real work. Only the 60min cap
        # catches this.
        should_kill, reason = should_kill_process(
            elapsed_seconds=3700,  # >60min
            output_changed=True,
            stall_count=0,
            proc_health={'zombie': False, 'cpu_percent': 0.2, 'num_children': 1, 'children_alive': 1},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=True,
            prev_num_children=1,
            progress_signals={'cpu_time_delta': 0.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is True
        assert '60-minute' in reason

    def test_busy_loop_survives_until_hard_timeout(self):
        # Known tradeoff (doc §5): a busy-loop burning CPU is judged healthy by
        # the liveness veto and survives until the 60min hard cap. Document the
        # behavior so it isn't mistaken for a bug.
        should_kill, reason = should_kill_process(
            elapsed_seconds=1800,  # 30min, past soft cap
            output_changed=False,
            stall_count=20,
            proc_health={'zombie': False, 'cpu_percent': 100, 'num_children': 0, 'children_alive': 0},
            output_anomalies={'oom_killed': False, 'killed_count': 0, 'repeated_errors': False},
            had_children=False,
            prev_num_children=0,
            progress_signals={'cpu_time_delta': 15.0, 'io_bytes_delta': 0, 'rss_delta': 0},
            mem_pressure=False,
        )
        assert should_kill is False  # liveness veto — intentional (see doc §5)

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

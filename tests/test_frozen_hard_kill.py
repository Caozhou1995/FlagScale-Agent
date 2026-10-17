"""
Regression test for the dead-code bug where the frozen-output advisory
(line 206) shadowed the hard kill (line 241) because both had the same
condition (stall_count >= 6, elapsed > 120).

After the 2026-10-17 advisory-downgrade:
- stall_count >= 6, elapsed > 120  → advisory (False, msg)
- stall_count >= 12, elapsed > 300 → advisory (False, msg) — no hard kill
  remains for frozen output; judgment is surfaced to the LLM every poll.
"""
from flagscale_agent.react.tools.process_health import should_kill_process


def _health(cpu=50.0, children=1, children_alive=1, zombie=False, mem=100.0):
    return {
        'alive': True, 'zombie': zombie,
        'cpu_percent': cpu, 'memory_mb': mem,
        'num_children': children, 'children_alive': children_alive,
        'cpu_time': 10.0, 'io_bytes': 0,
    }


def _anomalies():
    return {
        'oom_killed': False, 'killed_count': 0,
        'segfault': False, 'memory_error': False,
    }


def test_advisory_at_3min_not_killed():
    """At stall_count=6 (~3min), should return advisory (False), not kill."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=200, output_changed=False, stall_count=6,
        proc_health=_health(cpu=80), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals={'cpu_time_delta': 5.0, 'io_bytes_delta': 0, 'rss_delta': 0},
    )
    assert should_kill is False, f"Should be advisory, not kill. Got should_kill={should_kill}"
    assert 'frozen' in reason.lower(), f"Reason should mention frozen: {reason}"


def test_advisory_at_6min_not_killed():
    """At stall_count=12 (~6min), frozen output is advisory (False), not a kill."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=400, output_changed=False, stall_count=12,
        proc_health=_health(cpu=80), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals={'cpu_time_delta': 5.0, 'io_bytes_delta': 0, 'rss_delta': 0},
    )
    assert should_kill is False, f"Frozen output must be advisory, not kill. Got should_kill={should_kill}"
    assert 'advisory' in reason.lower(), f"Reason should be the downgrade advisory: {reason}"
    assert 'frozen' in reason.lower(), f"Reason should mention frozen: {reason}"


def test_advisory_fires_even_with_high_cpu():
    """Frozen output with high CPU (stockfish) is still advisory, never a kill."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=400, output_changed=False, stall_count=12,
        proc_health=_health(cpu=99.0), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals={'cpu_time_delta': 50.0, 'io_bytes_delta': 0, 'rss_delta': 0},
    )
    assert should_kill is False, "Frozen output with high CPU must be advisory, not kill"
    assert 'advisory' in reason.lower(), f"Reason should be the downgrade advisory: {reason}"


def test_no_kill_when_output_changing():
    """When output IS changing, neither advisory nor kill should fire from frozen rules."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=400, output_changed=True, stall_count=0,
        proc_health=_health(cpu=80), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals={'cpu_time_delta': 5.0, 'io_bytes_delta': 0, 'rss_delta': 0},
    )
    assert should_kill is False, "Should not kill when output is changing"


def test_advisory_fires_before_liveness_veto():
    """Advisory must fire even when making_progress (liveness veto) is True."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=200, output_changed=False, stall_count=6,
        proc_health=_health(cpu=80), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals={'cpu_time_delta': 10.0, 'io_bytes_delta': 0, 'rss_delta': 0},
    )
    assert should_kill is False, "Advisory should not hard-kill"
    assert reason, "Advisory should have a reason message"


def test_advisory_at_6min_with_no_progress():
    """Frozen output with no progress signals is still advisory (False)."""
    should_kill, reason = should_kill_process(
        elapsed_seconds=400, output_changed=False, stall_count=12,
        proc_health=_health(cpu=0.1), output_anomalies=_anomalies(),
        had_children=True, prev_num_children=1,
        progress_signals=None,
    )
    assert should_kill is False, "Frozen output must never hard-kill, even without progress signals"
    assert 'advisory' in reason.lower(), f"Reason should be the downgrade advisory: {reason}"

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License near the top of this file.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SleepSpacingGuard — nudge on sleep-padded rapid-fire probe rounds.

Contract:
- post-only: check_pre never fires.
- classification: foreground `shell` calls containing a sleep invocation are
  sleep probes; background=true never counts (resets); a single command whose
  sleeps sum >= 60s is patient waiting (no streak advance); any non-sleep
  shell command or other tool resets.
- streak advances once per LLM iteration (len(messages) marker), only for
  executed (non-blocked) rounds; blocked calls neither advance nor reset.
- streak >= 3 → inject; >= 4 → escalation ("STOP this loop" + one-longer-sleep
  / background advice).
"""

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.sleep_spacing import (
    SleepSpacingGuard,
    _probe_profile,
)


def _ctx(name, args=None, result=None, messages=None):
    return SleepSpacingGuard and GuardContext(
        tool_name=name, tool_args=args or {}, tool_result=result,
        messages=messages or [],
    )


def _iter(n):
    return [f"msg{i}" for i in range(n)]


# ── classification helper ────────────────────────────────────────────────────

def test_probe_profile_detects_sleep():
    assert _probe_profile("sleep 28; cat reward.txt") == (True, 28.0)
    # summed multi-arg sleep
    assert _probe_profile("sleep 5 5; ls") == (True, 10.0)
    # suffixed forms
    assert _probe_profile("sleep 0.5s && tail x") == (True, 0.5)
    assert _probe_profile("sleep 2m; date") == (True, 120.0)
    # no sleep at all
    assert _probe_profile("cat a; grep b | wc -l") == (False, 0.0)
    # sleep not in command position of its segment (pipe data position)
    assert _probe_profile("echo sleep 999 | cat") == (False, 0.0)
    # quoted sleep is data
    assert _probe_profile('echo "sleep 999"') == (False, 0.0)
    # background irrelevant to the profile fn (caller checks the flag)
    assert _probe_profile("sleep 28; cat x") == (True, 28.0)


def test_command_position_sleep_detected_across_delims():
    seen, total = _probe_profile("cd /jobs && sleep 28; cat x")
    assert seen and total == 28.0
    seen, total = _probe_profile("cat x || sleep 10")
    assert seen and total == 10.0


# ── guard behavior ───────────────────────────────────────────────────────────

def _probe(name="job", msgs=0):
    return ("shell", {"command": f"sleep 28; cat t{name}/reward.txt"},
            "out", _iter(msgs))


def test_first_rounds_silent_then_inject_at_3():
    g = SleepSpacingGuard()
    for i in (1, 2):
        v = g.check_post(_ctx(*_probe("a", i)))
        assert v is None, f"streak {i} must stay silent"
    v = g.check_post(_ctx(*_probe("a", 3)))
    assert v is not None and v.action == "inject"
    assert "sleep-padded probes" in v.message


def test_blocked_call_does_not_advance():
    g = SleepSpacingGuard()
    g.check_post(_ctx(*_probe("a", 1)))
    v = g.check_post(_ctx("shell", {"command": "sleep 28; x"},
                          "[BLOCKED BY GUARD] ...", _iter(2)))
    assert v is None
    # streak must still be 1, not 2 — blocked calls don't count
    g.check_post(_ctx(*_probe("a", 3)))
    v = g.check_post(_ctx(*_probe("a", 4)))
    assert v is not None and "3 sleep-padded" in v.message


def test_non_sleep_shell_resets():
    g = SleepSpacingGuard()
    g.check_post(_ctx(*_probe("a", 1)))
    g.check_post(_ctx(*_probe("a", 2)))
    v = g.check_post(_ctx("shell", {"command": "cat x"}, "ok", _iter(3)))
    assert v is None
    v = g.check_post(_ctx(*_probe("a", 4)))
    assert v is None  # streak restarted


def test_other_tool_resets():
    g = SleepSpacingGuard()
    g.check_post(_ctx(*_probe("a", 1)))
    g.check_post(_ctx(*_probe("a", 2)))
    v = g.check_post(_ctx("read_file", {}, "x", _iter(3)))
    assert v is None
    v = g.check_post(_ctx(*_probe("a", 4)))
    assert v is None


def test_background_shell_resets():
    g = SleepSpacingGuard()
    g.check_post(_ctx(*_probe("a", 1)))
    g.check_post(_ctx(*_probe("a", 2)))
    v = g.check_post(_ctx("shell", {"command": "sleep 90; while :; do ...; done",
                                    "background": True}, "job1", _iter(3)))
    assert v is None
    v = g.check_post(_ctx(*_probe("a", 4)))
    assert v is None


def test_patient_sleep_does_not_advance():
    g = SleepSpacingGuard()
    v = g.check_post(_ctx("shell", {"command": "sleep 60; cat t/reward.txt"},
                          "out", _iter(1)))
    assert v is None
    v = g.check_post(_ctx("shell", {"command": "sleep 60; cat t/reward.txt"},
                          "out", _iter(2)))
    assert v is None
    v = g.check_post(_ctx("shell", {"command": "sleep 90 5; cat t/reward.txt"},
                          "out", _iter(3)))
    assert v is None  # 95s >= 60 → patient


def test_multi_command_mixed_sleep_counts_as_probe():
    # sleeps below the patient bar in total → probe
    g = SleepSpacingGuard()
    args = {"command": "sleep 25; cat a; sleep 25; cat b"}  # 50s total
    v = g.check_post(_ctx("shell", args, "out", _iter(1)))
    assert v is None
    v = g.check_post(_ctx("shell", args, "out", _iter(2)))
    assert v is None
    v = g.check_post(_ctx("shell", args, "out", _iter(3)))
    assert v is not None  # 3rd rapid-fire round fires


def test_escalation_at_4():
    g = SleepSpacingGuard()
    msgs = 0
    for i in range(1, 5):
        msgs = i
        v = g.check_post(_ctx(*_probe("a", msgs)))
    assert v is not None and v.action == "inject"
    assert "Sleeping ~28s to slide under the 30s guard" in v.message \
        or "slide under" in v.message


def test_reset_turn_clears():
    g = SleepSpacingGuard()
    g.check_post(_ctx(*_probe("a", 1)))
    g.check_post(_ctx(*_probe("a", 2)))
    g.check_post(_ctx(*_probe("a", 3)))  # now at threshold
    g.reset_turn()
    v = g.check_post(_ctx(*_probe("a", 4)))
    assert v is None  # fresh turn: first round silent again


def test_parallel_calls_same_iteration_count_once():
    g = SleepSpacingGuard()
    msgs = _iter(7)
    v1 = g.check_post(_ctx("shell", {"command": "sleep 28; cat a"}, "o", msgs))
    v2 = g.check_post(_ctx("shell", {"command": "sleep 28; cat b"}, "o", msgs))
    assert v1 is None and v2 is None  # one iteration = one round
    msgs2 = _iter(8)
    g.check_post(_ctx("shell", {"command": "sleep 28; cat a"}, "o", msgs2))
    v = g.check_post(_ctx("shell", {"command": "sleep 28; cat b"}, "o", msgs2))
    assert v is None  # streak==2 < INJECT_THRESHOLD(3) → still silent
    # third distinct iteration → streak 3 → fires.
    msgs3 = _iter(9)
    v3 = g.check_post(_ctx("shell", {"command": "sleep 28; cat a"}, "o", msgs3))
    assert v3 is not None


def test_check_pre_never_fires():
    g = SleepSpacingGuard()
    assert g.check_pre(_ctx(*_probe("a", 5))) is None

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

"""Tests for KillSafetyGuard — deterministic self-kill hazard detection."""

import flagscale_agent.react.guard.kill_safety as ks
from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.kill_safety import KillSafetyGuard


def _shell(cmd):
    return GuardContext(tool_name="shell", tool_args={"command": cmd})


def _pin_agent_pids(monkeypatch, pids=(4242, 4243)):
    """Pin the agent pid set so tests are deterministic."""
    monkeypatch.setattr(ks, "_agent_cmdlines", lambda: (set(pids), []))


class TestAlreadyCoveredNoCryWolf:
    """Forms the shell rewrite already neutralises must NOT warn."""

    def test_pkill_returns_none(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        assert g.check_pre(_shell('pkill -f "some-job --flag=val"')) is None

    def test_killall_returns_none(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        assert g.check_pre(_shell("killall myjob")) is None

    def test_legacy_kill_flagscale_guard_injects(self, monkeypatch):
        # `kill ... flagscale` is a BARE kill — the guard's domain. The shell
        # rewrite is the silent safety net (it drops the agent's pids); the guard
        # still illuminates the bare-kill hazard with an advisory inject so the
        # decision stays with the LLM.
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        v = g.check_pre(_shell("kill -9 flagscale"))
        assert v is not None and v.action == "inject"

    def test_pgrep_flagscale_agent_blocks(self, monkeypatch):
        # Boundary fact: `\bflagscale\b` does NOT match `flagscale_agent` (_ is a
        # word char, so there is no boundary) — the shell rewrite's token-based
        # regex misses this form. The guard still blocks it via the process-list
        # rule (Rule 2): the target set is selected by pgrep at runtime, so the
        # selection can include the agent. Defense in depth: even where the
        # rewrite is blind, the guard illuminates.
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        v = g.check_pre(_shell("kill -9 $(pgrep -f flagscale_agent)"))
        assert v is not None and v.action == "block"


class TestLiteralSelfPidBlocks:
    """A bare kill naming the agent's own pid is a provable self-kill."""

    def test_kill_agent_pid_blocks(self, monkeypatch):
        _pin_agent_pids(monkeypatch, pids=(4242, 4243))
        g = KillSafetyGuard()
        v = g.check_pre(_shell("kill -9 4242"))
        assert v is not None
        assert v.action == "block"
        assert v.overridable is True
        assert "4242" in v.reason

    def test_kill_other_pid_does_not_block_on_rule1(self, monkeypatch):
        _pin_agent_pids(monkeypatch, pids=(4242, 4243))
        g = KillSafetyGuard()
        # 99999 is not an agent pid -> falls through to advisory inject.
        v = g.check_pre(_shell("kill -9 99999"))
        assert v is not None and v.action == "inject"


class TestProcessListKillBlocks:
    """qemu-startup shape: a kill fed by a process-list pipeline."""

    def test_kill_cmdsub_pgrep_blocks(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        v = g.check_pre(_shell("kill -9 $(pgrep -f polyglot)"))
        assert v is not None and v.action == "block"
        assert v.reason == "kill_from_process_list_may_select_agent"

    def test_kill_pipe_from_ps_blocks(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        v = g.check_pre(_shell("ps aux | grep polyglot | awk '{print $2}' | kill"))
        assert v is not None and v.action == "block"

    def test_kill_pipe_from_grep_blocks(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        v = g.check_pre(_shell("pgrep -f polyglot | xargs kill -9"))
        assert v is not None and v.action == "block"


class TestAdvisoryOnly:
    """A bare kill with an unverifiable target warns but does not block."""

    def test_kill_variable_injects(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        v = g.check_pre(_shell("kill -9 $TARGET_PID"))
        assert v is not None and v.action == "inject"

    def test_kill_backtick_injects(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        v = g.check_pre(_shell("kill -9 `cat /tmp/pid`"))
        assert v is not None and v.action == "inject"


class TestPassthrough:
    """Non-kill commands and non-shell tools pass untouched."""

    def test_no_kill_passes(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        assert g.check_pre(_shell("ls -la")) is None
        assert g.check_pre(_shell("grep -rn foo ./src")) is None

    def test_kill_in_word_passes(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        # 'kill' inside another word must not trip the guard.
        assert g.check_pre(_shell("echo skillets")) is None

    def test_non_shell_tool_passes(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        ctx = GuardContext(tool_name="read_file", tool_args={"path": "kill.txt"})
        assert g.check_pre(ctx) is None

    def test_empty_command_passes(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        assert g.check_pre(_shell("")) is None


class TestWordKillNotCommandPosition:
    """The WORD 'kill' inside an argument must never trip the guard.

    Regression: an earlier gate used a bare ``\\\\bkill\\\\b`` search, so a
    read-only ``grep -n kill README.md`` was BLOCKED and ``cat kill.sh`` warned.
    Command-position gating (clause start / after ;&| / xargs target) fixes it.
    """

    def test_grep_for_the_word_kill_passes(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        # Read-only grep whose PATTERN is the literal word "kill" — the severe FP.
        assert g.check_pre(_shell("grep -n kill README.md")) is None

    def test_cat_file_named_kill_passes(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        assert g.check_pre(_shell("cat kill.sh")) is None

    def test_echo_quoted_kill_passes(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        assert g.check_pre(_shell('echo "kill the process"')) is None

    def test_editor_on_kill_named_file_passes(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        assert g.check_pre(_shell("vim killnotes.txt")) is None


class TestRegionScoping:
    """Checks are scoped to the kill's OWN argument region, not the whole line."""

    def test_pid_in_prior_clause_does_not_block(self, monkeypatch):
        # A pid equal to an agent pid appearing in an EARLIER clause (outside the
        # kill's argument region) must not be read as the kill's target.
        _pin_agent_pids(monkeypatch, pids=(4242, 4243))
        g = KillSafetyGuard()
        v = g.check_pre(_shell("echo 4242 && kill -9 $TARGET_PID"))
        assert v is not None and v.action == "inject"

    def test_process_feed_in_prior_clause_does_not_block(self, monkeypatch):
        # A grep in a PRIOR clause (not feeding the kill) must not trigger the
        # process-list block — only a feed INSIDE the kill's region does.
        _pin_agent_pids(monkeypatch, pids=(4242, 4243))
        g = KillSafetyGuard()
        v = g.check_pre(_shell("grep foo bar.txt && kill -9 $TARGET_PID"))
        assert v is not None and v.action == "inject"


class TestOverride:
    def test_override_reason_accepted(self, monkeypatch):
        _pin_agent_pids(monkeypatch)
        g = KillSafetyGuard()
        assert g.accept_override("the pid is a stale leftover job", _shell("kill -9 1")) is True
        assert g.accept_override("", _shell("kill -9 1")) is False

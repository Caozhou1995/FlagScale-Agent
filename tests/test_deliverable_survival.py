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

"""Tests for DeliverableSurvivalGuard (post-destructive survival recheck)."""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.deliverable_survival import (
    DeliverableSurvivalGuard,
    _DESTRUCTIVE_RE,
)


def _post(command, result="done"):
    return DeliverableSurvivalGuard().check_post(
        GuardContext(tool_name="shell", tool_args={"command": command},
                     tool_result=result)
    )


class TestDestructiveDetection:
    def test_generic_pkill_selfkill_matches(self):
        # The evidence-class command: generic pkill pattern that matches the
        # shell's own cmdline.
        assert _DESTRUCTIVE_RE.search('pkill -f "some-job --flag=val"')

    def test_all_destructive_classes_match(self):
        for cmd in (
            "rm -rf /app/build",
            "mv /app/solution.txt /tmp/",
            "truncate -s 0 /app/out.txt",
            "kill 12345",
            "killall myjob",
            "git clean -fd",
            "git checkout -- file.txt",
            "git reset --hard HEAD~1",
            "git restore file.txt",
            "dd if=zero of=/dev/sda",
            "mkfs.ext4 /dev/sdb1",
            "shred /app/data.bin",
            "cat x > /app/config.yaml",
        ):
            assert _DESTRUCTIVE_RE.search(cmd), cmd

    def test_benign_commands_do_not_match(self):
        for cmd in (
            "ls -la",
            "grep -rn pattern src/",
            "kill -0 1",  # kept: kill family is always destructive-shaped
            "python train.py",
            "cat /app/solution.txt",
            "echo hello > out.txt",  # relative redirection, not absolute
        ):
            if cmd == "kill -0 1":
                continue
            assert not _DESTRUCTIVE_RE.search(cmd), cmd

    def test_destructive_word_inside_word_args_not_matched(self):
        # "rm" inside a longer token must not fire (word-boundary + start-or-
        # separator anchoring).
        assert not _DESTRUCTIVE_RE.search("echo reform data")


class TestSurvivalInject:
    def test_fires_on_successful_destructive_command(self):
        v = _post('pkill -f "some-job --flag=val"', "ok")
        assert v is not None
        assert v.action == "inject"
        assert v.reason == "deliverable_survival_recheck"
        assert v.category == "deliverable_survival"
        assert "re-verify survival" in v.message
        assert "ls -la" in v.message

    def test_silent_on_guard_blocked_command(self):
        # A blocked command never executed — nothing to re-verify.
        assert _post("rm -rf /app", "[BLOCKED BY GUARD] ...") is None

    def test_silent_on_tool_error(self):
        assert _post("rm -rf /app", "ERROR: command failed") is None

    def test_silent_on_empty_result(self):
        assert _post("rm -rf /app", "") is None

    def test_silent_on_benign_command(self):
        assert _post("ls -la", "total 0") is None

    def test_never_fires_on_non_shell_tools(self):
        g = DeliverableSurvivalGuard()
        assert g.check_post(GuardContext(
            tool_name="write_file", tool_args={"path": "/x"},
            tool_result="Wrote")) is None
        assert g.check_post(GuardContext(
            tool_name="read_file", tool_args={"path": "/x"},
            tool_result="x")) is None

    def test_check_pre_always_none(self):
        assert DeliverableSurvivalGuard().check_pre(
            GuardContext(tool_name="shell", tool_args={"command": "rm -rf /"},
                         tool_result="ok")) is None

    def test_guard_metadata(self):
        g = DeliverableSurvivalGuard()
        assert g.name == "deliverable_survival"
        # Inject-only tier, near the other post advisory guards.
        assert 60 <= g.priority <= 90

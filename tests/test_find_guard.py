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

"""Tests for FindGuard — block `find` in favor of internal retrieval."""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.find_guard import FindGuard


def _shell(cmd):
    return GuardContext(tool_name="shell", tool_args={"command": cmd})


class TestFindBlocked:
    """Every shell command invoking `find` is blocked (overridable)."""

    def test_bare_find_blocks(self):
        g = FindGuard()
        v = g.check_pre(_shell("find / -name foo.md"))
        assert v is not None
        assert v.action == "block"
        assert v.reason == "find_invocation"
        assert v.overridable is True

    def test_scoped_find_still_blocks(self):
        # User chose "block all" — even a tightly-scoped find is blocked;
        # override releases it.
        g = FindGuard()
        v = g.check_pre(_shell("find ./src -maxdepth 2 -name '*.yaml'"))
        assert v is not None
        assert v.action == "block"

    def test_find_after_cd_blocks(self):
        g = FindGuard()
        v = g.check_pre(_shell("cd /nfs && find . -name x"))
        assert v is not None
        assert v.action == "block"

    def test_find_after_pipe_blocks(self):
        g = FindGuard()
        v = g.check_pre(_shell("ls | find . -name x"))
        assert v is not None
        assert v.action == "block"

    def test_message_mentions_retrieval_order(self):
        g = FindGuard()
        v = g.check_pre(_shell("find / -name foo"))
        assert "memory" in v.message
        assert "conversation" in v.message.lower()
        assert "grep" in v.message


class TestFindAllowed:
    """Non-find commands and false-positive substrings pass through."""

    def test_non_find_shell_passes(self):
        g = FindGuard()
        assert g.check_pre(_shell("ls -la")) is None
        assert g.check_pre(_shell("grep -rn foo ./src")) is None

    def test_findutils_substring_not_blocked(self):
        g = FindGuard()
        assert g.check_pre(_shell("apt-get install findutils")) is None

    def test_myfind_word_not_blocked(self):
        g = FindGuard()
        assert g.check_pre(_shell("myfind . -name x")) is None
        assert g.check_pre(_shell("./myfind_tool /a")) is None

    def test_non_shell_tool_passes(self):
        g = FindGuard()
        ctx = GuardContext(tool_name="read_file", tool_args={"path": "find.txt"})
        assert g.check_pre(ctx) is None

    def test_empty_command_passes(self):
        g = FindGuard()
        assert g.check_pre(_shell("")) is None


class TestEveryIterBlocks:
    """Block fires on EVERY find call, not once-per-turn."""

    def test_repeated_find_all_block(self):
        g = FindGuard()
        # No reset_turn between calls — each independent find is still blocked.
        assert g.check_pre(_shell("find / -name a")).action == "block"
        assert g.check_pre(_shell("find / -name b")).action == "block"
        assert g.check_pre(_shell("find / -name c")).action == "block"

    def test_override_reason_accepted(self):
        # Default accept_override: any reason > 5 chars releases it.
        g = FindGuard()
        assert g.accept_override("root is bounded ./src", _shell("find ./src")) is True
        assert g.accept_override("", _shell("find ./src")) is False

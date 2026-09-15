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


class TestMultilineFindBlocked:
    """A `find` on its own line of a multi-line command is a real invocation.

    Regression: the old regex used `^` without re.MULTILINE, so a find preceded
    only by a newline escaped the guard entirely.
    """

    def test_find_on_own_line_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("cd /tmp\nfind . -name x")).action == "block"

    def test_find_after_assignment_lines_blocks(self):
        g = FindGuard()
        cmd = "SH=/x\nMC=$SH/megatron\nfind / -path '*/m.py' 2>/dev/null | head"
        assert g.check_pre(_shell(cmd)).action == "block"

    def test_find_alone_at_end_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("echo hi; find")).action == "block"

    def test_bare_find_word_at_line_end_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("ls\nfind")).action == "block"


class TestQuotedAndHeredocAllowed:
    """`find` inside quotes or a heredoc body is data, not an invocation."""

    def test_find_inside_single_quotes_not_blocked(self):
        g = FindGuard()
        assert g.check_pre(_shell("echo 'a | find b'")) is None

    def test_find_inside_double_quotes_not_blocked(self):
        g = FindGuard()
        assert g.check_pre(_shell('echo "x && find y"')) is None

    def test_find_in_python_heredoc_not_blocked(self):
        g = FindGuard()
        cmd = (
            "python3 - <<'PY'\n"
            "import re\n"
            "print(re.search('find', 'x'))\n"
            "PY\n"
        )
        assert g.check_pre(_shell(cmd)) is None

    def test_find_in_heredoc_with_quoted_body_not_blocked(self):
        g = FindGuard()
        cmd = "cat <<EOF\nfoo | find bar\nEOF\n"
        assert g.check_pre(_shell(cmd)) is None

    def test_real_find_still_blocked_with_quoted_args(self):
        # Quoted arguments are common in real finds; they must still block.
        g = FindGuard()
        assert g.check_pre(_shell("find . -name '*.yaml'")).action == "block"
        assert g.check_pre(_shell('find "$DIR" -name "*.py"')).action == "block"

    def test_find_after_separator_outside_quotes_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("ls 'a;b' && find . -name x")).action == "block"

    def test_unterminated_quote_does_not_crash(self):
        g = FindGuard()
        # No assertion on outcome beyond "returns without raising".
        g.check_pre(_shell("echo 'unterminated"))
        g.check_pre(_shell('echo "unterminated'))

    def test_bitshift_not_treated_as_heredoc(self):
        g = FindGuard()
        # `2 << 3` is not a heredoc; a following find on its own line still blocks.
        assert g.check_pre(_shell("echo $(( 2 << 3 ))\nfind . -name x")).action == "block"


class TestBroadGrepBlocked:
    """A RECURSIVE grep over a broad root is blocked (overridable)."""

    def test_grep_rn_on_root_blocks(self):
        g = FindGuard()
        v = g.check_pre(_shell("grep -rn foo /"))
        assert v is not None
        assert v.action == "block"
        assert v.reason == "broad_recursive_grep"

    def test_grep_rn_on_shared_mount_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep -rn 'pattern' /public-nvme")).action == "block"
        assert g.check_pre(_shell("grep -Rl x /mnt")).action == "block"

    def test_grep_rn_on_etc_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep -rn hostname /etc")).action == "block"

    def test_combined_short_flags_recursive_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep -RIn foo /usr")).action == "block"

    def test_grep_long_recursive_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep --recursive foo /opt")).action == "block"

    def test_grep_mid_command_blocks(self):
        g = FindGuard()
        assert g.check_pre(_shell("cd /tmp && grep -rn x /home")).action == "block"


class TestScopedGrepAllowed:
    """Scoped / non-recursive greps pass — they are what the guard recommends."""

    def test_grep_rn_on_subdir_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep -rn foo ./src")) is None
        assert g.check_pre(_shell("grep -rn foo /public-nvme/proj/src")) is None

    def test_non_recursive_grep_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("grep foo /etc/hosts")) is None

    def test_grep_without_path_allowed(self):
        g = FindGuard()
        assert g.check_pre(_shell("cat file | grep -n foo")) is None

    def test_grep_recursive_flag_letter_not_recursive(self):
        g = FindGuard()
        # `-i`/`-n` are not recursive even when combined with other letters.
        assert g.check_pre(_shell("grep -in foo ./src")) is None

    def test_grep_inside_heredoc_not_blocked(self):
        g = FindGuard()
        assert g.check_pre(_shell("python3 - <<'PY'\ngrep -rn x /\nPY\n")) is None

    def test_grep_word_as_arg_not_command(self):
        g = FindGuard()
        # 'grep' as a non-command token should not trip the broad-grep path.
        assert g.check_pre(_shell("echo grep -rn foo /")) is None

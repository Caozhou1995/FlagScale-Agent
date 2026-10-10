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

"""Unit tests for display.SentinelStripper — the streaming filter that hides
completion sentinels ([TASK_COMPLETE]/[NEED_USER_INPUT]) from the live terminal
stream so a gate-blocked completion never appears on screen with authority.
"""
from flagscale_agent.react.display import (
    SentinelStripper,
    COMPLETION_SENTINELS,
    trailing_sentinel,
    has_override_completion,
)


def _run(stripper, deltas):
    """Feed deltas, then flush; return the full displayed string."""
    out = "".join(stripper.feed(d) for d in deltas)
    out += stripper.flush()
    return out


class TestSentinelStripper:
    def test_plain_text_passes_through(self):
        s = SentinelStripper()
        assert _run(s, ["hello ", "world"]) == "hello world"

    def test_single_delta_sentinel_removed(self):
        s = SentinelStripper()
        assert _run(s, ["done.\n[TASK_COMPLETE]"]) == "done.\n"

    def test_sentinel_split_across_deltas(self):
        # The classic streaming case: sentinel arrives in fragments.
        s = SentinelStripper()
        assert _run(s, ["all set ", "[TASK", "_COMP", "LETE]"]) == "all set "

    def test_sentinel_split_char_by_char(self):
        s = SentinelStripper()
        deltas = list("result ready [TASK_COMPLETE]")
        assert _run(s, deltas) == "result ready "

    def test_need_user_input_sentinel(self):
        s = SentinelStripper()
        assert _run(s, ["question? ", "[NEED_USER_INPUT]"]) == "question? "

    def test_text_after_sentinel_still_shown(self):
        # Defensive: text following a sentinel is not swallowed.
        s = SentinelStripper()
        assert _run(s, ["a[TASK_COMPLETE]b"]) == "ab"

    def test_partial_that_never_completes_is_flushed(self):
        # A truncated stream ending in a real '[' fragment is genuine text.
        s = SentinelStripper()
        # "[TASK" held back mid-stream, but stream ends -> flush shows it.
        assert _run(s, ["progress ", "[TASK"]) == "progress [TASK"

    def test_bracket_text_not_a_sentinel(self):
        s = SentinelStripper()
        assert _run(s, ["see [NOTE] here"]) == "see [NOTE] here"

    def test_no_premature_emit_of_held_prefix(self):
        # After feeding a partial prefix, it must be held (not yet emitted).
        s = SentinelStripper()
        emitted = s.feed("ok [TASK")
        assert emitted == "ok "        # "[TASK" held back
        emitted2 = s.feed("_COMPLETE]")
        assert emitted2 == ""          # completes into sentinel -> stripped
        assert s.flush() == ""

    def test_held_prefix_resolves_to_plain_text(self):
        # Held "[TASK" then a non-matching continuation -> both shown.
        s = SentinelStripper()
        assert _run(s, ["x [TASK", " force"]) == "x [TASK force"

    def test_multiple_sentinels_in_stream(self):
        s = SentinelStripper()
        assert _run(s, ["[TASK_COMPLETE]middle[TASK_COMPLETE]"]) == "middle"

    def test_sentinels_constant_present(self):
        assert "[TASK_COMPLETE]" in COMPLETION_SENTINELS
        assert "[NEED_USER_INPUT]" in COMPLETION_SENTINELS


class TestDecoratedSentinelStripping:
    """Models often wrap the sentinel in markdown (``**[NEED_USER_INPUT]**``).
    The stripper must remove the decorations too — a bare-marker-only strip
    leaves ``****`` on screen (the reported bug) and a strict endswith() gate
    rejects the signal, forcing endless "no stop signal" retries.
    """

    def test_bold_sentinel_single_delta(self):
        s = SentinelStripper()
        assert _run(s, ["done. **[NEED_USER_INPUT]**"]) == "done. "

    def test_bold_task_complete_single_delta(self):
        s = SentinelStripper()
        assert _run(s, ["done. **[TASK_COMPLETE]**"]) == "done. "

    def test_bold_split_into_deltas(self):
        s = SentinelStripper()
        # Opening "**" and closing "**" arrive as their own deltas.
        assert _run(s, ["done. **", "[NEED", "_USER_INPUT]", "**"]) == "done. "

    def test_bold_char_by_char(self):
        s = SentinelStripper()
        assert _run(s, list("result **[TASK_COMPLETE]**")) == "result "

    def test_no_asterisk_leak_on_held_prefix(self):
        # The exact reported failure: after stripping the marker the orphaned
        # opening decoration must never be emitted.
        s = SentinelStripper()
        parts = [s.feed(d) for d in ["ok **", "[NEED_USER_INPUT]", "**"]]
        parts.append(s.flush())
        assert "".join(parts) == "ok "
        assert "*" not in "".join(parts)

    def test_underscore_bold_sentinel(self):
        s = SentinelStripper()
        assert _run(s, ["done. __[TASK_COMPLETE]__"]) == "done. "

    def test_italic_sentinel(self):
        s = SentinelStripper()
        assert _run(s, ["x *[TASK_COMPLETE]*"]) == "x "

    def test_decorated_mid_text_after_sentinel_shown(self):
        s = SentinelStripper()
        assert _run(s, ["a**[TASK_COMPLETE]**b"]) == "ab"


class TestTrailingSentinel:
    """Shared helper used by the kernel gate and the verification gate."""

    def test_bare(self):
        assert trailing_sentinel("line\n[TASK_COMPLETE]") == "[TASK_COMPLETE]"

    def test_bold(self):
        assert trailing_sentinel("x **[NEED_USER_INPUT]**") == "[NEED_USER_INPUT]"

    def test_underscore_bold(self):
        assert trailing_sentinel("x __[TASK_COMPLETE]__") == "[TASK_COMPLETE]"

    def test_trailing_whitespace(self):
        assert trailing_sentinel("x **[NEED_USER_INPUT]**  \n") == "[NEED_USER_INPUT]"

    def test_non_sentinel_none(self):
        assert trailing_sentinel("see [NOTE] here") is None

    def test_mention_not_trailing_none(self):
        # A quoted marker with words after it is not a terminator.
        assert trailing_sentinel("x [TASK_COMPLETE]\ntrailing words") is None

    def test_empty_none(self):
        assert trailing_sentinel("") is None


class TestOverrideCompletion:
    """The override completion shape: marker followed by an inline reason."""

    def test_bare_override(self):
        assert has_override_completion(
            "done [TASK_COMPLETE]\n_override_reason: trivial rename") is True

    def test_decorated_override(self):
        assert has_override_completion(
            "done **[TASK_COMPLETE]**\n_override_reason: trivial") is True

    def test_no_reason_false(self):
        assert has_override_completion("done [TASK_COMPLETE]") is False

    def test_empty_false(self):
        assert has_override_completion("") is False

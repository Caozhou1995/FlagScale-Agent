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

"""Tests: a grounding failure must EXPLAIN itself, not just refuse.

Live incident this pins: an agent pasted its own PLAN TITLE as a reading's
`source` six times in a row, each time only rewording it, because the block said
only "A reading is NOT grounded in the task text" — it never revealed (a) that
the corpus is the user's own message, (b) that the check is substring
membership, or (c) what the task text actually looks like. Every retry was the
same wrong shape of edit. These tests force the message to carry the mechanical
facts the agent needs, and force a repeated identical miss to be called out.
"""

import pytest

from flagscale_agent.react.guard import verification as V

TASK = (
    "请对着失败的题继续优化吧。此次有8个小时时间窗口，但是给你提个醒，"
    "北京时间10点之后api的返回速度可能变慢，所以怎么又快又好的解出题完成任务，"
    "也是你所面临一个重大挑战。"
)

PLAN_TITLE = "败局定向优化：18 题逐题归因 + 修复 + 重跑（8h 窗口）"


def _dr(source, normalized_side="neither"):
    return {
        "deliverable": "/tmp/x.md",
        "grader_visible": False,
        "readings": [{
            "reading": "r",
            "source": source,
            "acceptance_test": "true",
            "normalized_side": normalized_side,
        }],
        "hedge": {
            "shape": "intersection",
            "output": "o",
            "extras_direction": "penalized",
            "evidence": {"command": "c", "observed": "o"},
        },
    }


@pytest.fixture(autouse=True)
def _clear_miss_counts():
    V._grounding_miss_counts.clear()
    yield
    V._grounding_miss_counts.clear()


class TestGroundingFailureExplainsItself:
    def test_ungrounded_source_reports_what_was_checked(self):
        """Message must say the check is substring membership, not semantics."""
        v = V._check_deliverable_reading_field(
            _dr(PLAN_TITLE), task_text=TASK, observed=None)
        assert v is not None
        assert getattr(v, "reason", "") == "reading_source_not_grounded"
        assert "what was checked" in v.message
        assert "substring" in v.message.lower()

    def test_ungrounded_source_shows_the_real_task_text(self):
        """The agent must SEE the corpus it is supposed to quote from."""
        v = V._check_deliverable_reading_field(
            _dr(PLAN_TITLE), task_text=TASK, observed=None)
        assert "where the corpus comes from" in v.message
        # A distinctive phrase of the real task text must be echoed back.
        assert "请对着失败的题继续优化吧" in v.message

    def test_ungrounded_source_names_the_wrong_corpus(self):
        """It must say the corpus is NOT the plan/notes — that was the trap."""
        v = V._check_deliverable_reading_field(
            _dr(PLAN_TITLE), task_text=TASK, observed=None)
        assert "plan" in v.message.lower()

    def test_partial_overlap_shows_the_quotable_slice(self):
        """When a slice of the quote DOES occur, show that slice to paste."""
        partial = "怎么又快又好的解出题完成任务（我的计划标题后半部分）"
        v = V._check_deliverable_reading_field(
            _dr(partial), task_text=TASK, observed=None)
        assert v is not None
        assert "how to fix" in v.message


class TestRepeatedMissIsCalledOut:
    def test_second_identical_miss_says_rewording_is_not_the_fix(self):
        first = V._check_deliverable_reading_field(
            _dr(PLAN_TITLE), task_text=TASK, observed=None)
        assert "repeated failure" not in first.message
        second = V._check_deliverable_reading_field(
            _dr(PLAN_TITLE), task_text=TASK, observed=None)
        assert "repeated failure" in second.message
        assert "wording" in second.message.lower()

    def test_miss_counter_is_keyed_on_normalized_source(self):
        V._note_grounding_miss(PLAN_TITLE)
        V._note_grounding_miss("  " + PLAN_TITLE.upper() + " ")
        assert V._grounding_miss_counts[V._normalize_for_grounding(PLAN_TITLE)] == 2


class TestGroundedReadingStillPasses:
    def test_real_verbatim_quote_is_not_blocked(self):
        """No over-blocking: a genuine quote must sail through."""
        v = V._check_deliverable_reading_field(
            _dr("怎么又快又好的解出题完成任务"), task_text=TASK, observed=None)
        assert v is None or getattr(v, "reason", "") != "reading_source_not_grounded"

    def test_quote_is_case_and_whitespace_insensitive(self):
        v = V._check_deliverable_reading_field(
            _dr("怎么又快又好的解出题完成任务"), task_text="  " + TASK + "  ",
            observed=None)
        assert v is None or getattr(v, "reason", "") != "reading_source_not_grounded"


class TestTaskSegments:
    def test_composite_task_text_splits_into_segments(self):
        composite = "first real task line\n\nsecond real user turn"
        segs = V._task_segments(composite)
        assert segs == ["first real task line", "second real user turn"]

    def test_empty_task_text_yields_no_segments(self):
        assert V._task_segments("") == []
        assert V._task_segments("   ") == []

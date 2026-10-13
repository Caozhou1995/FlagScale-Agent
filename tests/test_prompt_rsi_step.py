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

"""Regression tests for the fixed plan tail (review → RSI) and the prompt
changes that accompany it: the Response Format's two-section close and the
language-mirroring rule.

A future edit must not silently drop the RSI step, reorder the tail, or weaken
the language rule back into an over-emphasis on Chinese.
"""

import pytest

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
from flagscale_agent.react.tools.plan_create import (
    PlanCreateTool,
    _REVIEW_STEP_TITLE,
    _RSI_STEP_TITLE,
    _RSI_STEP_ACCEPTANCE,
)


@pytest.fixture
def plan_dir(tmp_path):
    d = tmp_path / "plan"
    d.mkdir()
    return str(d)


def test_review_title_no_longer_claims_final_step():
    # RSI is now the last step; the review title must not lie about being last.
    assert "final step" not in _REVIEW_STEP_TITLE.lower()


def test_rsi_step_is_appended_after_review(plan_dir):
    tp = TaskPlan(plan_dir)
    PlanCreateTool(tp, session_id="t").execute(title="T", steps=["A", "B"])
    steps = tp.get_active()["steps"]
    titles = [s["title"] for s in steps]
    # tail order: ... → review → RSI (RSI LAST)
    assert titles[-2] == _REVIEW_STEP_TITLE
    assert titles[-1] == _RSI_STEP_TITLE


def test_rsi_title_says_recursive_self_improvement():
    # RSI stands for Recursive Self-Improvement, not a generic "reflection".
    assert "Recursive Self-Improvement" in _RSI_STEP_TITLE


def test_rsi_acceptance_is_mechanism_first():
    joined = " ".join(_RSI_STEP_ACCEPTANCE).lower()
    # primary channel = a MECHANISM change to the harness...
    assert "mechanism" in joined
    assert "proposal" in joined
    # ...with task-knowledge memory explicitly demoted to a fallback
    assert "fallback" in joined
    assert "memory_write" in joined
    # an unlanded improvement is called out as breaking the recursion
    assert "recursion" in joined or "never lands" in joined


def test_rsi_acceptance_makes_empty_close_explicit():
    joined = " ".join(_RSI_STEP_ACCEPTANCE).lower()
    assert "nothing" in joined and "explicit" in joined


def test_rsi_acceptance_demands_three_channels():
    # The wrap-up must enumerate all three RSI channels: landed mechanism
    # changes, proposals, and memory updates.
    joined = " ".join(_RSI_STEP_ACCEPTANCE).lower()
    assert "three rsi channels" in joined
    assert "landed mechanism" in joined
    assert "memory updates" in joined


def test_rsi_acceptance_demands_historical_open_proposals():
    # Proposals channel must include BOTH newly-raised AND historical OPEN
    # (unadjudicated) proposals, surfaced by id, never silently dropped.
    joined = " ".join(_RSI_STEP_ACCEPTANCE).lower()
    assert "historical open" in joined
    assert "newly raised" in joined
    assert "proposal(action='list')" in joined


def test_plan_create_note_announces_both_fixed_steps(plan_dir):
    tp = TaskPlan(plan_dir)
    result = PlanCreateTool(tp, session_id="t").execute(title="T", steps=["A"])
    assert "RSI" in result


def test_response_format_has_task_and_rsi_sections():
    p = SYSTEM_PROMPT_STATIC
    assert "## Response Format" in p
    assert "Task result" in p
    assert "RSI result" in p


def test_response_format_rsi_lists_all_three_channels():
    # The system prompt's RSI section must mandate the three sub-headings and
    # the historical-open-proposal duty, not just "lead with a mechanism change".
    p = SYSTEM_PROMPT_STATIC
    assert "Landed mechanism changes" in p
    assert "newly raised this session" in p
    assert "historical OPEN" in p or "historical open" in p
    assert "Memory updates" in p
    assert "proposal(action='list')" in p


def test_language_rule_is_general_and_per_message():
    # The rule must be a GENERAL, mechanical decision procedure: decide by the
    # LATEST message's dominant language, re-decided every turn, with context
    # never deciding it.
    p = SYSTEM_PROMPT_STATIC
    assert "Match user's language" in p
    assert "LATEST message" in p
    assert "re-decided EVERY turn" in p
    assert "Context NEVER decides it" in p


def test_language_rule_covers_mixed_message_and_original_script():
    # A mixed message must have a tie-break, and embedded code/identifiers must
    # stay in their original script rather than setting the reply language.
    p = SYSTEM_PROMPT_STATIC
    assert "majority language wins" in p
    assert "ORIGINAL script" in p


def test_language_rule_drops_the_chinese_special_case_framing():
    # Reframe hygiene: the superseded narrow framing (an English-vs-Chinese
    # special case) must be GONE, not merely shadowed by the new wording.
    p = SYSTEM_PROMPT_STATIC
    assert "Do NOT switch to Chinese" not in p
    assert "user writes English" not in p

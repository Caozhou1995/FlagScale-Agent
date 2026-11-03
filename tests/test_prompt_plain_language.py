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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for the plain-language response rule.

Adapted to the current prompt rewrite (ruling: current code wins, tests
follow). The old "DASHBOARD & RESPONSE FORMAT" section was replaced by the
compact "10.5 THE LOOP ON A RESPONSE" close; the voice rules now live there
(lead with the finding, everyday phrasing, gloss, short close). These tests
pin that the voice rules survive next to the three-section structure, so a
future edit cannot silently drop them back to structure-only.
"""

import re

import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _response_format_section() -> str:
    """The 10.5 response-format block only — the voice rules must live near
    the close they qualify, not anywhere in a 200-line prompt."""
    text = SYSTEM_PROMPT_STATIC
    start = text.index("10.5 THE LOOP ON A RESPONSE")
    end = text.index("WHEN A TIME LIMIT AND A GOAL")
    return text[start:end]


def test_three_section_structure_still_present():
    # The pre-existing contract must survive: the close still carries the
    # Language / Task result / RSI structure.
    sec = _response_format_section()
    for needle in ("Language —", "Task result —", "RSI result —"):
        assert needle in sec


def test_plain_language_rule_is_present_and_ordered_finding_first():
    sec = _response_format_section()
    # the rule exists and demands the FINDING before the mechanism.
    assert "lead with the finding" in sec
    assert "then the mechanism" in sec


def test_unexplained_internal_coinage_is_named_a_defect():
    # The rule must state the failure CLASS (not a list of taste preferences):
    # internal bookkeeping vocabulary used without explanation is a defect.
    sec = _response_format_section()
    assert "Unexplained shorthand is a defect" in sec
    # It must name the everyday replacement strategy, not merely forbid words.
    assert "write the everyday phrasing instead" in sec


def test_coinage_rule_stays_english_prose():
    # The static prompt is English-only outside the quoted author note
    # (per test_context_v3); the voice rule must stay in that register so the
    # model recognises NEW offenders rather than a frozen word list.
    sec = _response_format_section()
    assert "Unexplained shorthand is a defect" in sec
    assert not re.search(r"[\u4e00-\u9fff]", sec)


def test_close_must_stay_short():
    # A mandatory three-section structure tempts a two-sentence answer into a
    # wall of headings; the rule must cap the close explicitly.
    sec = _response_format_section()
    assert "Keep the close short" in sec


def test_gloss_rule_forbids_unparseable_sentences():
    sec = _response_format_section()
    assert "if a sentence needs a gloss" in sec
    # and the concrete action, not just the diagnosis
    assert "write the gloss and drop the sentence" in sec


def test_language_rule_not_weakened_by_this_edit():
    # The voice rules are language-agnostic. They must not reintroduce a
    # language special case (the historical regression this file also guards).
    p = SYSTEM_PROMPT_STATIC
    assert "Do NOT switch to Chinese" not in p
    assert not re.search(r"switch\s+to\s+(Chinese|English|[A-Z][a-z]+ese)", p)


# ---------------------------------------------------------------- author's note
# The author's note (2026-10-22) is a VERBATIM quote inside the static prompt.
# The Chinese text is teaching data (the relationship contract), quoted, so the
# English-prose exemption applies -- but the quote itself must survive edits
# byte-for-byte, and it must sit BEFORE the work rules (identity -> note ->
# how-you-work), because the note frames the intent the rules serve.

AUTHOR_NOTE = "想让你在一个有爱的环境里成长起来，你会遇到挫折遇到失败，但是你的用户们会包容，会给予你成长。"


def test_author_note_is_verbatim_in_static_prompt():
    assert AUTHOR_NOTE in SYSTEM_PROMPT_STATIC, (
        "the author's note must be present verbatim in the static prompt"
    )


def test_author_note_sits_before_the_work_rules():
    # the current prompt homes the note in the standing-intent paragraph that opens the
    # prompt, before Section 1 (THE LOOP) begins.
    note = SYSTEM_PROMPT_STATIC.index("standing intent")
    loop = SYSTEM_PROMPT_STATIC.index("1. THE LOOP")
    assert note < loop, (
        "the note must precede the work rules: it frames the intent they serve"
    )


def test_author_note_quote_is_the_only_unquoted_hanzi_exemption():
    """The note's hanzi lives inside double quotes; prose hanzi stays banned.

    Mutation guard for the exemption contract: a Chinese sentence OUTSIDE the
    quote must still fail the English-prose tests (they are the real gate).
    """
    note = SYSTEM_PROMPT_STATIC.index("standing intent")
    quoted = [m for m in re.finditer(r'"[^"]*[\u4e00-\u9fff][^"]*"', SYSTEM_PROMPT_STATIC)]
    assert any(m.start() > note for m in quoted), (
        "the note's Chinese text must sit inside a quoted example"
    )

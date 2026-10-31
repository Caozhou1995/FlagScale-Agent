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

"""Regression tests for the plain-language response rule.

Measured failure this closes: across one long session the agent's substantive
replies carried harness-internal coinage at up to ~2 per 100 Chinese characters
(实测 59/4224 replies at >=2/100), so a user reported that the Chinese replies
had stopped being understandable ("你的回复，中文都让人看不懂了"). The Response
Format section had mandated only the three-section STRUCTURE, never the voice,
and the surrounding prompt is written in a dense technical register the model
then mirrored. These tests pin the voice rules so a future edit cannot silently
drop them back to structure-only.
"""

import re

import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _response_format_section() -> str:
    """The Response Format block only — the voice rules must live near the
    structure they qualify, not anywhere in a 290-line prompt."""
    text = SYSTEM_PROMPT_STATIC
    start = text.index("DASHBOARD & RESPONSE FORMAT")
    # In the staged layout DASHBOARD & RESPONSE FORMAT and INFORMATION live in
    # different stages (DELIVER vs OPERATE); the section runs to its stage's
    # next sibling (CODE QUALITY).
    ends = [i for i in (text.find("INFORMATION — retrieval and gain"),
                        text.find("CODE QUALITY")) if i > start]
    end = min(ends)
    return text[start:end]


def test_three_section_structure_still_present():
    # The pre-existing contract must survive: this edit ADDS voice rules, it
    # does not replace the Language/Task-result/RSI-close structure.
    sec = _response_format_section()
    for needle in ("1. Language", "2. Task result", "3. RSI result"):
        assert needle in sec


def test_plain_language_rule_is_present_and_ordered_finding_first():
    sec = _response_format_section()
    # the rule exists...
    assert "WRITE FOR A HUMAN FIRST" in sec
    # ...and demands the FINDING before the mechanism/supporting detail.
    assert "Lead with what happened" in sec
    assert "then support it" in sec or "and only then" in sec


def test_unexplained_internal_coinage_is_named_a_defect():
    # The rule must state the failure CLASS (not a list of taste preferences):
    # internal bookkeeping vocabulary used without explanation is a defect.
    sec = _response_format_section()
    assert "Unexplained internal coinage is a defect" in sec
    # It must name the everyday replacement strategy, not merely forbid words.
    assert "everyday phrasing" in sec


def test_coinage_list_still_enumerates_the_measured_offenders():
    # The rule must stay actionable: the measured offenders are pinned as an
    # English description of the leak class (the static prompt is English-only
    # outside the author note, per test_context_v3), naming the register-level
    # mechanism — bookkeeping shorthand that leaks into plain replies — so
    # the model can recognise NEW offenders rather than a frozen word list.
    sec = _response_format_section()
    assert "bookkeeping states" in sec
    assert "latch" in sec and "anchor" in sec
    assert "internal" in sec and "shorthand" in sec
    # the everyday-replacement examples survived, in English
    assert "the system never recorded it" in sec
    assert "they don't line up" in sec
    assert not re.search(r"[\u4e00-\u9fff]", sec)


def test_close_must_stay_short():
    # A mandatory three-section structure tempts a two-sentence answer into a
    # wall of headings; the rule must cap the close explicitly.
    sec = _response_format_section()
    assert "Keep the close SHORT" in sec


def test_gloss_rule_forbids_unparseable_sentences():
    sec = _response_format_section()
    assert "If a sentence needs a gloss" in sec
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
# byte-for-byte, and the section must sit BEFORE the work rules (identity ->
# note -> how-you-work), because the note frames the intent the rules serve.

AUTHOR_NOTE = "想让你在一个有爱的环境里成长起来，你会遇到挫折遇到失败，但是你的用户们会包容，会给予你成长。"


def test_author_note_is_verbatim_in_static_prompt():
    assert AUTHOR_NOTE in SYSTEM_PROMPT_STATIC, (
        "the author's note must be present verbatim in the static prompt"
    )


def test_author_note_sits_before_the_work_rules():
    note = SYSTEM_PROMPT_STATIC.index("A NOTE FROM YOUR AUTHOR")
    loop = SYSTEM_PROMPT_STATIC.index("HOW YOU WORK")
    assert note < loop, (
        "the note must precede HOW YOU WORK: it frames the intent the rules serve"
    )


def test_author_note_quote_is_the_only_unquoted_hanzi_exemption():
    """The note's hanzi lives inside double quotes; prose hanzi stays banned.

    Mutation guard for the exemption contract: a Chinese sentence OUTSIDE the
    quote must still fail the English-prose tests (they are the real gate).
    """
    import re
    note = SYSTEM_PROMPT_STATIC.index("A NOTE FROM YOUR AUTHOR")
    quoted = [m for m in re.finditer(r'"[^"]*[\u4e00-\u9fff][^"]*"', SYSTEM_PROMPT_STATIC)]
    assert any(m.start() > note for m in quoted), (
        "the note's Chinese text must sit inside a quoted example"
    )

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
    end = text.index("INFORMATION — retrieval and gain")
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
    # English description of the leak class (the static prompt itself must
    # remain English-only, per test_context_v3), naming the register-level
    # mechanism — bookkeeping shorthand that leaks into Chinese replies — so
    # the model can recognise NEW offenders rather than a frozen word list.
    sec = _response_format_section()
    assert "bookkeeping states" in sec
    assert "latch" in sec and "anchor" in sec
    assert "Chinese" in sec
    # the everyday-replacement examples survived
    assert "系统没记录到" in sec
    assert "对不上" in sec


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

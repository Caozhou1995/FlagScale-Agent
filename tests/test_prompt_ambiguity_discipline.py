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

"""Regression tests for the UNDER-SPECIFIED CLAUSE + INVISIBLE GRADER discipline
in the static system prompt.

This is prompt-only guidance, so the test asserts the load-bearing clauses are
present in SYSTEM_PROMPT_STATIC (a future edit must not silently drop them).
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_ambiguity_section_present():
    assert "UNDER-SPECIFIED CLAUSE + INVISIBLE GRADER" in SYSTEM_PROMPT_STATIC


def test_ambiguity_three_steps_present():
    p = SYSTEM_PROMPT_STATIC
    # step 1: enumerate readings + acceptance sets
    assert "ACCEPTANCE SET" in p
    # step 2: a claimed hedge must be proven, never asserted
    assert "CHECK FOR A HEDGE AND PROVE IT" in p
    assert "do NOT assert it" in p
    # step 3: empty-intersection bet, with the two deterministic tiebreaks
    assert "INTERSECTION IS EMPTY" in p
    assert "WORDING-LITERAL PRIORITY" in p
    assert "ROBUSTNESS ON THE UNAMBIGUOUS AXES" in p


def test_ambiguity_is_regex_free_guidance():
    """The discipline must be an actionable procedure, not a keyword detector."""
    p = SYSTEM_PROMPT_STATIC
    # no enumeration of trigger keywords as the mechanism
    for kw in ("except/unless/when", "if the clause contains"):
        assert kw not in p


def test_pre_existing_dominance_clause_preserved():
    p = SYSTEM_PROMPT_STATIC
    assert "DOMINANCE TIE-BREAK & DROPPED-CONSTRAINT LEDGER" in p
    assert "Dropped-constraint ledger" in p


def test_no_grader_oracle_leak_in_prompt():
    """HARD CONSTRAINT: authored harness content must never leak a task's grader.

    The prompt once used a real grader's comparison (`orig != str(BeautifulSoup(
    orig,'html.parser'))`) and a verbatim task-text grant as worked examples.
    That handed the agent the checker and invited a false "satisfies both
    readings" rationalization. Illustrations must be GENERIC/synthetic — never a
    real grader's expression, reference implementation, or verbatim task phrase.
    """
    p = SYSTEM_PROMPT_STATIC
    # a leaked parser-oracle expression
    assert "BeautifulSoup" not in p
    assert "html.parser" not in p
    # a verbatim task-text grant quoted as an example
    assert "except for normalization that may occur" not in p

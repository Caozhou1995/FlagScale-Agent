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
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
from flagscale_agent.react.tools.plan_create import _FRAME_STEP_ACCEPTANCE


def test_frame_acceptance_carries_constructive_hedge_check():
    """The Frame step's acceptance must carry the same constructive check.

    The Frame acceptance is the ENFORCED surface: it is prepended to every
    plan and gated by the step_done evidence check. A new plan must therefore
    be required to write each reading's acceptance predicate, build a
    candidate, and test the intersection — betting only after it is
    constructively shown empty.
    """
    low = " ".join(_FRAME_STEP_ACCEPTANCE).lower()
    assert "constructive hedge-feasibility" in low
    assert "acceptance predicate" in low
    assert "intersection" in low
    assert "constructively shown empty" in low
    # must not hardcode any originating task's specifics
    assert not re.search(r"0x[0-9a-fA-F]{4,}", low), "Frame acceptance leaked a hex address"
    assert "beautifulsoup" not in low


def test_hedge_doctrine_is_synced_across_both_surfaces():
    """The constructive-hedge doctrine must hold on BOTH surfaces at once.

    The same rule lives in the enforced Frame acceptance AND the prose
    under-specified-clause section of the static prompt. Nothing ties them:
    an edit to one can silently drift from the other (the prompt-vs-live-
    source drift family). Assert the load-bearing CONCEPTS appear on both,
    so a future edit to a single surface fails loudly instead of leaving the
    two surfaces contradicting each other.
    """
    frame = " ".join(_FRAME_STEP_ACCEPTANCE).lower()
    prompt = SYSTEM_PROMPT_STATIC.lower()
    for concept in ("acceptance predicate", "shown empty"):
        assert concept in frame, f"Frame acceptance missing {concept!r}"
        assert concept in prompt, f"prompt missing {concept!r}"


def test_ambiguity_is_regex_free_guidance():
    """The discipline must be an actionable procedure, not a keyword detector."""
    p = SYSTEM_PROMPT_STATIC
    # no enumeration of trigger keywords as the mechanism
    for kw in ("except/unless/when", "if the clause contains"):
        assert kw not in p


def test_frame_acceptance_names_both_hedge_shapes():
    low = " ".join(_FRAME_STEP_ACCEPTANCE).lower()
    assert "union / superset" in low
    assert "ignores output" in low
    assert "both shapes" in low


def test_hedge_doctrine_no_longer_excludes_the_union_shape():
    """NEGATIVE assertion: the intersection-EXCLUSIVE phrasing must be GONE.

    The pre-fix rule said a hedge is real "iff a single output passes them all"
    and licensed emitting "from the intersection" only — wording that rules out
    the unpenalised-extras superset. Substring-presence tests stayed green while
    the prose excluded the shape that actually won. A reframe needs a negative
    assertion that the superseded exclusive phrasing is absent.
    """
    p = SYSTEM_PROMPT_STATIC
    assert "may you emit from the intersection" not in p
    low = " ".join(_FRAME_STEP_ACCEPTANCE).lower()
    assert "only an intersection constructively shown empty" not in low


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


def test_minimal_uncertainty_leaks_no_grader_or_posture_hardcode():
    """HARD: the new content must not leak a task's grader, and must not
    hardcode the originating task's specifics (it is a general posture)."""
    p = SYSTEM_PROMPT_STATIC
    assert "BeautifulSoup" not in p
    assert "html.parser" not in p
    low = " ".join(_FRAME_STEP_ACCEPTANCE).lower()
    for bad in ("beautifulsoup", "html.parser", "retry"):
        assert bad not in low, f"Frame acceptance leaked {bad!r}"
    # generic guards: the general posture must carry no re-serialization
    # artifact (HTML entity literal) and no concrete address constant
    assert not re.search(r"&[a-zA-Z]+;", low), "Frame acceptance leaked an HTML entity literal"
    assert not re.search(r"0x[0-9a-fA-F]{4,}", low), "Frame acceptance leaked a hex address"


def test_cross_engine_and_corpus_rules_leak_no_specifics():
    """The two new rules are GENERAL posture: no task's grader, engine, or
    construct may be hardcoded."""
    p = SYSTEM_PROMPT_STATIC
    for bad in ("BeautifulSoup", "html.parser", "selenium", "chrome"):
        assert bad not in p, f"prompt leaked {bad!r}"
    assert not re.search(r"0x[0-9a-fA-F]{4,}", p), "prompt leaked a hex address"



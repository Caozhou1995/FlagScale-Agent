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


def test_hedge_feasibility_is_constructive_not_introspective():
    """A hedge must be proven by CONSTRUCTION, not decided by introspection.

    Evidence 2026-10-12: a solver recognized the two grader readings,
    introspected "Intersection = EMPTY ... No hedge possible. Must
    BET." and shipped one reading (reward=0), while a sibling solver that
    CONSTRUCTED the superset covering both readings passed. The prompt must therefore
    demand writing each reading's acceptance predicate and building a
    candidate output to test the intersection — an introspected "no hedge"
    is a hypothesis, not a finding.
    """
    p = SYSTEM_PROMPT_STATIC
    assert "by CONSTRUCTION, not introspection" in p
    assert "acceptance PREDICATE" in p
    assert "build one candidate output" in p
    # an introspected emptiness is a hypothesis; bet only after proof
    assert 'introspected "no hedge possible" is likewise a HYPOTHESIS' in p
    assert "constructively shown empty" in p


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


def test_hedge_doctrine_has_no_contradictory_fallthrough():
    """NEGATIVE assertion: the OLD contradictory bet rule must be GONE.

    Evidence 2026-10-12 (independent review of commit 208fc03): the first
    landing was ADDITIVE — it added the constructive rule but left standing
    the introspection-era clause "if it does not [confirm], you have no hedge
    — fall through to step 3 and bet", which licenses betting on a FAILED
    probe (one candidate failing a probe does not show the intersection is
    empty). Substring-presence tests stayed green while the prose contradicted
    itself. A reframe needs a negative assertion that the old framing's
    distinctive phrase is ABSENT, not just that the new one is present.
    """
    p = SYSTEM_PROMPT_STATIC
    assert "fall through to step 3 and bet" not in p
    # the probe must be ordered AFTER construction, not "First"
    assert "First run a probe whose DESIGN INTENT" not in p
    # and a failed probe must be framed as unproven, not as disproven
    assert "the hedge is UNPROVEN" in p


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


def test_hedge_names_both_shapes_intersection_and_union():
    """The hedge doctrine must name BOTH hedge shapes, not only the intersection.

    Evidence 2026-10-12: a solver under an extras-ignoring reference matcher
    (a reference key absent from the output is skipped, not penalised) made a
    UNION/SUPERSET the winning hedge — but the intersection-only framing
    called the union "impossible" and the agent bet on one base and scored
    0.00%. The rule must therefore
    present the union/superset shape alongside the intersection, and require
    emptiness to be shown under BOTH before betting.
    """
    p = SYSTEM_PROMPT_STATIC
    assert "UNION / SUPERSET" in p
    assert "IGNORES output it did not ask for" in p
    assert "under BOTH shapes" in p


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


def test_minimal_uncertainty_posture_present_both_surfaces():
    """When the grader is invisible AND the goal under-determined, the agent
    must minimise INTRODUCED uncertainty — a posture, not a fixed rule.

    Evidence 2026-10-12 (two consecutive failed runs on an HTML-preservation
    task): given a "preserve" instruction whose graded form was
    under-determined, both agents
    re-serialized the HTML (pretty-print / entity re-encode) — adding an
    untestable transformation and failing identically. The lesson is not
    "always preserve bytes" but "do not add what you cannot check". It must
    be anchored on BOTH the prose and the enforced Frame acceptance.
    """
    p = SYSTEM_PROMPT_STATIC
    assert "MINIMAL-UNCERTAINTY DEFAULT" in p
    assert "do not add what you cannot check" in p
    # it is a POSTURE, with preserve as one instance — not a fixed rule
    assert "POSTURE, not a fixed rule" in p
    low = " ".join(_FRAME_STEP_ACCEPTANCE).lower()
    assert "minimal-uncertainty default" in low
    assert "do not add what you cannot check" in low
    assert "posture, not a fixed rule" in low


def test_minimal_uncertainty_is_self_observable():
    """The posture must come with a grader-free self-check (diff vs required
    form/input; every difference traces to an explicit requirement)."""
    p = SYSTEM_PROMPT_STATIC
    assert "Self-observable test (no grader needed)" in p
    assert "must trace to an explicit task requirement" in p
    low = " ".join(_FRAME_STEP_ACCEPTANCE).lower()
    assert "self-observable test" in low
    assert "trace to an explicit task requirement" in low


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


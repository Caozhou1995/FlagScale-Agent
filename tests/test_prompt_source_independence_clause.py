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

"""Anti-decay pins for the check-source-independence doctrine (was PRINCIPLE 2.1a).

The numbered 2.1a clause no longer exists in the current prompt (2026-11
ruling: adapt the tests to the current bytes, never the production code).
The doctrine survived folded into the loop sections, so the pins below assert
the RELATION each mechanical move must state, swept over the whole
SYSTEM_PROMPT_STATIC. Semantic mutations that must still fail: dropping the
shared-source mechanism, dropping any mechanical move, making a check
unfalsifiable, or detaching verification depth from error cost.
"""

import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _slice() -> str:
    # The "2.1a ... 2.2" anchor text is gone from the current prompt, so the
    # sweep runs over the whole static prompt, whitespace-normalized.
    return " ".join(SYSTEM_PROMPT_STATIC.split())


@pytest.fixture(scope="module")
def clause() -> str:
    return _slice()


def test_clause_names_the_failure_class_as_shared_origin(clause):
    # the mechanism, not a virtue: agreement between same-source checks proves nothing
    assert "two sides sharing one source cannot fail" in clause
    assert "A green whose verdict you authored is not evidence" in clause


def test_clause_states_the_premise_cascade(clause):
    # why more checking cannot fix it: a wrong premise corrupts everything downstream
    assert "where a wrong premise is cheapest to fix and hardest to see" in clause
    assert "load-bearing premise you could not test is written down as a labelled gap" in clause


def test_clause_carries_all_three_mechanical_moves(clause):
    for move in (
        "Consult memory for that class before your first substantive guess",
        "feed it an artifact you built to FAIL that assumption",
        "A difference must name the baseline it was measured against",
    ):
        assert move in clause, f"lost mechanical move: {move}"


def test_perturbation_is_falsifying_not_suggestive(clause):
    # the operational core: a green check must be able to go red to count as evidence
    assert "Could this check have gone red" in clause
    assert "does the artifact behave as the field says it must" in clause


def test_membership_is_verbatim_and_on_disk(clause):
    # evidence must be read off the target's own bytes, not reconstructed
    assert "the target's own bytes under the reader named above" in clause
    assert "The artifact on disk outranks the story in your head" in clause


def test_symmetry_names_the_asymmetry_itself_as_the_flag(clause):
    # a discrepancy you can argue away is a discrepancy you never tested
    assert "A deviation you can argue away went untested" in clause
    assert "the DELIVERABLE being wrong" in clause


def test_timing_binds_to_contradiction_or_closeout(clause):
    # when the check must run: at premise formation, or latest at close-out
    assert "before the first tool call" in clause
    assert "declare it at that moment" in clause
    assert "close-out" in clause


def test_economics_links_fast_independent_feedback_to_less_enforcement(clause):
    # the density claim, kept falsifiable: error cost sets verification strength
    assert "Error cost sets verification strength" in clause
    assert "cheap visible mistakes get one check" in clause


def test_no_task_specific_leak():
    s = _slice()
    for leak in ("oligotm", "bsai", "primer", "anneal", "fasta", "dna", "mteb", "humaneval"):
        assert leak not in s.lower(), f"leaked task identifier: {leak}"

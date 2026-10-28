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

"""Anti-decay pins for the PRINCIPLE 2.1a clause (source independence).

Same class as test_prompt_clause_anti_decay.py: presence-only assertions lock a
token shape, not the doctrine, so these tests assert the RELATION each
mechanical move must state, scoped to the 2.1a slice. Semantic mutations that
must fail: softening "zero information" into a suggestion, dropping any of the
three mechanical moves, making the timing advisory-only, or detaching the
economics (fast independent feedback) from the claim.
"""

import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _slice() -> str:
    start = SYSTEM_PROMPT_STATIC.index("2.1a A CHECK IS ONLY AS GOOD")
    end = SYSTEM_PROMPT_STATIC.index("2.2 VERIFY AT THE ALTITUDE")
    return " ".join(SYSTEM_PROMPT_STATIC[start:end].split())


@pytest.fixture(scope="module")
def clause() -> str:
    return _slice()


def test_clause_names_the_failure_class_as_shared_origin(clause):
    # the mechanism, not a virtue: the check's DATA shares origin with the judged thing
    assert "shares its origin" in clause
    assert "zero information" in clause  # strength: not "less reliable"


def test_clause_states_the_premise_cascade(clause):
    # why more checking cannot fix it: a wrong premise corrupts every downstream test
    assert "wrong PREMISE" in clause
    assert "every downstream test" in clause


def test_clause_carries_all_three_mechanical_moves(clause):
    for move in ("SOURCE PERTURBATION", "PROVENANCE MEMBERSHIP", "SYMMETRIC SOURCING"):
        assert move in clause, f"2.1a lost mechanical move: {move}"


def test_perturbation_is_falsifying_not_suggestive(clause):
    # the operational core: input claimed as load-bearing, changed, deliverable must move
    assert "change one element" in clause
    assert "does not move" in clause
    assert "never used that input" in clause


def test_membership_is_verbatim_and_on_disk(clause):
    assert "verbatim substring" in clause
    assert "on disk" in clause  # not a reconstruction


def test_symmetry_names_the_asymmetry_itself_as_the_flag(clause):
    assert "SAME source" in clause
    assert "asymmetry is itself the red flag" in clause


def test_timing_binds_to_contradiction_or_closeout(clause):
    # when the check must run: at premise formation, or latest at close-out
    assert "the moment the premise is formed" in clause
    assert "close-out" in clause


def test_economics_links_fast_independent_feedback_to_less_enforcement(clause):
    # the density claim the task asked for, kept falsifiable: six blind rounds -> two
    assert "LOWER the latency" in clause
    assert "six blind" in clause and "two converging rounds" in clause


def test_no_task_specific_leak():
    s = _slice()
    for leak in ("oligotm", "bsai", "primer", "anneal", "fasta", "dna", "mteb", "humaneval"):
        assert leak not in s.lower(), f"2.1a leaked task identifier: {leak}"

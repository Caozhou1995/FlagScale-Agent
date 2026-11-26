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

"""Anti-decay pins for "do not arbitrate a choice by guessing the consumer".

Why this clause exists (measured run, reward 0, no exception, full budget
spent): the agent faced a real choice, wrote out BOTH hypotheses explicitly,
and then settled it by reasoning about how the consumer's implementation
probably parses the artifact -- "if the grader parses from position 0, my extra
prefix breaks it" -- a claim about bytes it had never read and could not test.
It shipped the form that omitted a requirement the governing standard carried,
and the single failing assertion was exactly that omitted requirement.

What makes this distinct from its two neighbours:
  * `test_prompt_definition_axis_clause.py` pins "enumerate the axis".
  * `test_prompt_rejected_reading_column_clause.py` pins "compute the rejected
    reading as a column".
This agent had already enumerated the axis and considered both readings. What
it did INSTEAD of computing was arbitrate by imagined implementation. So the
repair has two halves, and both must stay resident:
  (1) STRICTEST MEANS WORST CASE -- settle against what the standard must
      survive at its worst (largest input, tightest tolerance), never against
      the minimal form that happens to clear the sample at hand;
  (2) COMPLIANT READINGS ARE COMPATIBLE -- when every defensible reading
      already conforms, the choice does not exist: satisfy them together
      instead of shipping one side.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _flat() -> str:
    """Whitespace-normalised source so line wrapping cannot hide a clause."""
    return re.sub(r"\s+", " ", SRC.read_text(encoding="utf-8"))


FLAT = _flat()

CLAUSE_HEAD = "Do not close a choice by guessing the consumer"
NEXT_SECTION = "THE TRAP INVENTORY"


def _clause() -> str:
    """The clause body, from its head to the start of the next sub-section.

    Scoped slice so the assertions below cannot be satisfied by a sentence
    that happens to live somewhere else in the prompt.
    """
    start = FLAT.find(CLAUSE_HEAD)
    assert start != -1, "the no-guessing clause head is missing from the prompt"
    end = FLAT.find(NEXT_SECTION, start)
    assert end != -1, f"clause has no following {NEXT_SECTION} -- prompt shape changed"
    return FLAT[start:end]


def test_clause_exists_inside_section_5():
    """Present, non-trivial, and inside the ruling section."""
    assert CLAUSE_HEAD in FLAT
    body = _clause()
    assert len(body) > 200, f"clause body unexpectedly short: {len(body)} chars"
    sec5 = FLAT.find("UNDERDETERMINED CHOICES")
    assert sec5 != -1 and sec5 < FLAT.find(CLAUSE_HEAD), (
        "clause must sit in the UNDERDETERMINED CHOICES ruling section"
    )


def test_guessing_the_consumer_is_named_as_a_bet_not_evidence():
    """The failure mechanism, named at the level of the act performed."""
    body = _clause()
    assert "guessing the consumer" in body
    assert "bet on unread bytes" in body


def test_the_guess_is_traced_to_its_provenance_failure():
    """Why an existing self-check does not save it: E4 provenance."""
    body = _clause()
    assert "E4" in body, "clause must cite the provenance rule that voids the guess"


def test_strictest_is_defined_as_worst_case_not_as_sample_clearance():
    """Half (1): 'strictest' must mean worst case, not minimal-but-passing."""
    body = _clause()
    assert "WORST case the standard must survive" in body
    assert "largest input" in body
    assert "tightest tolerance" in body
    assert "not the minimal sample-clearing form" in body


def test_compliant_readings_must_be_satisfied_together():
    """Half (2): compatibility, not a side-pick, when everything conforms."""
    body = _clause()
    assert "every reading that complies" in body
    assert "satisfy TOGETHER" in body


def test_a_compliance_derived_conclusion_cannot_be_vetoed_by_silence():
    """Half (3): the asymmetry of the two reversals.

    Adding what every reading demands is justified by compliance alone; removing
    it is justified only by evidence. So a conclusion already backed by the
    ordering rule or the strictest-standard rule must not be repealed by
    imagining a consumer that dislikes it -- the imagined side carries no bytes.
    """
    body = _clause()
    assert "repeal a conclusion already backed by" in body, (
        "clause must name the asymmetric repeal case"
    )
    assert "overturn it only with a task sentence, a measurement or observed bytes" in body, (
        "clause must state the evidence floor for overturning"
    )
    # The repeal rule must NOT be phrased as an absolute ban -- evidence still wins.
    assert "cannot REPEAL" not in body and "never repeal" not in body.lower(), (
        "an absolute ban would block a genuinely evidenced reversal"
    )


def test_clause_stays_task_agnostic():
    """Scientific pattern only -- no task nouns that would leak the live instance."""
    body = _clause()
    leaks = [
        "primer", "Primer", "BsaI", "bsai", "NEB", "GGTCTC", "ggtctc",
        "clamp", "overhang", "anneal", "dna-assembly", "plasmid",
        "primers.fasta",
    ]
    for word in leaks:
        assert word not in body, f"task-specific literal leaked into clause: {word!r}"


def test_clause_does_not_prescribe_reading_the_grader():
    """The clause must NOT recommend inspecting the consumer's implementation.

    That route is unavailable in this environment, so a clause recommending it
    would be an unexecutable instruction -- the exact failure it replaces.
    """
    body = _clause().lower()
    for banned in ["read the grader", "inspect the grader", "open the tests",
                   "read the tests", "look at the verifier"]:
        assert banned not in body, f"clause recommends an unexecutable route: {banned!r}"

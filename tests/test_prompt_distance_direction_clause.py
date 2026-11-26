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
"""Pins the DISTANCE-vs-DIRECTION half of a placement convention.

Measured failure this file targets (a run that had a rule quote in hand and
still landed on the wrong side): the agent transcribed a vendor convention
that adds a number of bases relative to a named recognition feature, recorded
the COUNT correctly, then placed those bases on the far side of the feature
from the one the convention meant. Every count in its own notes agreed with
itself; the consumer's structural assertion rejected the bytes because the
feature sat at the very start of the string.

That is a different failure from "never read the rule". The rule was quoted in
the agent's own reasoning one paragraph earlier. What was missing is the
sentence that names distance and side as TWO independent facts, and asks for
the side to be asserted on the delivered bytes.

The sibling clause `test_prompt_definition_axis_clause.py` pins "enumerate the
axis"; `test_prompt_rejected_reading_column_clause.py` pins "compute the
rejected construction as a column". This file pins the placement half: the same
number on either side of a feature satisfies every arithmetic check.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _flat() -> str:
    """Whitespace-normalised source so line wrapping cannot hide a clause."""
    return re.sub(r"\s+", " ", SRC.read_text(encoding="utf-8"))


FLAT = _flat()

CLAUSE_HEAD = "A quoted rule can land on the wrong side."


def _clause() -> str:
    """Scoped slice: from the clause head up to the next sub-section marker."""
    start = FLAT.find(CLAUSE_HEAD)
    assert start != -1, "the placement-convention clause head is missing from the prompt"
    end = FLAT.find("(d) THE CLOSE-OUT QUESTION", start)
    assert end != -1, "the clause has no following (d) section -- prompt shape changed"
    return FLAT[start:end]


def test_clause_exists_and_is_substantial():
    body = _clause()
    assert len(body) > 300, f"clause body unexpectedly short: {len(body)} chars"


def test_distance_and_direction_are_named_as_two_independent_facts():
    """The core claim: a distance and a side are separate facts."""
    body = _clause()
    assert "DISTANCE from a named feature" in body
    assert "DIRECTION from it" in body
    assert "two independent facts" in body


def test_same_count_on_either_side_satisfies_every_number():
    """The failure mechanism, stated as a falsifiable property."""
    body = _clause()
    assert "the same count placed before instead of after" in body
    assert "satisfies every number and still fails" in body


def test_requires_asserting_the_side_on_the_delivered_bytes():
    """The repaired behaviour: an order/offset assertion on the artifact."""
    body = _clause()
    assert "assert the SIDE on the delivered bytes" in body
    assert "must not sit at the very start" in body
    assert "prefix is non-empty" in body


def test_a_count_or_substring_test_is_named_as_insufficient():
    """Why the naive check passes: it accepts either side."""
    body = _clause()
    assert "A count or substring test accepts either" in body
    assert "only an order/offset" in body


def test_clause_stays_task_agnostic():
    """Scientific pattern only -- no task nouns that would leak the live instance."""
    body = _clause().lower()
    for banned in ("oligotm", "primer3", "bsai", "overhang", "anneal",
                   "dna-assembly", "plasmid", "circular", "clamp",
                   "cut site", "restriction site", "recognition site",
                   "ggtctc", "golden gate", "melting", "temperature",
                   "primer", "nucleotide", "enzyme"):
        assert banned not in body, banned

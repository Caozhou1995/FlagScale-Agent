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

"""Anti-decay pins for the SLOT-DECLARATION clause.

Why this file exists (measured, single-trial run 2026-11-23 19:06:50): the agent
consulted the authoritative manual, FOUND the requirement, wrote it into prose --
and still shipped a formula whose slots omitted it, because the formula had been
fixed several hundred lines earlier and every artifact-level check was blind to a
property that never enters the artifact. The delivered bytes failed the consumer
on exactly that omitted slot, while the agent's own checks were all green.

The capability this pins: a structural formula declares WHICH SLOTS EXIST; a
position absent from the template you will generate from is absent from
everything downstream, however the artifact-level checks read. The moment to map
the authority's terms onto your own formula is BEFORE the formula is fixed, and a
term the formula has no place for must be recorded as a decision, never omitted
silently.

Task-agnostic on purpose: no domain nouns, so this file adds no instance leak.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
FLAT = re.sub(r"\s+", " ", SRC.read_text(encoding="utf-8"))


def test_slot_declaration_headline():
    """The headline rule: presence is decided at the formula, not at the artifact."""
    assert "A structural formula declares which slots exist" in FLAT


def test_absence_propagates_downstream():
    """Why the artifact-level checks cannot save you: the slot never gets there."""
    assert "absent from everything downstream" in FLAT


def test_mapping_happens_before_the_formula_is_fixed():
    """The rule needs a moment in time, or it arrives after the miss."""
    assert "Before the template is fixed" in FLAT


def test_authority_terms_are_mapped_onto_your_own_formula():
    """Whose terms, whose formula -- the mapping has a direction."""
    assert "map the authority's own terms onto" in FLAT


def test_an_unmappable_term_is_a_decision_not_a_silent_omission():
    """The exit for a term that does not fit: record it, do not drop it."""
    assert "record any term your formula has no place for as a decision" in FLAT
    assert "never omit it silently" in FLAT


def test_clause_names_no_task_instance():
    """Mechanism only: a domain literal here would leak an answer."""
    start = FLAT.find("A structural formula declares which slots exist")
    assert start != -1
    clause = FLAT[start : start + 420].lower()
    for banned in (
        "ggtctc",
        "bsai",
        "primer",
        "clamp",
        "overhang",
        "backbone",
        "plasmid",
        "raman",
    ):
        assert banned not in clause, banned

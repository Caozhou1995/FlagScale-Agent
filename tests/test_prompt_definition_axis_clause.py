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

"""Anti-decay pins for the DEFINITION-AXIS requirement (current wording).

Why this file exists (measured terminal-bench run): the agent DID sweep design parameters -- a full `for Lf in 15..36:
for Lr in 15..36:` grid calling the real temperature tool on every candidate --
and it DID apply the knob/intersection rule. It still scored 0, because it
pinned the MEASUREMENT DEFINITION to one construction of the measured input
(the annealing tract alone) while the deciding party constructed that same
quantity over the annealing tract PLUS the leading bases of the 5' tail that
re-match the template. The design axis was swept; the definition axis was not,
and every row of its table agreed with itself.

Adapted to the current prompt rewrite (ruling: current code wins, tests
follow). The old PRINCIPLE clauses were re-homed into Section 3 (a/b/c)
of THE LOOP; each test below quotes the current verbatim clause that carries
the same capability: the consumer's span/input construction is an axis, it is
swept like any other, and an un-swept axis leaves the miss invisible.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _flat() -> str:
    """Whitespace-normalised source so line wrapping cannot hide a clause."""
    return re.sub(r"\s+", " ", SRC.read_text(encoding="utf-8"))


FLAT = _flat()


def test_span_not_formula_ambiguity_is_pinned():
    """The measured failure: not the formula, but WHICH SPAN feeds it."""
    assert "The ambiguity is rarely the formula but the span fed in" in FLAT


def test_consumer_fixes_the_span_so_sweep_it():
    """The consumer's input construction is an axis: name how it fixes the
    span, and sweep that parameter like any other."""
    assert "name how the consumer fixes its span and sweep that parameter" in FLAT


def test_reading_set_built_from_self_picked_offsets_is_incomplete():
    """The wrong-family verdict: a reading set you built from your own choices
    cannot cover the deciding construction."""
    assert "A reading set built from offsets you picked is incomplete by construction" in FLAT


def test_axes_multiply_and_are_enumerated():
    """The definition axis is one of several independent axes; enumerate, do
    not anchor on a counted number."""
    assert "An axis is a choice no fact settled, and axes multiply" in FLAT
    assert "Never anchor on a counted number of variants" in FLAT


def test_do_not_trust_the_agreement_of_unchecked_rows():
    """Every row of the sweep agreeing is not evidence: the untried members
    must be checked before the agreement is trusted (the invisible miss)."""
    assert "check the ones you did not try before you trust the agreement" in FLAT


def test_unseen_gate_axes_are_swept():
    """The clause generalises to any deciding party's gate (format, boundary,
    unit, scale, order, defaults)."""
    assert ("it varies along axes too (format, boundary, unit, scale, order, "
            "defaults)") in FLAT


def test_independent_computation_paths_for_each_axis():
    """Compute each reading by a path that does not share source with the
    others -- the shared-source check cannot fail."""
    assert ("compute each by a path that does not share source with the "
            "others") in FLAT


def test_clause_stays_task_agnostic():
    """Scientific pattern only -- no task nouns that would leak the instance.

    Measured addition (2026-11-05): a clause shipped with "a clamp 5' of a cut
    site" as its illustrative example -- a verbatim instance of a live failing
    task -- and this gate passed it, because neither "clamp" nor "cut site" was
    on the list. The gate was only as wide as the nouns someone had already
    tripped over. Any domain noun that names a concrete instance belongs here.
    """
    for banned in ("oligotm", "primer3", "bsai", "overhang", "anneal",
                   "dna-assembly", "plasmid", "circular",
                   "clamp", "cut site", "restriction site", "recognition site",
                   "ggtctc", "golden gate", "lz77", "deflate", "metacirc"):
        assert banned not in FLAT.lower(), banned

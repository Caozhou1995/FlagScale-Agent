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

"""Anti-decay pins for PRINCIPLE 2.4 / 2.9's DEFINITION-AXIS requirement.

Why this file exists (measured 2026-10-28, terminal-bench dna-assembly trial
fbokQzg): the agent DID sweep design parameters -- a full `for Lf in 15..36:
for Lr in 15..36:` grid calling the real temperature tool on every candidate --
and it DID apply the knob/intersection rule. It still scored 0, because it
pinned the MEASUREMENT DEFINITION to one construction of the measured input
(the annealing tract alone) while the deciding party constructed that same
quantity over the annealing tract PLUS the leading bases of the 5' tail that
re-match the template. The design axis was swept; the definition axis was not,
and every row of its table agreed with itself.

So the clause must assert two things that cannot be paraphrased away:
  (a) the axis list is not the object's axes alone -- the consumer's INPUT
      CONSTRUCTION is a second kind of axis, and
  (b) the knob search must fold the definition axis into the grid BEFORE
      choosing a value, or it returns "the best of the wrong family".
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _flat() -> str:
    """Whitespace-normalised source so line wrapping cannot hide a clause."""
    return re.sub(r"\s+", " ", SRC.read_text(encoding="utf-8"))


FLAT = _flat()


def test_axis_list_covers_the_consumers_input_construction_not_only_the_object():
    assert "each axis the CONSUMER's input construction varies along" in FLAT


def test_the_two_kinds_of_axis_are_named_as_different_kinds():
    assert "Those two kinds of axis are different axes" in FLAT


def test_the_failure_shape_is_stated_so_the_clause_cannot_be_softened():
    """The exact miss: sweep design params, pin the definition, lose the run."""
    assert ("sweeping the design parameters while pinning the measurement "
            "definition leaves the deciding dimension unscanned") in FLAT


def test_hidden_party_ambiguity_is_located_in_the_span_not_the_formula():
    assert "the ambiguity is rarely in its formula" in FLAT
    assert "almost always in WHICH SPAN gets fed to that formula" in FLAT


def test_knob_search_must_fold_the_definition_axis_into_the_grid_first():
    assert "fold the definition axis into the grid FIRST" in FLAT
    assert ("sweep the design parameter and each defensible construction of "
            "the measured input TOGETHER") in FLAT


def test_knob_search_under_a_pinned_definition_is_named_as_the_wrong_family():
    assert ("A knob search whose reading set is missing the deciding member "
            "returns a confident answer that is merely the best of the wrong "
            "family") in FLAT


def test_the_miss_is_invisible_from_inside_because_every_row_agrees():
    assert ("the miss is invisible from inside, because every row agrees") in FLAT


def test_2_9_axis_count_points_back_at_both_kinds_of_axis():
    assert ("the count is set by the axes the object AND the consumer's input "
            "construction vary along") in FLAT


def test_clause_stays_task_agnostic():
    """Scientific pattern only -- no task nouns that would leak the instance."""
    for banned in ("oligotm", "primer3", "bsai", "overhang", "anneal",
                   "dna-assembly", "plasmid", "circular"):
        assert banned not in FLAT.lower(), banned

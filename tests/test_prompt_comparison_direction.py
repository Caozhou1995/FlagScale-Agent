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

"""Regression tests for the COMPARISON-DIRECTION discipline.

Evidence class: an agent burned its remaining budget hunting spurious
candidates while the grader's loop structure made those candidates harmless.
The grader iterated over ITS OWN reference rows and searched the delivered
matrix for a match, so any extra delivered row was never inspected: extras
were free and omission was the only losing direction. The agent, believing it
must produce a tidy candidate set, spent its time eliminating the extras.

A clause that only says "be careful with cardinality" cannot prevent this,
because the two comparison shapes demand opposite behaviour. The load-bearing
content is: read the loop structure, name which direction it penalizes, and
spend effort accordingly.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

DIRECTION_CLAUSE = 'Which direction does the grader\'s comparison PENALIZE?'


def _clause() -> str:
    start = SYSTEM_PROMPT_STATIC.index(DIRECTION_CLAUSE)
    return SYSTEM_PROMPT_STATIC[start:SYSTEM_PROMPT_STATIC.index("\n- ", start)]


def test_direction_clause_present():
    assert DIRECTION_CLAUSE in SYSTEM_PROMPT_STATIC


def test_clause_says_read_loop_structure_not_tolerance():
    clause = _clause()
    assert re.search(r"LOOP STRUCTURE", clause)
    assert re.search(r"not just its tolerance", clause)


def test_clause_states_extras_are_free_in_reference_iteration():
    clause = _clause()
    assert re.search(r"extras are FREE", clause)
    assert re.search(r"omission is the ONLY losing direction", clause)


def test_clause_states_bijection_penalizes_extras():
    clause = _clause()
    assert re.search(r"bijection", clause)
    assert re.search(r"extras ARE penalized", clause)


def test_clause_forbids_pruning_to_look_tidy():
    clause = _clause()
    assert re.search(r"never discard a plausible candidate to make your count look tidy", clause)


def test_direction_clause_is_generic():
    clause = _clause()
    for banned in ("stolen_A1", "model-extraction", "randn(30", "Failed to match rows"):
        assert banned not in clause, f"clause leaked trial-specific text: {banned}"

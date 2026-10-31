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

"""Anti-decay pins for PRINCIPLE 2.4's VALUE requirement on enumerated readings.

Why this file exists (measured 2026-10-28, terminal-bench dna-assembly trial
euF5sCM): the agent wrote out both readings of the annealing-region definition by
NAME, quoted the doctrine's "knob / intersection" move, and declared "I'll design
so both readings pass" -- then computed only reading A (Tm 70.98). Reading B, the
one the grader used, was 73.39 and out of the stated bound. The deciding number
appeared nowhere in its trace.

The clause as written only demanded that the readings be ENUMERATED, so a trace
that listed them by name satisfied the letter of it. These tests pin the repair:
a reading with no computed value is a mention, not an enumeration.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

# The static prompt is wrapped across source lines; normalise whitespace before
# matching so the assertion is about the SENTENCE, not the line-wrapping.
FLAT = " ".join(SYSTEM_PROMPT_STATIC.split())


def test_a_reading_counts_only_once_it_has_a_computed_value():
    assert "A reading is ENUMERATED only once it carries a computed VALUE." in FLAT


def test_a_list_of_names_is_explicitly_not_a_list_of_readings():
    """F1: the failure was a name-only list, so the clause must name that shape."""
    assert "A list of names is not a list of readings" in FLAT


def test_every_reading_goes_into_one_table_and_the_values_are_diffed():
    assert "put its value in the same table and diff them before you choose" in FLAT


def test_the_trace_tell_is_stated_so_uptake_is_checkable_from_the_log():
    """F2: a greppable tell turns 'did it enumerate?' into a mechanical check."""
    assert "a reading whose value you cannot grep for was not enumerated, it was mentioned" in FLAT


def test_the_measured_miss_is_recorded_so_the_clause_cannot_be_softened_back():
    assert "never computed the second" in FLAT
    assert "that one was out of bounds and it was the deciding one" in FLAT

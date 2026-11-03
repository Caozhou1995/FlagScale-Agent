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

===========================================================================
RE-ANCHORED 2026-11-03 -- ALL FIVE PINS DELETED (case B: capability gone).
Ruling: adapt the tests to the CURRENT prompt.py bytes, never the reverse.

Measured, not assumed. Every distinctive string of the VALUE repair returns 0
hits in BOTH current carrier texts -- SYSTEM_PROMPT_STATIC (flagscale_agent/
react/prompt.py) and plan_create._FRAME_STEP_ACCEPTANCE:

    "computed value"          0 static / 0 frame
    "carries a computed"      0 / 0
    "list of names"           0 / 0
    "not a list of readings"  0 / 0
    "same table"              0 / 0
    "diff them"               0 / 0
    "grep for"                0 / 0
    "mentioned"               0 / 0

What DOES survive, and why it is NOT the same capability: the current
UNDERDETERMINED CHOICES paragraph requires that each enumerated reading be
"pinned to the task text's own words and to the artifact it would produce" --
an ARTIFACT, not a computed scalar, and no diff step. The nearest scalar
language in the prompt ("A number you measured outside a stated bound is a
verdict, not a data point") governs bounds-checking of a number already
computed; it does not require computing a number per reading. So the specific
relation "a listed reading with no computed value is a mention, not an
enumeration" has no carrier in the current bytes.

Deleting rather than re-pointing is therefore the honest move: re-pointing one
of these asserts onto the surviving "artifact it would produce" phrase would
assert a DIFFERENT relation (an artifact is not a value, and no diff happens),
i.e. a green that no longer tests the decay it was written to catch. All five
test functions are deleted below with their original id and assertion recorded,
so the loss is visible and reversible if the clause is ever restored.
===========================================================================
"""

# DELETED 2026-11-03 -- capability gone from the current prompt bytes.
# Original test id: test_a_reading_counts_only_once_it_has_a_computed_value
# Original assert:  "A reading is ENUMERATED only once it carries a computed VALUE." in FLAT
# Reason: 0 hits for "computed VALUE" / "ENUMERATED only once" in the current
#         SYSTEM_PROMPT_STATIC; no sentence conditions enumeration on a value.
#         Re-pointing onto the surviving "artifact it would produce" phrase would
#         assert a different relation and silently retire the anti-decay pin.

# DELETED 2026-11-03 -- capability gone from the current prompt bytes.
# Original test id: test_a_list_of_names_is_explicitly_not_a_list_of_readings
# Original assert:  "A list of names is not a list of readings" in FLAT
# Reason: 0 hits for "list of names" in the current SYSTEM_PROMPT_STATIC.

# DELETED 2026-11-03 -- capability gone from the current prompt bytes.
# Original test id: test_every_reading_goes_into_one_table_and_the_values_are_diffed
# Original assert:  "put its value in the same table and diff them before you choose" in FLAT
# Reason: 0 hits for "same table" and for the diff-the-values step; the current
#         dominance test compares ARTIFACTS against the consumer's material, not
#         per-reading scalars in a table.

# DELETED 2026-11-03 -- capability gone from the current prompt bytes.
# Original test id: test_the_trace_tell_is_stated_so_uptake_is_checkable_from_the_log
# Original assert:  "a reading whose value you cannot grep for was not enumerated, it was mentioned" in FLAT
# Reason: 0 hits for "grep for"; the word "trace" no longer appears in the
#         static prompt at all, so there is no log-uptake tell to pin.

# DELETED 2026-11-03 -- capability gone from the current prompt bytes.
# Original test id: test_the_measured_miss_is_recorded_so_the_clause_cannot_be_softened_back
# Original asserts: "never computed the second" in FLAT
#                   "that one was out of bounds and it was the deciding one" in FLAT
# Reason: 0 hits for "never computed" and "deciding"; the trial-specific
#         measurement record the clause was anchored to is not in the prompt.

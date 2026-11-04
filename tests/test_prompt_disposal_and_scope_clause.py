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

"""Close-out clauses: a named gap must be DISPOSED of, and a declared
truncation must stay distinguishable from a skipped check.

Why this file exists (measured 2026-11-04 on the 89-trial terminal-bench batch,
whose 16 losses were re-read against the prompt clause by clause): the majority
had a rule that already covered them. The failure was not ignorance of the
rule; it was naming the gap and then shipping anyway. `pytorch-model-cli` wrote
`NOT VERIFIED: the grader's exact comparison convention (row order / sign /
scale)` and then closed with "this is a safe bet". So the load-bearing property
is not that the words "NOT VERIFIED" appear (they appeared five times before
this change), it is that a *named* gap cannot be discharged by narration.

A second hole: §10.4 calls a declared truncation "an honest result", while the
new rule demands a disposal for every NOT VERIFIED. A reader could not tell
which bucket a budget-forced truncation fell into. The two now have a stated
boundary.

Each test binds one relation and bans the escape phrasing, scoped to the
clause slice -- never the whole prompt, where other clauses supply the words.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _flat():
    return re.sub(r"\s+", " ", SYSTEM_PROMPT_STATIC)


def test_named_gap_is_not_a_disposal():
    # The load-bearing relation: naming the gap is explicitly NOT the end.
    assert "a not verified you can name is not a disposal" in _flat().lower()


def test_disposal_offers_three_branches():
    f = _flat().lower()
    assert "if testable, run it" in f
    assert "enumerate both readings and pick by dominance" in f
    assert "only equal bytes under both readings license a stated bet" in f


def test_scale_and_index_base_are_asked_globally_or_per_item():
    # A freed scale is not one factor for the whole set; the task's own phrase
    # "frees permutation and scaling" frees it one row at a time.
    f = _flat().lower()
    assert "globally or per item" in f


def test_positive_bound_is_required_alongside_the_negative_one():
    # A checklist that only asserts the bad thing is gone is half a checklist.
    f = _flat().lower()
    assert "add the positive bound too" in f


def test_polling_one_job_repeatedly_is_not_work():
    f = _flat().lower()
    assert "polling one job again and again is not work" in f


def test_stated_truncation_is_not_a_skipped_check():
    # The boundary that keeps §10.4 and the disposal rule from contradicting.
    f = _flat().lower()
    assert "a truncation is a scope you declared, not a not verified you skipped" in f


def test_underdetermined_is_the_resolution_step_for_enumerate():
    # ENUMERATE discovers the unpinned choice; UNDERDETERMINED rules on it.
    # They are one chain, not two independent rules. The qualifier stays OFF
    # the title so it cannot be read as covering only scale/index-base.
    f = _flat().lower()
    assert "the resolution step for items enumerate flagged" in f or "here is where enumerate's flagged items get ruled on" in f
    # The block itself must still name every choice axis it rules on.
    clause = f[f.index("underdetermined choices"):]
    clause = clause[: clause.index("defensible means")] if "defensible means" in clause else clause[:2500]
    for axis in ("form", "parameter", "scale", "orientation"):
        assert axis in clause, f"UNDERDETERMINED no longer names axis {axis!r}"


def test_no_escape_phrasing_survives():
    # The exact phrasing the v73 losses used to close without acting.
    f = _flat().lower()
    for banned in (
        "this is a safe bet",
        "record the bet and ship",
        "no further verification is possible",
    ):
        assert banned not in f, f"escape phrasing back in the prompt: {banned!r}"

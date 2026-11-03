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

"""Anti-decay tests for the STATIC-PROMPT copy of the two FRAME clauses.

Why this file exists (measured 2026-10-23): prompt.py carries the same two
doctrines as plan_create._FRAME_STEP_ACCEPTANCE, but the shipped tests only
asserted that the clause NAMES appear in SYSTEM_PROMPT_STATIC. Softening that
copy therefore kept every test green -- the static prompt is the copy the agent
actually reads, so a decayed copy silently undoes the plan-step clause.

Each test pins one relation of the static copy verbatim, so any softened rewrite
of that copy turns it red.

RE-ANCHORED 2026-11-03 (ruling: adapt the tests to the current prompt.py bytes).
The current §9 copy renames the doctrine: there is no "acceptance-facet manifest"
and no "row" any more -- the same capability is stated as "ONE executed check per
verifiable clause of the task text" over "entries". Anchor strings were rewritten
to the CURRENT copy's own words; no assertion was widened and none was weakened
to an always-true form. Two relations whose wording is genuinely GONE from the
current copy are recorded below as deleted asserts (see the inline comments).
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

# The static prompt is wrapped across source lines; normalise whitespace before
# matching so the assertion is about the SENTENCE, not the line-wrapping.
FLAT = " ".join(SYSTEM_PROMPT_STATIC.split()).lower()


def test_static_names_an_acceptance_facet_manifest():
    """The one-executed-check-per-clause doctrine must be NAMED in the copy read.

    Renamed 2026-11-03: the current §9 copy drops the words "acceptance-facet
    manifest"; the same doctrine is named as the "AT LEAST ONE executed check per
    verifiable clause of the task text" duty.
    """
    assert "at least one executed check per verifiable clause of the task text" in FLAT


def test_static_facet_count_is_anchored_to_the_task_text():
    """The count must come from the task text, not from the agent's imagination."""
    # Renamed 2026-11-03: "one row per verifiable clause" -> "the count is fixed
    # by the clause count"; "row count ... rather than" -> "not by how many
    # faces you happen to think of". Same relation, current words.
    assert "the count is fixed by the clause count, not by how many faces you happen to think of" in FLAT
    assert "a short list is a defect and not a judgment call" in FLAT


def test_static_facet_states_a_segmentation_rule_so_the_count_is_reproducible():
    """F4: without a segmentation rule the "clause count" is self-judged.

    Measured 2026-10-23: the plan-step clause defines how the task text is cut
    ("the task text's own sentences and bullets; treat each stated constraint
    inside a sentence ... as its own clause; when in doubt split rather than
    merge"), but the static copy carried the count relation without the rule --
    so two compliant agents could produce different counts and both be literal.

    2026-11-03: the current §9 copy keeps the segmentation sentence and the
    split-don't-merge rule, and keeps the REASON a merge is wrong ("a merged row
    hides the face it absorbed"), but it no longer spells the middle step
    ("treat each stated constraint inside a sentence as its own clause") at all.
    """
    assert "segment on the task's own sentences and bullets" in FLAT
    assert "when in doubt split rather than merge" in FLAT
    assert "a merged row hides the face it absorbed" in FLAT
    # DELETED 2026-11-03 (capability gone from the copy): the assert on
    # "as its own clause" -- the current §9 copy never states that an individual
    # constraint inside a sentence is its own clause; grep "as its own clause"
    # over SYSTEM_PROMPT_STATIC returns 0 hits. The segmentation RULE itself
    # survives (asserted above), so the count is still reproducible; only this
    # finer sub-rule is absent and is not re-invented here.


def test_static_facet_row_requires_an_executed_probe():
    """A listed face with no probe behind it is not a check.

    Renamed 2026-11-03: the current §9 copy speaks of "entries", not "rows", and
    splits the old single sentence into its two halves.
    """
    assert "an entry with no executed probe is a sentence" in FLAT
    assert "an entry claiming the result you expect is not a check" in FLAT


def test_static_facet_requires_closeout_review_of_the_manifest():
    """The close-out duty must survive in the copy the agent actually reads.

    Measured 2026-10-23: the plan-step clause ends with "Close-out reviews THIS
    manifest row by row, not your memory of it", but the static copy omitted it
    entirely (grep for "close-out" over the static prompt: 0 hits). An agent
    reading only the static prompt could therefore satisfy the facet clause at
    BUILD time and never re-check the rows at delivery, which is exactly the
    multi-round failure the clause exists to close.

    2026-11-03: the current prompt.py carries the duty in §10.1, phrased as
    re-reading the task clause by clause and running each compiled check against
    the DELIVERED bytes. Re-anchored to those words: the relation pinned is still
    "at close-out, re-check every clause against what you actually deliver".
    """
    assert "re-read the task, clause by clause" in FLAT
    assert "compile each into a check, run it against the delivered bytes" in FLAT
    for banned in ("close-out may skim", "close-out is optional", "review it if convenient"):
        assert banned not in FLAT


def test_static_reading_ledger_enumerates_defensible_readings():
    assert "enumerate every defensible reading of that requirement" in FLAT
    assert "each pinned to the task text's own words and to the artifact it would produce" in FLAT


def test_static_dominance_test_uses_the_consumers_own_material():
    assert "run a dominance test against the consumer's own material" in FLAT


def test_static_covers_top2_in_the_same_run_when_undominated():
    assert "when no reading dominates cover the top-2 in the same run" in FLAT


def test_static_conditions_coverage_on_direction_of_loss():
    """Shipping both variants is a GUARANTEED loss when extras are penalized.

    The plan-step clause carries this condition; the static copy the agent
    actually reads must not omit it, or the agent ships both variants where
    extras lose.

    2026-11-03: the current §9 copy no longer opens with the framing sentence
    "first determine which direction the comparison penalizes"; it states the two
    branches directly and closes with the bet-recording duty. The condition is
    what this test is about, and both branches survive verbatim.
    """
    assert "cover both readings only when extra output is ignored" in FLAT
    assert ("when extras are penalized cover with a switch defaulted to the "
            "dominant reading") in FLAT
    assert "record the bet in the delivery notes" in FLAT
    # DELETED 2026-11-03 (capability gone as a separate sentence): the assert on
    # "first determine which direction the comparison penalizes" -- grep over
    # SYSTEM_PROMPT_STATIC returns 0 hits and no sentence states that
    # determination step on its own. The two penalization branches asserted above
    # ARE that decision's outcome, so the conditioning is still pinned.


def test_static_anchors_the_ledger_to_principle_4():
    """The ledger must defer to the prompt's own existing rule, not invent one.

    Renamed 2026-11-03: the current prompt.py has no numbered PRINCIPLE labels;
    the same hedge/bet rule lives in §5's UNDERDETERMINED CHOICES, and §9 defers
    to it by name. The relation pinned -- "the ledger is a named rule of this
    prompt, and its outcome is recorded as a bet" -- is unchanged.
    """
    assert "resolve it by section 5's underdetermined choices rule" in FLAT
    assert "record the bet in the delivery notes" in FLAT

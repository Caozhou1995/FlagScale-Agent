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

Why this file exists (measured 2026-10-23): prompt.py:277 carries the same two
doctrines as plan_create._FRAME_STEP_ACCEPTANCE, but the shipped tests only
asserted that the clause NAMES appear in SYSTEM_PROMPT_STATIC. Softening that
copy therefore kept every test green -- the static prompt is the copy the agent
actually reads, so a decayed copy silently undoes the plan-step clause.

Each test pins one relation of the static copy verbatim, so any softened rewrite
of that copy turns it red.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

# The static prompt is wrapped across source lines; normalise whitespace before
# matching so the assertion is about the SENTENCE, not the line-wrapping.
FLAT = " ".join(SYSTEM_PROMPT_STATIC.split()).lower()


def test_static_names_an_acceptance_facet_manifest():
    assert "acceptance-facet manifest enumerates every face the consumer will drive" in FLAT


def test_static_facet_count_is_anchored_to_the_task_text():
    """The count must come from the task text, not from the agent's imagination."""
    assert "one row per verifiable clause of the task text" in FLAT
    assert "row count is fixed by the clause count rather than by what you happen to think of" in FLAT


def test_static_facet_states_a_segmentation_rule_so_the_count_is_reproducible():
    """F4: without a segmentation rule the "clause count" is self-judged.

    Measured 2026-10-23: the plan-step clause defines how the task text is cut
    ("the task text's own sentences and bullets; treat each stated constraint
    inside a sentence ... as its own clause; when in doubt split rather than
    merge"), but the static copy carried the count relation without the rule --
    so two compliant agents could produce different counts and both be literal.
    """
    assert "segment on the task text's own sentences and bullets" in FLAT
    assert "as its own clause" in FLAT
    assert "when in doubt split rather than merge" in FLAT


def test_static_facet_row_requires_an_executed_probe():
    assert "each row carries the exact probe you ran plus its observed result" in FLAT
    assert "a row without an executed probe is a sentence, not a check" in FLAT


def test_static_facet_requires_closeout_review_of_the_manifest():
    """The close-out duty must survive in the copy the agent actually reads.

    Measured 2026-10-23: the plan-step clause ends with "Close-out reviews THIS
    manifest row by row, not your memory of it", but prompt.py:277 -- the static
    copy -- omitted it entirely (grep for "close-out" over the static prompt:
    0 hits). An agent reading only the static prompt could therefore satisfy the
    facet clause at BUILD time and never re-check the rows at delivery, which is
    exactly the multi-round failure the clause exists to close.
    """
    assert "close-out reviews this manifest row by row" in FLAT
    assert "not your memory of it" in FLAT
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
    """
    assert "first determine which direction the comparison penalizes" in FLAT
    assert "cover both readings only when extra output is ignored" in FLAT
    assert ("when extras are penalized cover with a switch defaulted to the "
            "dominant reading") in FLAT


def test_static_anchors_the_ledger_to_principle_4():
    assert "principle 4's hedge/bet rule applied at build time" in FLAT

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

"""Anti-decay tests for the supplied-artifacts and axis-calibration clauses.

Why this file exists (measured 2026-10-24): two failed terminal-bench trials
shared one root cause -- the environment shipped the graded source of truth
(a model weights file; a spectra data file whose x axis needed a unit
transform) and the agent REPLACED it with its own synthesis (trained a model;
fit on raw axis values), so the grader's reference values could never match.
prompt.py now carries two clauses against exactly this failure family; each
test below pins one relation of the shipped copy so a softened rewrite turns
red (mutation gate: soften any pinned sentence and this file must fail).
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

FLAT = " ".join(SYSTEM_PROMPT_STATIC.split()).lower()


def test_static_orders_an_inventory_before_any_synthesis():
    assert "inventory the working directory first" in FLAT


def test_static_makes_supplied_artifacts_presumptive_truth():
    assert (
        "presumptively the graded source of truth even when the task text never mentions it"
        in FLAT
    )


def test_static_names_the_consequence_of_self_synthesis():
    """The clause must say WHY retraining/re-deriving cannot reproduce the
    reference values, not merely warn against it."""
    assert (
        "retraining your own model, re-deriving your own sequence, or re-collecting your own data cannot reproduce them"
        in FLAT
    )
    assert "self-built stand-in is almost certainly the wrong answer" in FLAT


def test_static_axis_clause_requires_known_feature_positions():
    """Axis calibration must be grounded in literature-standard feature
    positions -- not 'sanity check your data' vibes."""
    assert "known feature positions" in FLAT
    assert "you have a unit or column mismatch" in FLAT


def test_static_axis_clause_pins_feature_labels_to_fitted_positions():
    """2026-10-24 batch: both peaks were fitted correctly but their G/2D
    labels were swapped -- labels must come from position matching, not
    processing order."""
    assert (
        "never by the order you happened to encounter or process them" in FLAT
    )


def test_static_axis_clause_tests_transforms_numerically():
    """The same batch: the agent enumerated the reciprocal transform, then
    REJECTED it on physical grounds (anti-Stokes conventions) without ever
    computing 1e7/6329 = 1580.3. A physically-motivated objection must not
    override a numerically-matching transform."""
    assert "simplest numeric form" in FLAT
    assert (
        "must not override a transform that numerically matches" in FLAT
    )
    assert "an encoding you rejected without computing its numbers is not rejected" in FLAT


def test_static_axis_clause_forbids_fitting_on_raw_axis_values():
    assert "never fit on raw axis values" in FLAT

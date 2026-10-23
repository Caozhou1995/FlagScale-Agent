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

"""Anti-decay tests: each load-bearing ACTION of the two FRAME clauses is bound
literally, so a softened rewrite cannot keep this suite green.

Why this file exists (measured 2026-10-23): a fresh-context review ran six
semantic mutations of the two clauses -- merge rows, expect-instead-of-observe,
drop close-out review, make hedging optional, make the dominance test optional,
drop pinning to the task text -- and all six left the suite green. Only the two
mutations the shipped gates happened to cover went red. Same class as
pitfall/flagscale_agent/prompt_test_token_shape_only, third recurrence:
presence-only assertions lock a token SHAPE, not the doctrine. The clause text
was verified present in plan_create._FRAME_STEP_ACCEPTANCE; the hole was in the
tests, not the clauses.

Each test bans the specific escape phrasing and requires the specific relation,
scoped to the clause slice (never the whole prompt, where other clauses supply
the same words).
"""

from flagscale_agent.react.tools.plan_create import _FRAME_STEP_ACCEPTANCE

FACET = "Acceptance-facet manifest"
LEDGER = "Reading ledger"


def _clause(prefix):
    items = [i for i in _FRAME_STEP_ACCEPTANCE if i.startswith(prefix)]
    assert len(items) == 1, f"expected exactly one clause starting {prefix!r}, got {len(items)}"
    return " ".join(items[0].lower().split())


def test_facet_forbids_merging_rows():
    """Merging clauses into one row silently hides the face it absorbed."""
    t = _clause(FACET)
    assert "never fold several clauses into one row" in t
    for banned in ("may fold several clauses", "merge related clauses",
                   "combine clauses into a single row"):
        assert banned not in t


def test_facet_requires_an_observed_result_not_an_expected_one():
    """A row claiming what the agent BELIEVES would happen is not a check."""
    t = _clause(FACET)
    assert "observed result" in t and "you ran it" in t
    for banned in ("result you expect", "expected result you believe",
                   "what you expect to see"):
        assert banned not in t


def test_facet_requires_closeout_review_of_the_manifest():
    t = _clause(FACET)
    assert "close-out reviews this manifest row by row" in t
    for banned in ("close-out may skim", "close-out is optional",
                   "review it if convenient"):
        assert banned not in t


def test_facet_states_a_segmentation_rule_so_the_count_is_reproducible():
    """Without a segmentation rule the clause count is self-judged."""
    t = _clause(FACET)
    assert "segment on the task text's own sentences" in t
    assert "split rather than merge" in t


def test_ledger_coverage_is_not_optional():
    """Coverage is a duty, not a time-permitting extra."""
    t = _clause(LEDGER)
    assert "cover the top-2 (or top-k) in the same run" in t
    for banned in ("if you have time", "if cheap", "if convenient",
                   "cover the top-2 if"):
        assert banned not in t


def test_ledger_dominance_test_is_mandatory_before_choosing():
    t = _clause(LEDGER)
    assert "then run a dominance test" in t
    for banned in ("dominance test is optional", "dominance test if",
                   "may skip the dominance"):
        assert banned not in t


def test_ledger_readings_stay_pinned_to_the_task_text():
    t = _clause(LEDGER)
    assert "each pinned to the task text's own words" in t
    for banned in ("described in your own words",
                   "your preferred reading as the pin"):
        assert banned not in t


def test_ledger_conditions_coverage_on_direction_of_loss():
    """Shipping both variants is a GUARANTEED loss when extras are penalized."""
    t = _clause(LEDGER)
    assert "which direction the comparison penalizes" in t
    assert "cover both readings only when extra output is ignored" in t
    assert "when extras are penalized" in t

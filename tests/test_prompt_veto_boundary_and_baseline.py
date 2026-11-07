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

"""Two system-prompt clauses landed from real failed runs.

FORBIDDEN-LIST VETO BOUNDARY.
Evidence (measured run, reward=0): the agent
wrote a forbidden item "don't add unless confirmed" and used it to veto a domain
principle it had (a) recalled from durable memory and (b) itself proved harmless
by local boundary analysis — because external confirmation was impossible in the
no-network container. It shipped the no-clamp form and died on the grader's
clamp gate. A self-authored prohibition must be able to veto only what is known
harmful or costly to reverse, NOT whatever lacks external confirmation.

A DIFFERENCE MUST NAME ITS BASELINE.
A comparison whose baseline was re-measured after a regression was observed is
unfalsifiable ("zero new failures" vs a baseline that already ate the failure),
and two sides sharing one source (e.g. a bind mount where source == target)
cannot fail at all. Both shapes produced green verdicts that measured nothing.

These tests pin the CLAUSE, not the wording: they assert the operative
distinction survives a rephrasing, so the discipline cannot be silently dropped
by a future prompt edit.
"""

from __future__ import annotations

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC, prompt_len


def _text() -> str:
    return SYSTEM_PROMPT_STATIC


# ── ────────────────────────────────────────────────────────────

def test_forbidden_item_may_veto_only_known_harm():
    low = _text().lower()
    assert "forbidden" in low
    assert "veto only what is known harmful" in low, (
        "the veto boundary clause is missing: a self-authored prohibition without "
        "a harm test is what killed the measured run")


def test_missing_external_confirmation_is_not_a_reason_to_omit():
    low = _text().lower()
    assert "external confirmation" in low
    # The operative contrast: unavailability of confirmation is NOT sufficient
    # grounds for dropping the correct form.
    assert "never justifies omitting the correct form" in low


def test_veto_boundary_names_the_positive_evidence_route():
    # The clause must say WHAT licenses the override, otherwise it reads as
    # "always override" — the opposite failure.
    low = _text().lower()
    assert "cost analysis" in low and "harmless" in low
    assert "residual uncertainty" in low


# ── ────────────────────────────────────────────────────────────

def test_a_difference_must_name_its_baseline():
    low = _text().lower()
    assert "name the baseline" in low, (
        "baseline-naming clause missing: 'zero new failures' is unfalsifiable "
        "when the baseline was re-measured after the regression")


def test_re_measured_baseline_is_disqualified():
    low = _text().lower()
    assert "re-measured after a regression" in low


def test_shared_source_comparison_is_flagged_as_unable_to_fail():
    low = _text().lower()
    assert "sharing one source cannot fail" in low


# ── budget still holds after both clauses landed ─────────────────────────────

def test_both_clauses_fit_the_budget():
    # The cap was raised 15000 -> 20000 (user ruling 2026-11-01), then
    # 20000 -> 25000 by user ruling, then 25000 -> 31000 -> 32000;
    # the anti-accumulation ratchet now lives in test_c3_total_size_bounded.
    # raised 37000 -> 38000 on 2026-11-06 (four-situations edition, net +740
    # after ~940 chars of compaction); the ratchet itself is test_c3.
    # raised 38000 -> 39000 -> 40000 by USER RULING, to seat the step_done
    # verdict clause (from the 19 replay losses: right number computed, wrong
    # reading shipped). Ratchet unchanged — test_c3 still fails on clause piles.
    assert prompt_len() <= 40_000

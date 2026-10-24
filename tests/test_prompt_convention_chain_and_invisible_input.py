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

"""Anti-decay tests for the two tb2.1-derived clauses (landed 2026-10-24):

A. "Does a provided artifact's own BEHAVIOR decide a convention..." —
   the convention-source chain (task text > provided artifacts >
   discrimination experiment; external standard LOSES; fewest-
   transformations default; full-table alignment before finalizing).
   Root cause evidence: pytorch CLI task lost with a domain-standard
   MNIST normalization chosen over the artifact's own raw pipeline —
   10/50 expected-table entries mismatched while every visible sample
   looked fine under all candidate conventions.

B. "INVISIBLE-INPUT CONTRACT" — the detector-vs-lookup-table obligation
   for deliverables the consumer runs on inputs the agent never sees
   (domain-of-validity statement; per-event signature checks; ordering
   relation). Root cause evidence: video task lost with takeoff=233 on
   an unseen 330-frame test video while the example was calibrated
   perfectly — the shipped answer was example-bound, not a detector.

Each test bans the specific escape phrasing and requires the specific
relation, scoped to its own clause. Mutation gate (2026-10-24): softening
"a DISCRIMINATION EXPERIMENT" to "a DISCUSSION" and "THE DELIVERABLE IS
THE DETECTOR" to "THE DELIVERABLE MAY BE THE EXAMPLE ANSWER" turns 2 of
these tests red; restoring the text restores green.
"""

import pathlib

PROMPT_PATH = pathlib.Path(__file__).resolve().parents[1] / (
    "flagscale_agent/react/prompt.py"
)
PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

CONV = "Does a provided artifact's own BEHAVIOR decide a convention"
INV = "INVISIBLE-INPUT CONTRACT"


def _slice(marker):
    start = PROMPT.index(marker)
    return PROMPT[start : start + 6000]


# ---------------------------------------------------------------- A: chain


def test_chain_names_three_sources_in_order():
    s = _slice(CONV)
    assert "the task's OWN chain, in strict order" in s
    assert "(1) the task text" in s
    assert "(2) the provided artifacts" in s
    assert "(3) a DISCRIMINATION EXPERIMENT" in s


def test_external_standard_explicitly_loses():
    s = _slice(CONV)
    assert "never a domain-standard imported from outside" in s
    assert "not evidence" in s


def test_fewest_transformations_default_is_hard():
    s = _slice(CONV)
    assert "hard default" in s
    assert "FEWEST transformations" in s
    assert "loses by construction" in s


def test_nondiscriminating_sample_proves_nothing():
    s = _slice(CONV)
    assert "NOT SEPARABLE" in s
    assert "that sample proves nothing" in s
    assert "build a discriminating experiment yourself" in s


def test_full_table_alignment_required():
    s = _slice(CONV)
    assert "EVERY entry" in s
    assert "match all of them before you may finalize" in s
    assert "not a near-pass to be shipped" in s


# ------------------------------------------------------- B: invisible input


def test_detector_not_lookup_table():
    s = _slice(INV)
    assert "THE DELIVERABLE IS THE DETECTOR, NOT THE EXAMPLE'S ANSWER" in s
    assert "a lookup table, not a detector" in s


def test_domain_of_validity_stated():
    s = _slice(INV)
    assert "DOMAIN OF VALIDITY" in s
    assert "envelope the numbers carry" in s


def test_multi_event_signature_and_order():
    s = _slice(INV)
    assert "MULTI-EVENT SANITY" in s
    assert "each reported event must independently pass" in s
    assert "a landing follows its takeoff" in s


def test_falsifying_question_is_run_not_inspected():
    s = _slice(INV)
    assert "answer it by RUNNING the synthesized variant, not by inspecting" \
        " the code" in s

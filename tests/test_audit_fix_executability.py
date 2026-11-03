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

"""a FIX line's embedded command must be RUN before it is written.

Measured defect (Audit #87-F3): a FIX line prescribed
`any(k in tb for k in ({...}|{'A1'}))` — a set union inside any()'s iterable,
which is invalid Python. It read fine, so it shipped; it was caught only by
re-running it by hand. A prescribed-but-broken command is worse than no finding:
the parent executes it and reads the failure as its own mistake.

The charter now requires each embedded command to be run once, and an unrun one
to be LABELLED as such. These tests pin the requirement's presence and its
operability (the template must still format and must not disturb the
five-template VIOLATION count the guard's contract test asserts).
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.guard.reviewer_audit import _AUDIT_GOAL


@pytest.fixture
def charter():
    return _AUDIT_GOAL.format(
        snapshot="/s/transcript.jsonl",
        doctrine_source="its first record, role=system",
        output="/s/findings.md",
        continuity="This is audit #1 of an ongoing run.",
    )


def test_charter_requires_running_embedded_commands(charter):
    assert "FIX EXECUTABILITY" in charter
    assert "must have been RUN" in charter
    assert "run each embedded command once" in charter


def test_charter_names_the_failure_mode_it_prevents(charter):
    # The reason must be stated, or the requirement reads as bureaucracy and is
    # the first thing dropped under time pressure.
    # The charter is hard-wrapped at ~78 cols, so a phrase can straddle a
    # newline: normalise whitespace before asserting on it.
    flat = " ".join(charter.split())
    assert "fails silently" in flat
    assert "worse finding than no finding" in flat


def test_unrun_command_must_be_labelled_not_presented_as_runnable(charter):
    assert "unverified: needs the delivered tree" in charter
    assert "rather than presenting it as runnable" in charter


def test_template_still_renders_and_keeps_its_finding_slots(charter):
    # A stray brace in the new prose would break .format() at dispatch time —
    # silently, since the dispatch is wrapped in try/except.
    assert "{snapshot}" not in charter
    assert charter.count("VIOLATION:") == 5
    assert charter.count("EVIDENCE:") == 5
    assert charter.count("FIX:") == 5

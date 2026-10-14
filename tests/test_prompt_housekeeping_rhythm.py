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

"""Regression tests for the Housekeeping Rhythm section of the static prompt.

The section pre-states the recurring guard requirements (memory/plan/knowledge/
context/long-command/search/step-declaration/deliverable) so the agent runs them
on its own cadence instead of being force-reminded by a guard. This is prompt-
only guidance, so the test asserts the section and its load-bearing clauses
survive a future edit, and that it introduces no grader leak and no new
``.format`` placeholder.
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_housekeeping_section_present():
    assert "## Housekeeping Rhythm" in SYSTEM_PROMPT_STATIC
    assert "don't wait to be reminded" in SYSTEM_PROMPT_STATIC.lower()


def _housekeeping_section() -> str:
    """The Housekeeping Rhythm section, sliced from its H2 up to the next H2.

    The assertions below MUST be scoped to this slice: a whole-prompt ``in``
    check is vacuously green — it can pass even if the entire section is deleted
    — so the load-bearing clauses are asserted against this slice, not the whole
    prompt. (Some probes are additionally unique to the section, but uniqueness
    is not relied on.)
    """
    match = re.search(r"## Housekeeping Rhythm.*?(?=\n## )", SYSTEM_PROMPT_STATIC, re.S)
    assert match, "Housekeeping Rhythm section not found"
    return match.group(0).lower()


def test_housekeeping_covers_every_recurring_guard_domain():
    # each domain a recurring guard force-reminds; the section must lead on it.
    # Scoped to the section slice so deleting the section FAILS this test.
    low = _housekeeping_section()
    for concept in (
        "memory",           # MemoryDiscipline
        "knowledge",        # KnowledgeSkill
        "plan",             # PlanUpdate
        "evict",            # ContextPressure (context-pressure relief)
        "background",       # LongTimeShell / ShellJobsWait
        "scope",            # FindGuard (scoped search — token is "scope", not "find")
        "step_done",        # VerificationGuard convention/deliverable_reading
        "deliverable",      # DeliverableSurvival / completion verify
    ):
        assert concept in low, f"Housekeeping section missing {concept!r}"


def test_housekeeping_frames_guard_as_last_resort():
    """A guard firing must be framed as YOUR rhythm failing, not a service."""
    low = _housekeeping_section()
    assert "a guard firing is a failure of your own rhythm" in low
    assert "on your own cadence" in low


def test_housekeeping_pre_states_read_back_before_declaring():
    """The section must lead on the declared-reading == delivered-artifact invariant.

    That invariant is the (B)-class failure the deliverable_reading guard can only
    catch after the fact: a capable agent REASONS the correct output form, then
    ships an artifact that quietly contradicts it (or records a doubt and ships
    anyway). The doctrine belongs here — prompt-level, so the agent acts on its
    own cadence — and is asserted SCOPED to the section slice so deleting it
    fails this test rather than passing on a whole-prompt substring hit.
    """
    low = _housekeeping_section()
    assert "read-back" in low
    assert "declared reading == delivered artifact" in low
    assert "re-read the actual bytes" in low
    assert "unsettled premise" in low
    # Negative binding: assert the doctrine is NOT self-contradicted. A reframe
    # that keeps the tokens but inverts the meaning (e.g. "ship it anyway") must
    # FAIL — token presence alone would pass on inverted doctrine.
    assert "not a licence to ship" in low
    assert "ship it anyway" not in low
    assert "may ship" not in low


def test_housekeeping_leaks_no_grader_or_specifics():
    """GENERAL posture: no task's grader, engine, or construct may be hardcoded."""
    p = SYSTEM_PROMPT_STATIC
    for bad in ("BeautifulSoup", "bs4", "html.parser", "html5lib", "lxml",
                "str(soup)", "selenium", "chrome"):
        assert bad not in p, f"prompt leaked {bad!r}"
    assert not re.search(r"0x[0-9a-fA-F]{4,}", p), "prompt leaked a hex address"


def test_housekeeping_adds_no_new_placeholder():
    """The section must not add a .format placeholder (only cwd/knowledge/skills/tools)."""
    assert set(re.findall(r"{([a-z_]+)}", SYSTEM_PROMPT_STATIC)) == {
        "cwd", "knowledge", "skills", "tools"}


def test_housekeeping_pre_states_canonical_round_trip():
    """The section must lead on canonical-serializer-over-hand-rolled-transform.

    Generic (no engine literal): a hand-rolled lexical transformer that edits raw
    text spans passes in-sample inputs and silently corrupts preserved regions on a
    held-out corpus; a standard parser + canonical re-serializer makes 'untouched
    regions stay unchanged' a structural property. Asserted SCOPED to the section
    slice so deletion fails this test.
    """
    low = _housekeeping_section()
    assert "canonical round-trip" in low
    assert "canonical serializer" in low
    assert "hand-rolled" in low
    # negative binding (F1): the doctrine must forbid the hand-rolled transform, not
    # merely mention it — an inverted rewrite ("prefer a hand-rolled transformer") must
    # fail.
    assert "do not hand-roll a lexical transformer" in low
    assert "do hand-roll" not in low
    assert "prefer a hand-rolled" not in low


def test_housekeeping_pre_states_held_out_corpus():
    """The section must lead on 'your own examples are not a held-out test set'."""
    low = _housekeeping_section()
    assert "held-out test set" in low
    assert "broad" in low
    # negative binding (F1): in-sample-only validation is forbidden, not optional —
    # an inverted rewrite ("in-sample validation is sufficient") must fail.
    assert "in-sample-only validation is not evidence" in low
    assert "in-sample validation is sufficient" not in low

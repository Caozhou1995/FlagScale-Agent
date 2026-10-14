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


def test_housekeeping_covers_every_recurring_guard_domain():
    low = SYSTEM_PROMPT_STATIC.lower()
    # each domain a recurring guard force-reminds; the prompt must lead on it
    for concept in (
        "memory",           # MemoryDiscipline
        "knowledge",        # KnowledgeSkill
        "plan",             # PlanUpdate
        "evict",            # ContextPressure
        "background",       # LongTimeShell / ShellJobsWait
        "scope",            # FindGuard
        "step_done",        # VerificationGuard convention/deliverable_reading
        "deliverable",      # DeliverableSurvival / completion verify
    ):
        assert concept in low, f"Housekeeping section missing {concept!r}"


def test_housekeeping_frames_guard_as_last_resort():
    """A guard firing must be framed as YOUR rhythm failing, not a service."""
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "a guard firing is a failure of your own rhythm" in low
    assert "on your own cadence" in low


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

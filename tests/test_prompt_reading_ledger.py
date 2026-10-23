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

"""Regression tests for the READING-LEDGER discipline (Step B).

Evidence class this locks (filter-js-from-html, six consecutive rounds,
2026-10-23): the task text is genuinely AMBIGUOUS about the output form. Its own
sentence says "Do not alter the formatting ... (except for normalization that
may occur during parsing)", while the verifier compared
`str(BeautifulSoup(clean, 'html.parser'))` against the agent's output -- i.e. it
required the bs4-normalized form. Six rounds were burned because each round
picked ONE reading and rewrote the whole filter.py around it, in a modality
(byte surgery vs parse-and-reserialize) that takes a full run to evaluate:

  R3 built byte surgery (byte-identical on 8/8 of its own cases) -> graded 0.
  R4 built byte surgery again with a "dominance" argument -> 0.
  R5 reasoned about the ambiguity explicitly and still chose byte surgery, and
     reread the clause's own word "dominance" to justify the losing branch.
  R6 used html.parser to REBUILD rather than round-trip -> 0.

What every round lacked was not effort but an ENUMERATION step: list the
defensible readings of the ambiguous requirement FIRST, decide by a dominance
test which one the consumer's own material supports, and when no dominance
exists, cover the top-2 with BOTH implementations in ONE run instead of betting
the whole run on one. That makes ambiguity cost one run instead of six.

These tests lock that discipline and keep it anchored to the EXISTING
PRINCIPLE 4 hedge/bet rule rather than inventing a parallel doctrine.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
from flagscale_agent.react.tools.plan_create import _FRAME_STEP_ACCEPTANCE

LEDGER_CLAUSE = "Reading ledger"


def _frame_ledger_text():
    return "\n".join(i for i in _FRAME_STEP_ACCEPTANCE if "eading ledger" in i)


def test_reading_ledger_present_in_frame_step():
    assert LEDGER_CLAUSE in _frame_ledger_text()


def test_reading_ledger_present_in_static_prompt():
    assert LEDGER_CLAUSE in SYSTEM_PROMPT_STATIC


def test_ledger_requires_enumerating_every_defensible_reading():
    """The ledger's whole point: enumerate FIRST, build second."""
    text = " ".join(_frame_ledger_text().lower().split())
    assert "every defensible reading" in text or "each defensible reading" in text
    # grounded in the task text, not invented from the parser's quirks
    assert "task text" in text or "task's own words" in text


def test_ledger_requires_a_dominance_test_not_a_preference():
    """Choosing a reading by preference is what burned six rounds; the choice
    must be decided by evidence from the consumer's own material."""
    text = " ".join(_frame_ledger_text().lower().split())
    assert "dominance" in text
    for banned in ("pick the one you prefer", "whichever seems right",
                   "your preferred reading"):
        assert banned not in text


def test_ledger_requires_top2_coverage_when_undominated():
    """No dominance -> cover the top-2 in ONE run. This is the discipline that
    turns an ambiguity from a six-run search into a one-run cost.

    The assertion pins the COVERAGE ACTION ('cover the top-2 ... in the SAME
    run'), not the mere presence of the string 'top-2' -- a softened rewrite
    that still mentions top-2 somewhere but tells the agent to pick one reading
    must fail.
    """
    text = " ".join(_frame_ledger_text().lower().split())
    assert "no reading dominates" in text or "undominated" in text
    assert "cover the top-2" in text and "in the same run" in text, (
        "the clause must instruct COVERING the top-2 in the SAME run, not "
        "merely mention top-2"
    )
    # No escape hatch back to per-run betting.
    for banned in ("pick whichever reading", "pick the most likely reading",
                   "choose one reading and ship it"):
        assert banned not in text, f"clause reopened per-run betting: {banned}"


def test_ledger_defers_to_principle4_not_a_new_doctrine():
    """It must reuse PRINCIPLE 4's hedge/bet rule so the two cannot drift."""
    text = _frame_ledger_text()
    assert "PRINCIPLE 4" in text


def test_ledger_clause_is_generic():
    text = _frame_ledger_text() + "\n" + SYSTEM_PROMPT_STATIC
    for banned in ("filter-js", "BeautifulSoup", "html.parser", "clean_html",
                   "byte surgery", "R5"):
        assert banned not in text, f"clause leaked trial-specific text: {banned}"

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

"""Regression tests for the ACCEPTANCE-FACET MANIFEST discipline (Step A).

Evidence class this locks (measured across two runs of the same 13-task set,
2026-10-23): the SAME tasks failed at the SAME assertion both runs
(mcmc -> test_stan_model_sampling, dna-assembly -> test_primers,
filter-js -> test_clean_html_unchanged, video -> test_jump_analyzer_test_video).
Sampling count n was therefore NOT the limiting factor. Reading the mcmc
trajectory (2793 lines, 1800s, wall hit) showed why: the agent spent its depth
enumerating which PRIOR PARAMETER was right (m=2.5, 2.0, 3.0 ... a space the
model itself invented), and never once enumerated WHICH FACES THE CONSUMER
WOULD DRIVE -- it even reasoned "need refresh=0 maybe to avoid huge output",
i.e. it deliberately muted the very surface the graded check reads, while the
prompt already contained the sentence telling it not to. Meanwhile the three
tasks that WON in that batch (mailman / git-multibranch / sam-cell-seg) all
did exactly that enumeration: "the verifier checks the From header", "the test
harness will clone over git@localhost", "the grader checks closed ring, point
count, IoU".

Root cause: every prior mechanism (close-out gate, prompt clause, guard, plan
step) lets the MODEL decide what is worth checking, so the enumeration set is
unbounded and drifts. The fix is an EXTERNALLY ANCHORED set: the row count is
fixed by the number of verifiable clauses in the TASK TEXT, which the model
cannot silently shrink. These tests lock the two mechanical properties
(external anchoring, count = clause count) so a future edit cannot soften the
clause back into "be careful".

They also lock that it stays GENERIC -- no trial identifier, no threshold
copied out of a graded instance.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
from flagscale_agent.react.tools.plan_create import _FRAME_STEP_ACCEPTANCE

ACCEPTANCE_FACET_CLAUSE = "Acceptance-facet manifest"


def _frame_facet_text():
    """Concatenate the FRAME acceptance items that talk about the manifest."""
    return "\n".join(i for i in _FRAME_STEP_ACCEPTANCE if "acet" in i)


def test_acceptance_facet_clause_present_in_frame_step():
    """The FRAME fixed step must carry the acceptance-facet enumeration duty."""
    assert ACCEPTANCE_FACET_CLAUSE in _frame_facet_text()


def test_acceptance_facet_clause_present_in_static_prompt():
    """The static prompt's FRAME line must name it too, so it survives a
    plan-tool bypass (agents that skip plan_create still see the discipline)."""
    assert ACCEPTANCE_FACET_CLAUSE in SYSTEM_PROMPT_STATIC


def test_facet_set_is_externally_anchored():
    """THE load-bearing property: the number of facets is fixed by the task
    text's own clauses, not by what the model happens to think of.

    Without this, the clause degrades into the same unbounded "think of what
    might be checked" advice that four earlier mechanisms already failed with.
    The assertion must therefore pin the ANCHORING RELATION, not merely the
    words 'clause' and 'row count' -- a softened rewrite that keeps those words
    but drops the relation ("as many as you can think of") must fail.
    """
    text = " ".join(_frame_facet_text().lower().split())
    # (a) the anchoring relation, stated as a count fixed BY the clause count
    assert "fixed by the clause count" in text, (
        "the clause must state that the row count is FIXED BY the clause count"
    )
    # (b) one row per clause -- the shape that makes the count checkable
    assert "one row per verifiable clause" in text
    # (c) no escape hatch back to unbounded model choice
    for banned in ("as many as you can think", "whatever feels",
                   "list as many", "as you see fit", "use your judgment"):
        assert banned not in text, f"clause reopened unbounded choice: {banned}"


def test_every_facet_requires_a_concrete_probe():
    """A facet with no executed probe is a sentence, not a check -- the mcmc
    agent named the console surface and then muted it. Each row must carry the
    exact command/input that exercises that face, and its observed result."""
    text = _frame_facet_text()
    assert "probe" in text.lower()
    assert "observed" in text.lower() or "result" in text.lower()


def test_facet_clause_is_generic():
    """No trial identifier or magic value may leak into the clause."""
    text = _frame_facet_text() + "\n" + SYSTEM_PROMPT_STATIC
    for banned in ("mcmc", "stan_model_sampling", "filter-js", "dna-assembly",
                   "refresh=0", "2793", "1800s"):
        assert banned not in text, f"clause leaked trial-specific text: {banned}"

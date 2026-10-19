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

"""Regression tests for the MEASURED-QUANTITY PROVENANCE discipline.

Evidence class this locks (two independent trials of the same 4-task set):
  * dna-insert: the agent's self-check defined the "annealed part" by
    template anchoring (15/16/17 nt, PASS); the grader's construction
    (primers_concat[insert_end:]) yields 56 nt, so `assert 56 <= 45` failed.
  * mteb-retrieve: the agent computed the mteb instruction-variant answer
    (MTEB) and knowingly delivered the no-instruction variant (HumanEval),
    though the task NAMES mteb, whose own default path applies the prompt.

Both are SELF-ORACLE DEFINITION MISMATCHES: the agent's evidence is
internally consistent, so no guard can see it — only the agent's own
discipline can. That is why the load-bearing fix is prompt text.

These tests assert the clauses are present in the STATIC prompt (a future
edit must not silently drop them) and that they stay generic (no trial
specifics leaked in).
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _prompt_low() -> str:
    return " ".join(SYSTEM_PROMPT_STATIC.split()).lower()


def test_prompt_requires_same_quantity_as_grader():
    """The self-check must be required to measure the same quantity the grader does."""
    low = _prompt_low()
    assert "am i measuring the same quantity the grader measures" in low
    # the falsifier: both definitions must be run and diffed, not argued about
    assert "run both and diff the numbers" in low
    # and the trap it closes: a passing self-oracle is not evidence of agreement
    assert "self-check that passes while the grader fails" in low


def test_prompt_requires_named_tool_default_path():
    """When the task NAMES a tool, that tool's own default path defines the value."""
    low = _prompt_low()
    assert "which path produced the value the task asked for" in low
    assert "named path wins" in low
    # must forbid the observed behaviour: delivering the simplified emulation
    assert "simplified emulation" in low
    assert "do not deliver the other one" in low


def test_prompt_requires_obtaining_the_real_grader():
    """A reachable grader beats any reconstruction of it; state the fallback order."""
    low = _prompt_low()
    assert "can i obtain the grader instead of guessing it" in low
    # must name the concrete artifacts to look for, not just "look for it"
    assert "tests/ directory" in low and "official solution script" in low
    # must state the ORDER: real grader first, diff-two-definitions only as fallback
    assert "only when no such artifact is reachable" in low


def test_quantity_clauses_stay_generic():
    """No originating-task specifics may leak into the generic prompt."""
    low = _prompt_low()
    for leak in ("dna-insert", "mteb-retrieve", "primers_concat", "humaneval", "bge-small"):
        assert leak not in low, f"prompt leaked task-specific identifier: {leak}"
    assert not re.search(r"0x[0-9a-f]{4,}", low)


def test_prompt_requires_cosource_check_before_trusting_a_differential():
    """Two implementations that AGREE can be co-source and both wrong.

    Real evidence: m19/m20 mteb-retrieve agent ran mteb-wrapper(no prompt) and
    raw sentence-transformers(no prompt), got identical 30 cosines and rank5,
    called it a "differential oracle confirming", and shipped the wrong line.
    Both sides were the BARE encode path; the named tool's retrieval evaluator
    (prompt_type=query/passage) was never exercised.
    """
    low = _prompt_low()

    assert "co-source" in low
    # the co-source warning must say agreement is not proof
    i = low.index("co-source")
    window = low[i:i + 500]
    assert "same wrong answer" in window
    # and the named tool's high-level task entry point must be named as THE path
    assert "task entry point" in low
    assert "bare encode" in low


def test_prompt_requires_grader_criterion_expression_not_metric_substitute():
    """A self-check must restate the grader's exact criterion expression.
    Substituting cosine similarity for an elementwise ratio-constancy test is a
    DIFFERENT quantity: |cos|=0.999997 coexists with per-element ratio spread
    1.8e-2 against a 1e-4 tolerance (m21 model-extraction, 28/30 rows failed).
    Removing this clause must break the assertion."""
    text = SYSTEM_PROMPT_STATIC
    assert "restate its exact expression" in text, (
        "prompt must require restating the grader criterion's exact expression")
    assert "a different metric is a different question" in text, (
        "prompt must forbid substituting a different similarity metric")

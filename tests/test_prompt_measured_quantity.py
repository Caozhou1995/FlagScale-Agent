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

Evidence class this locks (from trials of a multi-task set, described generically):
  * a design task where the agent's self-check defined the measured segment by
    ANCHORING ON ITS OWN TEMPLATE (a short length, PASS) while the check re-derived
    that segment from an assembled construct (a much longer length, FAIL);
  * a retrieval task where the agent computed the named library's task-typed answer
    and knowingly delivered the library's bare-primitive answer instead, though the
    task NAMES that library, whose own registry defines the task-typed parameters.

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


def test_prompt_requires_same_quantity_as_the_check():
    """The self-check must be required to measure the same quantity the check does."""
    low = _prompt_low()
    assert "am i measuring the same quantity the check measures" in low
    # the falsifier: both definitions must be run and diffed, not argued about
    assert "run both and diff the numbers" in low
    # and the trap it closes: a passing self-oracle is not evidence of agreement
    assert "self-check that passes while the real check fails" in low


def test_prompt_requires_named_tool_default_path():
    """When the task NAMES a tool, that tool's own default path defines the value."""
    low = _prompt_low()
    assert "which path produced the value the task asked for" in low
    assert "named path wins" in low
    # must forbid the observed behaviour: delivering the simplified emulation
    assert "simplified emulation" in low
    assert "do not deliver the other one" in low


def test_prompt_forbids_obtaining_the_hidden_checker():
    """Build the acceptance check from the specification; never fetch the checker."""
    low = _prompt_low()
    # the boundary must be stated as a prohibition, in these words
    assert "or from the checker" in low
    assert "transcription test" in low
    assert "do not fetch" in low
    # and the legitimate substitute must be named
    assert "from the specification" in low
    # the old, leak-inviting directive must be GONE
    assert "can i obtain the grader instead of guessing it" not in low
    assert "official solution script" not in low


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

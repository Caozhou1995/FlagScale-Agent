"""Anti-decay: the decisive-move clause must be TWO-SIDED and
consumer-relative, not "bytes must not move".

History (why this file was rewritten):

v1 (commit 806b532) said: name a must-survive input and a must-change input,
run both, and "a must-survive input that moved is a FAILURE ... re-serializing
rewrites order, quoting, self-closing form and whitespace". Measured
(job 2026-11-20__10-59-35, filter-js-from-html): the trial obeying that clause
refused parser normalization, emitted original bytes, and failed with
"Filter modified 5 clean files out of 12".

Then the checker was read (tests/test_outputs.py::test_clean_html_unchanged).
It is ASYMMETRIC: expected = str(BeautifulSoup(original, "html.parser")) with
spaces/newlines stripped; actual = the raw output text with spaces/newlines
stripped. So the fixed point of "clean input unchanged" IS the normalizer's
output, and byte-identity FAILS. Verified on the checker's own 12 samples:
  - identity output        -> fails on samples {4,6,9,11,12}  = 5/12
  - str(BeautifulSoup(...)) -> fails 0/12
And across 110 historical trials of this task, all 13 reward=1.0 trials used the
parser's own serializer as their output; the trials that only produced original
bytes never won the clean-file assertion.

So the v1 wording named a real hazard but located the failure on the wrong
side: "survive" is defined by HOW THE CONSUMER COMPARES, not by "no bytes moved".
That is the class-generic lesson (no parser name, no task vocabulary), and this
file locks it.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _clause() -> str:
    start = SYSTEM_PROMPT_STATIC.index("(a2) THE DECISIVE MOVE")
    end = SYSTEM_PROMPT_STATIC.index("(b) THE THREE QUESTIONS")
    return " ".join(SYSTEM_PROMPT_STATIC[start:end].split())


def test_decisive_move_is_two_sided():
    """Both directions must be named on the same clause."""
    s = _clause()
    assert "must survive" in s
    assert "must CHANGE" in s
    assert "read BOTH" in s


def test_survive_is_judged_by_the_consumer_not_by_byte_identity():
    """The measured trap: byte-identity fails when the consumer normalizes."""
    s = _clause()
    assert "Judge 'survive' by how the CONSUMER compares" in s
    assert "leaving bytes untouched can FAIL" in s


def test_clause_forbids_refusing_the_normalizer():
    """The action, not the principle: reproduce the normalizer's output."""
    s = _clause()
    assert "reproducing that normalizer's output is what passes" in s
    assert "reproduce it, do not refuse it" in s


def test_clause_does_not_assert_that_reserializing_is_the_error():
    """v1's inverted claim must not come back: the data refuted it."""
    s = _clause().lower()
    assert "re-serializing rewrites order" not in s


def test_clause_stays_class_generic():
    """No task literals and no library names: the example must not leak."""
    s = _clause().lower()
    for leak in ("html", "attribute", "filter", "beautifulsoup", "bs4",
                 "parser name", "clamp", "primer", "dna", "raman"):
        assert not re.search(r"\b" + re.escape(leak), s), f"leaked task token: {leak}"

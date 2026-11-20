"""Anti-decay: the decisive-move clause must be TWO-SIDED.

Measured (job 2026-11-20__10-59-35, filter-js-from-html): reward=0.0 with the
single failing assertion `test_clean_html_unchanged` — "Filter modified 5 clean
HTML files out of 12". The trial DID run a variant check (it built a held-out
file where removals were expected, its "must-change" direction) and DID read the
result back; it never tested the opposite direction — an input the task says
must come through BYTE-IDENTICAL, where ANY difference is a failure. Its own
deliberate parse-and-re-serialize path rewrote attribute order, quoting and
self-closing form on 5 of 12 clean files and nothing in the loop asked it to
check that direction.

So the clause needs both halves on the same line: a must-survive input and a
must-change input, with "must-survive moved" being a FAILURE in its own right.
Class-generic wording only — zero task vocabulary.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _clause() -> str:
    start = SYSTEM_PROMPT_STATIC.index("(a2) THE DECISIVE MOVE")
    end = SYSTEM_PROMPT_STATIC.index("(b) THE THREE QUESTIONS")
    return " ".join(SYSTEM_PROMPT_STATIC[start:end].split())


def test_decisive_move_is_two_sided():
    """Both directions must be named on the same clause, not just the changing one."""
    s = _clause()
    assert "must survive UNCHANGED" in s
    assert "must CHANGE" in s
    assert "read BOTH" in s


def test_must_survive_failure_is_not_waived_by_the_other_half():
    """The measured failure mode: the must-change half passed, so the trial shipped."""
    s = _clause()
    assert "is a FAILURE though the must-change one worked" in s


def test_clause_names_why_a_byte_identical_input_can_move():
    """The mechanism, so the rule is checkable rather than a slogan."""
    s = _clause()
    assert "re-serializing rewrites order, quoting, self-closing form and whitespace" in s


def test_clause_stays_class_generic():
    """No task literals: the example that produced this rule must not leak."""
    s = _clause().lower()
    for leak in ("html", "attribute", "filter", "clamp", "primer", "dna", "raman"):
        assert not re.search(r"\b" + re.escape(leak), s), f"leaked task token: {leak}"

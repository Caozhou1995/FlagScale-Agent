"""Pin the span-disagreement adjudication clause in §3(b).

Failure it encodes (dna-assembly, 2026-08): the agent computed a quantity under
TWO spans, got 73.64 under the consumer's (matched) span and the target value
under its own. Instead of treating the disagreement as a defect in the
artifact, it reclassified the failing reading as its own arithmetic slip and
re-derived under the passing one — so the delivered number came from the wrong
span and the run scored zero.

The prompt already told it to compute under BOTH spans. What it did not say is
what the DISAGREEMENT means. That is this clause.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as PROMPT

FLAT = re.sub(r"\s+", " ", PROMPT)


def _slice() -> str:
    start = FLAT.find("Where that parameter is not yours to choose")
    end = FLAT.find("(b2) A REJECTED READING")
    assert start != -1 and end != -1 and start < end
    return FLAT[start:end]


def test_disagreement_is_a_defect_not_a_slip():
    """The ruling: a disagreement between the two spans indicts the artifact."""
    s = _slice()
    assert "When they disagree, treat it as a REAL defect" in s


def test_reclassifying_the_failing_reading_is_forbidden():
    """The exact wrong move this run made, named as forbidden."""
    s = _slice()
    assert "never call the failing reading your own arithmetic slip" in s
    assert "re-derive under the passing one" in s


def test_the_failing_reading_is_the_consumers_number():
    """Why it fails: a value that moves with the span belongs to the span the
    consumer measures, so the one that moves is the one that counts."""
    s = _slice()
    assert "a value that moves only with the span is the consumer's number" in s
    assert "and it fails" in s


def test_clause_attaches_to_the_both_spans_sentence_it_qualifies():
    """It must sit exactly where the both-span instruction is given, not float
    elsewhere in the prompt."""
    i = FLAT.find("compute the quantity under BOTH spans")
    j = FLAT.find("When they disagree, treat it as a REAL defect")
    assert i != -1 and j != -1
    assert 0 < j - i < 200, "the adjudication must follow the both-span instruction"


def test_clause_stays_task_agnostic():
    """No benchmark vocabulary: this is a general rule, not this task's."""
    banned = [
        "dna", "bsai", "ggtctc", "primer", "overhang", "anneal",
        "oligotm", "primer3", "plasmid", "egfp", "clamp", "fasta",
        "terminal-bench", "dna-assembly", "73.6",
    ]
    low = _slice().lower()
    for word in banned:
        assert not re.search(r"\b" + re.escape(word) + r"\b", low), (
            f"clause leaked task vocabulary: {word}"
        )

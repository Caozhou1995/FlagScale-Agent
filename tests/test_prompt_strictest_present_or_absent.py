"""Pin E9's `"Strictest"` definition to cover PRESENT-OR-ABSENT, not only multi-value.

Measured gap this test was written from (2026-11-28, dna-assembly trial):
the prompt's two existing "strictest" clauses both read as *which of several
values to pick* ("then adopt the strictest"; "which value ships").  An agent
facing a SINGLE-VALUE element (does the primer carry a clamp base at all?) had
no clause that owned that shape, so it re-read "strictest" as "the standard's
minimal form", emitted clamp=0 for all 8 primers, and died on the verifier's
`Primer must have clamp of at least 1 nucleotide before BsaI site`.

The clause under test is the mapping inserted into the existing Strictest
sentence -- it is a one-line addition, not a new paragraph, because the prompt
is at budget (51700/51700).
"""
import re

from flagscale_agent.react import prompt as prompt_mod


def _static() -> str:
    return prompt_mod.SYSTEM_PROMPT_STATIC


def test_present_or_absent_is_mapped_onto_strictest():
    """The mapping exists, exactly once, so it cannot silently decay."""
    text = _static()
    assert text.count("INCLUDED is strictest") == 1


def test_mapping_names_present_or_absent_verbatim():
    """The element-vs-value shape is named in the prompt's own words."""
    assert "present-or-absent" in _static()


def test_mapping_sits_inside_the_strictest_sentence():
    """Not a loose sentence elsewhere: it must be bound to the Strictest rule.

    Locating it by adjacency (same paragraph, between the Strictest opening and
    the `satisfy TOGETHER` close) is what makes it an INTERPRETATION of
    `"Strictest"` rather than an unrelated aside.
    """
    text = _static()
    start = text.find('"Strictest" = the WORST case')
    end = text.find("every reading that complies, satisfy TOGETHER", start)
    assert start != -1 and end != -1, "the Strictest sentence itself must exist"
    assert start < end, "Strictest sentence order must be preserved"
    window = text[start:end]
    assert "INCLUDED is strictest" in window, (
        "the present-or-absent mapping must live inside the Strictest sentence"
    )


def test_mapping_does_not_reintroduce_a_minimal_form_reading():
    """The sentence must keep forbidding the minimal-form reading.

    This is the positive bound: the mapping is an ADDITION, so the clause it
    qualifies (`not the minimal sample-clearing form`) must survive verbatim.
    """
    text = _static()
    assert "not the minimal sample-clearing form" in text


def test_mapping_keeps_the_satisfy_together_obligation():
    """Both readings must still be satisfiable together."""
    assert "every reading that complies, satisfy TOGETHER" in _static()


def test_prompt_still_within_authorised_budget():
    """The addition is one short clause; it must not blow the budget."""
    assert prompt_mod.prompt_len() <= prompt_mod.PROMPT_BUDGET


def test_no_task_specific_nouns_leaked_into_the_prompt():
    """prompt.py is scanned wholesale, so no task vocabulary may enter it.

    The failure this guards (learned the hard way): test fixtures grep the whole
    SOURCE file, not just the string literals, so a code comment naming a task
    trips the leak scan.
    """
    src = open(prompt_mod.__file__, encoding="utf-8").read()
    for noun in ("dna-assembly", "gpt2-codegolf", "gcode-to-text", "BsaI", "primer"):
        assert noun.lower() not in src.lower(), f"task vocabulary leaked: {noun}"

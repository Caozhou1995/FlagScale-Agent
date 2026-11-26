"""Ablation-pinned tests for the "text is a FLOOR" axis clause in _BLINDSPOT_PROMPT.

Grounded in a measured A/B ablation (2026-11-24, real dna task text, production
config 65536/16384, real TaskStatementReadersTool.read_blindspots path, arms
differing by 137 chars = this clause alone):

  ARM_A (with clause):    7 lines, includes "5' flanking bases before the BsaI
                          recognition site: carries them vs omits them" (2 lines
                          on the 5'-flanking axis)
  ARM_B (clause removed): 9 lines, 5'-flanking axis = 0 lines

The control axis (spacer, which the task text itself neighbours) appeared in
BOTH arms (2 lines each) — so the difference is not "ARM_B produced less", it is
selective appearance of the presence-type axis. That is what makes this clause
causal rather than incidental.
"""
import re

from flagscale_agent.react.tools.read_task_statement import _BLINDSPOT_PROMPT


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def test_axis_clause_is_present_in_the_rendered_definition():
    """The clause must live in the RENDERED prompt, not only in a source constant."""
    rendered = _BLINDSPOT_PROMPT.format(sentinel="S", task_text="T")
    assert "the field's standard demands what" in _flat(rendered)


def test_text_is_explicitly_a_floor_not_a_ceiling():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "the text is a FLOOR" in flat
    assert "no practitioner would accept it otherwise" in flat


def test_axis_belongs_to_not_pinned_so_it_enumerates_rather_than_guesses():
    """It must sit inside definition (A) NOT PINNED, not in the out-of-scope list."""
    flat = _flat(_BLINDSPOT_PROMPT)
    a = flat.find("(A) NOT PINNED")
    clause = flat.find("the field's standard demands what")
    out_of_scope = flat.find("DO NOT report")
    assert a != -1 and clause != -1 and out_of_scope != -1
    assert a < clause < out_of_scope, "axis clause must be inside (A), before DO NOT report"


def test_axis_does_not_name_any_particular_domain_or_artifact():
    """Generality guard: the clause must not hard-code a benchmark's field.

    A clause that says "NEB", "BsaI", "primer", "FASTA" would be case-by-case
    patching — exactly what this mechanism is meant to replace.
    """
    flat = _flat(_BLINDSPOT_PROMPT).lower()
    for banned in ("neb", "bsai", "primer", "fasta", "golden gate", "spacer"):
        assert banned not in flat, f"axis prompt must stay domain-agnostic, found {banned!r}"


def test_presence_form_is_still_available_for_standard_derived_elements():
    """The settlement must be expressible as carried-vs-omitted (a presence axis)."""
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "carries it vs omits it" in flat or "carries them vs omits them" in flat


def test_deleting_the_clause_removes_the_floor_semantics():
    """Ablation: with the clause spliced out, no floor language survives.

    Mirrors the measured ARM_B construction exactly (remove from the preceding
    comma through the clause's closing period).
    """
    START, END = "OR the field's standard demands", "otherwise)."
    i = _BLINDSPOT_PROMPT.find(START)
    j = _BLINDSPOT_PROMPT.find(END, i) + len(END)
    before = _BLINDSPOT_PROMPT.rfind(",", 0, i)
    assert 0 < before < i < j
    arm_b = _BLINDSPOT_PROMPT[:before] + "." + _BLINDSPOT_PROMPT[j:]
    flat_b = _flat(arm_b)
    # NB: the prompt carries an UNRELATED floor concept elsewhere (the
    # set-membership ceiling-vs-floor reading of a named list), so the
    # ablation must be asserted on THIS clause's own words, not on "FLOOR".
    assert "the field's standard" not in flat_b
    assert "no practitioner would accept it otherwise" not in flat_b
    assert len(arm_b) == len(_BLINDSPOT_PROMPT) - (j - before) + 1
    # and the clause IS present before ablation, so the removal is meaningful
    assert "the field's standard" in _flat(_BLINDSPOT_PROMPT)

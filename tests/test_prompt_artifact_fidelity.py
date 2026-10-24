"""Anti-attenuation tests for the artifact-fidelity extension of the
VERBATIM-beats-PROPERTY clause (protein tag-loss root cause, zZzfXM6).

The clause extension pins three behaviors the protein-assembly trial
violated: (1) the checker's expected value is derived FROM the handed
artifact and therefore carries its built-in tag; (2) an unexpressible symbol
(X) is a hint about checker tolerance, not a licence to swap source records;
(3) deliver the artifact's own form with only the transformation the task
text explicitly names.
"""
import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _clause_text():
    for line in SYSTEM_PROMPT_STATIC.splitlines():
        if "A VERBATIM CONSTRAINT BEATS A PROPERTY CONSTRAINT" in line:
            return line
    return ""


def test_artifact_fidelity_extension_present():
    text = _clause_text()
    assert "DECIDING WHICH TEXT IS VERBATIM-GRADED" in text


def test_extension_pins_tag_kept_from_handed_artifact():
    text = _clause_text()
    # the checker was written against the raw artifact, not a tidied one
    assert "carries the artifact's own incidental structure" in text
    assert "a built-in affinity/purification tag" in text


def test_extension_pins_unexpressible_symbol_as_tolerance_hint():
    text = _clause_text()
    assert "X, B, Z, U" in text
    assert "NOT a licence to swap in a different source record" in text


def test_extension_pins_symbol_resolution_from_semantic_source():
    """Third-layer root cause (zu6rmoT): the agent resolved X from the
    artifact's in-file SEQADV annotation; the checker builds its expected
    value from the semantic source (the database record the pipeline reads)."""
    text = _clause_text()
    assert "RESOLVING AN AMBIGUOUS SYMBOL" in text
    assert "resolve the symbol from the SEMANTIC SOURCE the pipeline reads" in text
    assert "an in-file annotation is a provenance note" in text


def test_extension_pins_explicit_transformation_only():
    text = _clause_text()
    assert "ONLY the transformation the task text explicitly names" in text
    assert "never a transformation you inferred from plausibility" in text


@pytest.mark.parametrize("soft,hard", [
    ("may usually carry the artifact's own structure", "carries the artifact's own incidental structure"),
    ("consider whether the checker tolerates the symbol", "NOT a licence to swap in a different source record"),
])
def test_mutation_softening_detected(soft, hard):
    """Mutation gate: softening the load-bearing phrases must break the test."""
    assert hard not in soft
    assert hard in _clause_text() or soft == hard

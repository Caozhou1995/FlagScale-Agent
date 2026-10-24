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
    """Third-layer root cause (zu6rmoT): disambiguation is the agent's job in
    single-shot mode — the clause must pin the PURPOSE test (raw input to a
    downstream transformation -> upstream semantic source), plus the
    fetch-and-diff probe. The clause itself must stay GENERAL: no entity
    literal from the concrete task may appear in it."""
    text = _clause_text()
    assert "DISAMBIGUATE by asking what the deliverable is FOR" in text
    assert "raw INPUT to a downstream transformation" in text
    assert "resolve from THAT" in text
    assert "fetch its record yourself and diff it against the handed artifact" in text


@pytest.mark.parametrize("literal", [
    "GYG", "SYG", "5WJ2", "Clover", "fpbase", "FPbase", "SEQADV", "MODRES",
    "FASTA", "fasta", "chromophore", "His-tag",
])
def test_symbol_resolution_clause_is_general_no_task_literals(literal):
    """Design principle: prompt clauses are class-level rules, never task-
    specific answers. The protein run's entities must not leak into the
    clause (anti-overfit hard-code gate)."""
    assert literal not in _clause_text()


def test_symbol_resolution_clause_determinism():
    """Stability principle: the decision chain must be a total order with a
    unique standard at every node — no branch left undefined, no
    'either/may/consider' freedom at a decision point."""
    text = _clause_text()
    # every ambiguity case lands in exactly one of two mutually exclusive branches
    assert "Exactly one reading is graded" in text
    assert "Both readings never apply to one deliverable" in text
    # the fallback for an undecided case is itself deterministic
    assert "read the task's own verb" in text
    assert "only the task text can move a case between branches" in text
    # the fetch-and-diff probe has a deterministic tie-break, not a judgement call
    assert "prefer the branch whose source the task text itself names" in text
    # no permissive hedges at decision nodes
    for hedge in ("you may", "consider resolving", "either reading is fine"):
        assert hedge not in text


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

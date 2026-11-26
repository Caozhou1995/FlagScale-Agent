"""A judged quantity listed without its BOUND is only half-listed.

Measured on the real dna trial ZyjwKkQ (2026-11-25 22:47, standard 30 min):
the agent DID enumerate the judged quantities (the clamp/5'-flank axis was
fixed -- 8/8 primers shipped `aattGGTCTCa...`, ggtctc_idx=4, the verifier got
past `assert i >= 1` for the first time in four runs).  It then failed on the
NEXT axis with a pure out-of-range error:

    AssertionError: Reverse primer annealing (incl. overhang match) must be 15-45 nt.
    assert 48 <= 45
     +  where 48 = len('ggtctagaaataattttgtttaactttaagaaggagatatacatatga')

So the gap was not "did not list length" -- `length` is already in the
enumeration list.  The gap is that the LISTED kind was never paired with the
task's stated RANGE, so nothing in the delivery check could turn red on 48>45.
The clause below names that second half of the action.
"""
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
import re

FLAT = re.sub(r"\s+", " ", SYSTEM_PROMPT_STATIC).strip()

CLAUSE = "one listed without its BOUND is half-listed"
ACTION = "transcribe each task-stated range beside it and test the delivered value against it"


def test_bound_half_of_the_enumeration_is_stated():
    """The clause must exist verbatim, next to the enumeration it completes."""
    assert CLAUSE in FLAT
    assert ACTION in FLAT


def test_bound_clause_lives_inside_the_single_value_enumeration():
    """It must sit in THE SINGLE VALUE section, right after FIRST LIST THE
    JUDGED QUANTITIES -- a bound rule placed anywhere else is not read at the
    moment the list is made."""
    head = FLAT.index("FIRST LIST THE JUDGED QUANTITIES")
    body = FLAT[head:head + 700]
    assert CLAUSE in body, "bound rule drifted away from the enumeration it completes"
    assert ACTION in body


def test_bound_clause_names_the_failure_mode_not_the_answer():
    """Abstract only: a concrete number/kind pair would BE the answer and let an
    agent pattern-match instead of transcribing its own task's bounds."""
    seg = FLAT[FLAT.index(CLAUSE) - 220: FLAT.index(CLAUSE) + 260].lower()
    for leaked in ("clamp", "ggtctc", "bsai", "neb", "golden gate", "primer",
                   "raman", "fasttext", "sam-cell-seg", "gcode", "doom",
                   "15", "45", "48", "nt)"):
        assert leaked not in seg, f"benchmark literal {leaked!r} leaked into the bound clause"


def test_enumeration_kinds_still_intact():
    """Positive control: adding the BOUND half must not drop the KIND half."""
    head = FLAT[FLAT.index("FIRST LIST THE JUDGED QUANTITIES"):][:700]
    for kind in ("field", "flag", "length", "position", "offset", "count",
                 "present-or-absent"):
        assert kind in head, f"enumeration kind {kind!r} lost"

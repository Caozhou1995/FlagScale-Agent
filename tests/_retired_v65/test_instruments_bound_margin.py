"""Bound-margin instrument: a pass is only decided when the margin beats the
smallest defensible change of the quantity's own definition.

Measured loss this closes (terminal-bench dna-assembly, two consecutive runs):
the agent measured a reverse-primer annealing length of 44 against a stated
upper bound of 45 -- a 1 nt margin -- while the grader's construction of the
same quantity (binding + the overhang's match on the template) produced 48.
Two defensible definitions of one quantity, 4 nt apart; the agent's self-check
reported PASS on the definition that happened to fit under the bound.

The mechanism is deliberately NOT another self-narrative field whose
`within_tolerance: yes` the harness must trust. The agent supplies two numbers
(its value, the stated bound) plus the shift a rival definition causes, and the
HARNESS DOES THE ARITHMETIC: if |value - bound| is smaller than that shift, the
step is refused as UNDECIDED and both numbers are echoed back.
"""

import pytest

from flagscale_agent.react.instruments import (
    check_bound_margin,
    bound_margin_decided,
)


# --------------------------------------------------------------- the arithmetic
def test_margin_larger_than_rival_shift_is_decided():
    """Plenty of room: a rival definition could not push it over the bound."""
    bm = {"value": 30, "bound": 45, "direction": "max",
          "rival_shift": 4, "rival_definition": "includes the overhang match"}
    assert check_bound_margin(bm) is None
    assert bound_margin_decided(bm) is True


def test_margin_equal_to_rival_shift_is_undecided():
    """Borderline: |41-45| == 4 means the rival definition lands exactly on the
    bound. Equality is NOT a pass -- ties go to the refusal."""
    bm = {"value": 41, "bound": 45, "direction": "max",
          "rival_shift": 4, "rival_definition": "includes the overhang match"}
    v = check_bound_margin(bm)
    assert v is not None and v[0] == "bound_margin_too_tight"
    assert bound_margin_decided(bm) is False


def test_margin_smaller_than_rival_shift_is_undecided():
    """THE MEASURED CASE: value 44 vs bound 45 (margin 1) with a 4 nt rival
    shift -- the dna-assembly shape. Must be refused, not passed."""
    bm = {"value": 44, "bound": 45, "direction": "max",
          "rival_shift": 4,
          "rival_definition": "annealing tract = binding + overhang suffix "
                              "matched on the template"}
    v = check_bound_margin(bm)
    assert v is not None and v[0] == "bound_margin_too_tight"
    msg = v[1]
    # The refusal must echo BOTH numbers so the next attempt can self-correct.
    assert "44" in msg and "45" in msg
    assert bound_margin_decided(bm) is False


def test_lower_bound_direction_is_read_the_same_way():
    """A minimum bound is not special-cased: margin is |value - bound|."""
    tight = {"value": 16, "bound": 15, "direction": "min",
             "rival_shift": 4, "rival_definition": "excludes the 5' spacer"}
    assert check_bound_margin(tight)[0] == "bound_margin_too_tight"
    roomy = {"value": 30, "bound": 15, "direction": "min",
             "rival_shift": 4, "rival_definition": "excludes the 5' spacer"}
    assert check_bound_margin(roomy) is None


# ------------------------------------------------------------ malformed input
@pytest.mark.parametrize("bad,reason", [
    ("not-a-dict", "bound_margin_not_dict"),
    ({"bound": 45}, "bound_margin_field_missing"),
    ({"value": 44, "bound": 45, "direction": "max",
      "rival_shift": "big", "rival_definition": "x"},
     "bound_margin_bad_shift"),
    ({"value": 44, "bound": 45, "direction": "max",
      "rival_shift": -4, "rival_definition": "x"},
     "bound_margin_bad_shift"),
    ({"value": 44, "bound": 45, "direction": "sideways",
      "rival_shift": 4, "rival_definition": "x"},
     "bound_margin_bad_direction"),
])
def test_malformed_declarations_are_refused(bad, reason):
    v = check_bound_margin(bad)
    assert v is not None and v[0] == reason
    assert bound_margin_decided(bad) is False


def test_absent_declaration_is_not_a_pass():
    assert check_bound_margin(None)[0] == "bound_margin_missing"
    assert bound_margin_decided(None) is False


# --------------------------------------------------------- stated rival shift
def test_rival_definition_must_be_named():
    """A numeric shift with no named rival definition is a bare number: the
    agent must say WHAT would produce it, or the shift is unauditable."""
    bm = {"value": 44, "bound": 45, "direction": "max", "rival_shift": 4,
          "rival_definition": ""}
    v = check_bound_margin(bm)
    assert v is not None and v[0] == "bound_margin_blank_rival"

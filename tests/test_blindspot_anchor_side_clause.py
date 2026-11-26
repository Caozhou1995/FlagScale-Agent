"""Ablatable gate for the region reader's ANCHOR AND SIDE requirement.

Why this file exists (nothing else pins it):
  tests/test_blindspot_axis_clause.py pins the FLOOR / "field's standard" clause
  and the "carries it vs omits it" wording, but a measured ablation showed it
  stays GREEN when the ANCHOR AND SIDE rule is deleted. Without a gate, an edit
  that softens that rule passes every test while silently degrading the region
  output to BETWEEN-anchor points (spacer / overhang) — which is exactly the
  off-axis failure observed on trial yrH2Y22 (region named only "carries
  overhang vs omits", 5' side 0 lines, delivered site_at=0/8).

The rule being pinned (read_task_statement.py, _BLINDSPOT_PROMPT HARD RULES):
  a presence point must name its ANCHOR AND SIDE — the element plus the named
  feature it sits against and on which side of it — because the same collected
  noun can name two different required elements (BETWEEN two features vs
  OUTSIDE a feature), and one satisfied instance must not silently discharge
  the other.
"""

import re

from flagscale_agent.react.tools.read_task_statement import _BLINDSPOT_PROMPT


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", s)


# --- the rule is present, in its own words ---------------------------------
def test_anchor_and_side_rule_present():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "ANCHOR AND SIDE" in flat
    assert "on which side of it" in flat


def test_anchor_rule_explains_why_it_is_needed():
    """The rule must carry its own reason, not just the imperative."""
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "two DIFFERENT required elements" in flat
    assert "BETWEEN two" in flat and "OUTSIDE a named feature" in flat


def test_anchor_rule_forbids_merging_into_one_line():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "settle those on separate lines" in flat
    assert "never as one carries-it-or-not choice" in flat


def test_anchor_rule_states_the_discharge_failure_it_prevents():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "silently discharging the other" in flat


# --- it stays domain-agnostic (no benchmark literal leaked in) -------------
def test_anchor_rule_is_domain_agnostic():
    flat = _flat(_BLINDSPOT_PROMPT)
    for banned in ("GGTCTC", "GAGACC", "BsaI", "BsaI-HF", "primer", "oligotm"):
        assert banned.lower() not in flat.lower(), f"leaked {banned!r}"


# --- ablation: deleting THIS clause must actually turn a test red ----------
def test_ablation_anchor_rule_removal_is_detected():
    """Build the ablated prompt via the same slice the rule occupies.

    The assertion is on THIS clause's own bytes, so a fix elsewhere (e.g. the
    FLOOR clause) cannot make this test green by accident.
    """
    a = _BLINDSPOT_PROMPT.index("AND name its")
    b = _BLINDSPOT_PROMPT.index("Naming the anchor")
    assert a < b, "anchor clause must precede its rationale"
    arm_b = _BLINDSPOT_PROMPT[:a] + _BLINDSPOT_PROMPT[b:]
    assert len(arm_b) == len(_BLINDSPOT_PROMPT) - (b - a)
    flat_b = _flat(arm_b)
    assert "ANCHOR AND SIDE" not in flat_b
    assert "on which side of it" not in flat_b
    # and the real prompt still has it
    assert "ANCHOR AND SIDE" in _flat(_BLINDSPOT_PROMPT)

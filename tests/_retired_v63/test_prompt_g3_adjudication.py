"""G3 adjudication clause: a measured violation of a stated bound IS the verdict.

Regression guard for the delivery-gate text in flagscale_agent/react/prompt.py.

Why this test exists: across a 23-loss batch the LARGEST single loss bucket was not
"never measured" and not "capability wall" — it was the adjudication step: the agent's
own instrument already printed a value that violated a bound the task stated (file size
over a ceiling, accuracy under a floor, a similarity short of a threshold, a boolean
False where True was required, a string missing its required prefix) and the agent
re-explained that number as acceptable and shipped anyway. G3 covered "a deviation you
can argue away" but said nothing about "a number you measured outside the bound", which
is the same failure one step later. This clause pins that step.

The assertions are pinned to phrasing, so a later edit that softens the rule turns this
red instead of silently deleting it.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"

# Read the SOURCE bytes, not an imported symbol: the clause lives in the prompt file
# and must be checkable without importing the package (no runtime deps, no config).
FLAT = re.sub(r"\s+", " ", SRC.read_text(encoding="utf-8")).lower()



def test_g3_a_measured_out_of_bound_number_is_a_verdict():
    """The clause must exist and name the shape: measured value outside the bound."""
    assert "a number you measured outside the stated bound is a verdict" in FLAT
    # the failure must be typed as FAILED, not as "close"
    assert "the run is failed" in FLAT


def test_g3_rejects_the_re_verdict_phrasings():
    """The specific re-adjudications that lost real trials must be named as refused."""
    for phrase in (
        "not \"close\"",
        "not \"fast enough\"",
        "good enough",
        "already n times better",
        "works fine",
        "within the tolerance i chose",
    ):
        assert phrase in FLAT, phrase


def test_g3_forbids_a_self_chosen_baseline_as_the_comparison():
    """Comparing against a baseline you selected is the A2 shape of the same error."""
    assert "self-chosen baseline" in FLAT
    # the only two licensed responses
    assert "fix it until the measured value is inside the bound" in FLAT
    assert "recorded degrade" in FLAT


def test_g3_does_not_grow_a_new_top_level_gate_step():
    """CONTAINMENT RULE: the clause is a G3 paragraph, not a new G-numbered check."""
    import re

    steps = re.findall(r"^\s+(S\d+)\s+[A-Z]", SRC.read_text(encoding="utf-8"), re.M)
    gates = re.findall(r"^\s+(G\d)\s+[A-Z]", SRC.read_text(encoding="utf-8"), re.M)
    assert steps == ["S1", "S2", "S3", "S4"], steps
    assert gates == ["G1", "G2", "G3", "G4", "G5"], gates

"""G3 adjudication clause: a measured violation of a stated bound IS the verdict.

Regression guard for the measurement-adjudication text in
flagscale_agent/react/prompt.py, adapted to the CURRENT bytes. RULING:
以现在代码为准，修改单测. Why this test exists: across a 23-loss batch the LARGEST
single loss bucket was the adjudication step — the agent's own instrument
already printed a value that violated a bound the task stated and the agent
re-explained that number as acceptable and shipped anyway. The prompt rewrite
re-homed the clause into the measurement section (§3) and the closing-mode
rule (§7); the assertions below pin the CURRENT phrasings, so a later edit
that softens the rule turns this red instead of silently deleting it.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"

# Read the SOURCE bytes, not an imported symbol: the clause lives in the prompt file
# and must be checkable without importing the package (no runtime deps, no config).
FLAT = re.sub(r"\s+", " ", SRC.read_text(encoding="utf-8")).lower()


def test_g3_a_measured_out_of_bound_number_is_a_verdict():
    """The clause must exist and name the shape: measured value outside the bound."""
    assert "a number you measured outside a stated bound is a verdict" in FLAT
    # it is a verdict, not a data point to be explained away
    assert "not a data point" in FLAT


def test_g3_rejects_the_re_verdict_phrasings():
    """The specific re-adjudications that lost real trials must be named as refused.

    Current refusal list (the old 'good enough'/'works fine'/'already n
    times better' strings are gone from the file — grep 0 — and were collapsed
    into these three); 'not "close enough"' still lives in the §5 deliverable
    rule and is pinned there.
    """
    for phrase in (
        'not "fast enough"',
        'not "within my own tolerance"',
        'not "explained by something i did not check"',
        'not "close enough"',
    ):
        assert phrase in FLAT, phrase


def test_g3_forbids_a_self_chosen_baseline_as_the_comparison():
    """Comparing against a baseline you selected is the A2 shape of the same error.

    Current wording: a difference must name the baseline it was measured
    against; re-measuring after a regression proves nothing; a continuous
    metric keeps claiming against the best value seen, NOT the initial
    baseline. The only two licensed responses: fix it, or record the out-of-
    bounds line as a stated degrade.
    """
    assert "a difference must name the baseline it was measured against" in FLAT
    assert "one re-measured after a regression was seen proves nothing" in FLAT
    assert "not the initial baseline" in FLAT
    assert "either fix it or record in one line that it is out of bounds" in FLAT
    assert "that licence covers only a bound you chose" in FLAT


def test_g3_does_not_grow_a_new_top_level_gate_step():
    """CONTAINMENT RULE: the clause lives inside the situation loop, not as a
    new numbered gate. Current file has S1..S5 situation clauses and NO
    G-numbered gates at all."""
    steps = re.findall(r"^\s+(S\d+)\s+[A-Z]", SRC.read_text(encoding="utf-8"), re.M)
    gates = re.findall(r"^\s+(G\d)\s+[A-Z]", SRC.read_text(encoding="utf-8"), re.M)
    assert steps == ["S1", "S2", "S3", "S4", "S5"], steps
    assert gates == [], gates

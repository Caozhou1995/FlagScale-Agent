"""Pin §3(c)'s constant-offset clause.

Evidence (2026-11-23, dna-assembly, job 2026-11-23__15-16-38):
the agent's own self-check sliced `ann = seq[11:]` — 11 hard-coded as
6(site)+1(spacer)+4(overhang) — so the 5' flank axis was structurally
unmeasurable in its own check and the self-check reported green while the
grader's `assert i >= 1` failed. A check whose slicing constant already
encodes the answer is a fixed value, not a measurement.
"""
from flagscale_agent.react.prompt import render_prompt


def test_prompt_forbids_constant_offset_as_measurement():
    p = render_prompt()
    assert "hard-codes as a constant offset" in p
    assert "parse it out of the delivered bytes" in p


def test_prompt_keeps_the_original_red_question():
    # the pre-existing sentence must survive (not be replaced by the new one)
    p = render_prompt()
    assert "Could this check have gone red" in p
    assert "Passing tests do not imply the domain's question was asked." in p

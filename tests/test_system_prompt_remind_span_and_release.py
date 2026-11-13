# Tests for the 2026-11-10 beat revision: W3 slot SPAN + bounded release.
#
# Two independent defects were measured in job 2026-11-10__10-44-55 (22 trials,
# 102 beat blocks across 16 trials). Each is pinned separately here so neither
# can hide the other.
#
# (A) W3 SLOT SCOPE. The checker looked for the arrow on the FIRST line carrying
#     the W3 label only. The natural two-line write
#         W3 FRESH PROBE - python3 -m pytest tests/x.py -q
#         4 passed in 0.61s; bar=all green; VERDICT=PASS
#     was refused although the reading was right there. The model's own words
#     while guessing the accepted shape: "Maybe the backticks are the issue",
#     "Let me try with plain ASCII arrow". Four re-issues spent guessing FORMAT.
#
# (B) NO EXIT. accept_override refuses every reason by design (72/72 trials had
#     escaped that way before), so a re-issue that never matched the shape had
#     no way forward: 3 trials hit "re-issued still empty" and the only
#     remaining move was `step_skip` past the step.
#
# Boundary kept deliberately: the W3 span does NOT run past the next label, so
# RISK SWEEP's arrows cannot satisfy W3, and a duplicated W3 whose SECOND copy
# carries the arrow is still refused. Broadening to the whole blob turns those
# two red - they are the reason this file exists.

from __future__ import annotations

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.system_prompt_remind import (
    SystemPromptRemindGuard,
    _review_labels_present,
)

ARROW = "\u2192"  # the character the beat's message asks for


def _filled(w3: str) -> str:
    return (
        "W0 UNVERIFIED LAYER - x\n"
        "W1 FAILURE RECEIPT - x\n"
        "W2 TRAP INVENTORY WALK - x\n"
        f"{w3}\n"
        "W4 TRAP SHAPES - x\n"
        "RISK SWEEP - r1 " + ARROW + " e1 " + ARROW + " side"
    )


def _covers(text: str):
    return _review_labels_present({"verification": [text]})


def _ctx(args):
    return GuardContext(tool_name="plan_update", tool_args=args)


def _step_done(verification=None, step_id=1, notes=None):
    a = {"action": "step_done", "step_id": step_id}
    if verification is not None:
        a["verification"] = verification
    if notes is not None:
        a["notes"] = notes
    return a


# -- (A) W3 slot span --------------------------------------------------------
def test_two_line_w3_is_accepted():
    """THE measured regression: command on the W3 line, its reading next line."""
    text = _filled(
        "W3 FRESH PROBE - python3 -m pytest tests/x.py -q\n"
        "4 passed in 0.61s; bar=all green; VERDICT=PASS"
    )
    covers = _covers(text)
    assert bool(covers) is True, f"two-line W3 refused: {covers.missing}"


def test_one_line_arrow_w3_still_accepted():
    text = _filled(
        "W3 FRESH PROBE - `git status --porcelain` " + ARROW
        + " 3 ' M' lines; bar=only contaminated files; VERDICT=PASS"
    )
    assert bool(_covers(text)) is True


def test_three_line_w3_is_accepted():
    """cmd / output / bar on three lines - all inside the same labelled slot."""
    text = _filled("W3 FRESH PROBE - `wc -l f`\nprinted 42\nbar=<=50; VERDICT=PASS")
    assert bool(_covers(text)) is True


def test_heading_only_w3_is_still_refused():
    """The span is not a licence to write a heading and skip the reading."""
    for w3 in (
        "W3 FRESH PROBE - tests were run",
        "W3 FRESH PROBE - nothing",
        "W3 FRESH PROBE -",
        "W3 FRESH PROBE - `pytest`\nit passed",
    ):
        covers = _covers(_filled(w3))
        assert bool(covers) is False, f"a reading-less W3 passed: {w3!r}"


def test_risk_arrow_cannot_satisfy_w3():
    """BOUNDARY: an arrow in RISK SWEEP (every filled review has one) must not
    leak across the label boundary into W3."""
    covers = _covers(_filled("W3 FRESH PROBE - nothing"))
    assert bool(covers) is False
    assert any("W3" in m for m in covers.missing), covers.missing


def test_duplicate_w3_second_copy_still_first_match():
    """BOUNDARY, unchanged: the span starts at the FIRST W3-bearing line, so a
    second W3 copy carrying the arrow does not rescue the first. Refusing is
    the safe direction - a false reject costs one retry, a false accept ships
    an unreviewed step."""
    dup = (
        "W0 x\nW1 x\nW2 x\nW3 nothing here\nW3 `c` -> o\nW4 x\n"
        "RISK r -> e"
    )
    covers = _covers(dup)
    assert bool(covers) is False
    assert any("W3" in m for m in covers.missing), covers.missing


def test_span_does_not_swallow_a_later_probe():
    """W3 empty but a LATER slot happens to contain 'cmd -> out': the span must
    stop at that later label."""
    text = (
        "W0 x\nW1 x\nW2 x\nW3 FRESH PROBE -\n"
        "W4 TRAP SHAPES - regen -> compare (an arrow inside W4)"
    )
    assert bool(_covers(text)) is False


# -- (B) bounded release -----------------------------------------------------
def test_first_and_second_refusals_still_block():
    """The release must not become a one-shot. Refusals 1 and 2 are BLOCKS -
    without this the beat is decorative again."""
    g = SystemPromptRemindGuard()
    assert g.check_pre(_ctx(_step_done())).action == "block"          # beat
    for i in (1, 2):
        v = g.check_pre(_ctx(_step_done(notes="still no slots")))
        assert v is not None and v.action == "block", (
            f"refusal {i} did not block: {getattr(v, 'action', None)}"
        )


def test_release_after_max_refusals_is_an_inject_not_a_block():
    """After _MAX_REFUSALS the declaration goes through, and the gap is put on
    the record: the verdict is an INJECT (non-blocking) naming what is missing."""
    g = SystemPromptRemindGuard()
    assert g.check_pre(_ctx(_step_done())).action == "block"           # refusal 1
    assert g.check_pre(_ctx(_step_done(notes="no slots"))).action == "block"  # 2
    assert g.check_pre(_ctx(_step_done(notes="no slots"))).action == "block"  # 3
    v = g.check_pre(_ctx(_step_done(notes="no slots")))                # 4th: out
    assert v is not None and v.action != "block", (
        "the guard kept blocking after its own refusal budget - a wall, not a gate"
    )
    assert "W3" in v.message or "slot" in v.message, (
        f"the release hid what was missing: {v.message[:200]}"
    )


def test_release_does_not_fire_for_a_conforming_reissue():
    """A declaration that fills the slots passes on the SECOND try - the release
    valve is for unshapeable text, not a substitute for the review."""
    g = SystemPromptRemindGuard()
    assert g.check_pre(_ctx(_step_done())).action == "block"
    v = g.check_pre(_ctx(_step_done(verification=[
        _filled("W3 FRESH PROBE - `id` " + ARROW + " uid=0; bar=<1; VERDICT=PASS")
    ])))
    assert v is None, f"a conforming re-issue was still refused: {v.action}"


def test_per_signature_budget_is_independent():
    """Each declaration identity carries its own budget: a second step is not
    released just because the first one burned its refusals."""
    g = SystemPromptRemindGuard()
    for _ in range(4):
        g.check_pre(_ctx(_step_done(step_id=1)))
    v = g.check_pre(_ctx(_step_done(step_id=2)))
    assert v is not None and v.action == "block", (
        "a fresh step inherited the previous step's exhausted budget"
    )


# -- the boundary that must NOT have moved ----------------------------------
def test_guard_still_never_judges_content():
    """Structure only. A deliberately WRONG but shaped review passes; confident
    prose with no slots does not. Both directions on the SAME guard instance."""
    g = SystemPromptRemindGuard()
    g.check_pre(_ctx(_step_done(step_id=9)))
    wrong_but_shaped = [
        _filled("W3 FRESH PROBE - `echo hi` " + ARROW + " hi; bar=none; VERDICT=PASS")
    ]
    assert g.check_pre(
        _ctx(_step_done(verification=wrong_but_shaped, step_id=9))
    ) is None


# -- slot segmentation: the two shapes that broke the FIRST version ----------
def test_w3_mentioned_midline_in_w2_does_not_hijack_the_span():
    """MEASURED DEFECT (found by the beat itself at close-out): a W2 line whose
    prose MENTIONS another slot -- "a duplicated W3 whose SECOND copy carries
    the arrow is still refused" -- used to open the span early, so the real
    W3 line never got read and a filled-in review was refused. Slot boundaries
    are line HEADINGS; a mid-line mention is prose, not a boundary."""
    text = (
        "W0 UNVERIFIED LAYER - x\n"
        "W1 FAILURE RECEIPT - x\n"
        "W2 TRAP INVENTORY WALK - BOUNDARY: a duplicated W3 whose SECOND copy "
        "carries the arrow is still refused; also RISK arrows do not count\n"
        "W3 FRESH PROBE - `grep -c x f` " + ARROW + " 3; bar=>=3; VERDICT=PASS\n"
        "W4 TRAP SHAPES - x\n"
        "RISK SWEEP - r1 " + ARROW + " e1 " + ARROW + " side"
    )
    covers = _covers(text)
    assert bool(covers) is True, f"a W2 mid-line mention hijacked the span: {covers.missing}"


def test_all_slots_on_one_line_still_accepted():
    """The compact one-line form (every slot crammed onto one line, no line
    boundary to segment on) must not be lost by head-anchoring the labels."""
    one = ("W0 x W1 x W2 x W3 cmd=" + ARROW + " out verdict=pass bar=<1 "
           "W4 x RISK SWEEP r1 " + ARROW + " e1 " + ARROW + " side")
    assert bool(_covers(one)) is True, "the compact one-line review was refused"

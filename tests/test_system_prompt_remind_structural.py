"""Structural self-review gate: the beat must be released by FILLING SLOTS, not by prose.

Live evidence this pins (job 2026-11-10__00-11-22, 89 trials):
  * the beat fired on 72 trials, 227 times total (3.2 per trial)
  * all 72 (100%) carried `OVERRIDE REQUIRED: Add "_override_reason"` — every
    agent released the block with a 5+ char reason and never wrote a review
  * the guard's own re-issue path was `if sig in self._beaten: return None`,
    i.e. the second submission passed with NO check whatsoever

The fix under test has two independent halves, and each is falsified separately
so that neither can hide the other:
  (A) the beat is non-overridable — `_override_reason` must NOT release it;
  (B) the re-issue is accepted only when this declaration's own
      verification/notes text carries the six literal labels W0/W1/W2/W3/W4/
      RISK, with W3 additionally carrying the `<command> → <output>` form.

The guard does NOT judge the CONTENT of any answer. A deliberately wrong but
filled-in review must pass — that is asserted below, because a test suite that
only ever feeds good answers cannot tell a structure check from a content check.
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.system_prompt_remind import (
    SystemPromptRemindGuard,
    _review_labels_present,
)

FILLED = [
    "W0 UNVERIFIED LAYER: did not read the grader for task B; skipped that read",
    "W1 FAILURE RECEIPT: the check I left unrun is the empty-input path",
    "W2 TRAP INVENTORY WALK: ENUMERATE=boundary; PARSE-SCHEMA=n/a; RANK=order; RUN-TIME=n/a",
    "W3 FRESH PROBE: python3 /tmp/probe.py → 42 rows",
    "W4 TRAP SHAPES: (1) self-check (2) trailing newline (3) one sample (4) notes-only (5) unit",
    "RISK SWEEP: risk=r1 → evidence=e1 → side=fixed",
]

# Deliberately WRONG content, but every slot filled. Must pass: the guard
# checks structure, never correctness.
FILLED_BUT_WRONG = [
    "W0 UNVERIFIED LAYER: nothing at all, everything verified (a lie)",
    "W1 FAILURE RECEIPT: no failure possible here (a lie)",
    "W2 TRAP INVENTORY WALK: none of the four layers can bite (a lie)",
    "W3 FRESH PROBE: echo hello → this is not a real probe (a lie)",
    "W4 TRAP SHAPES: none apply (a lie)",
    "RISK SWEEP: no risks (a lie)",
]


def _ctx(args):
    return GuardContext(tool_name="plan_update", tool_args=args)


def _step_done(verification=None, notes="", step_id=1):
    a = {"action": "step_done", "step_id": step_id}
    if verification is not None:
        a["verification"] = verification
    if notes:
        a["notes"] = notes
    return a


# ── (A) non-overridable ────────────────────────────────────────────────────
def test_beat_is_not_releasable_by_override_reason():
    g = SystemPromptRemindGuard()
    v = g.check_pre(_ctx(_step_done()))
    assert v is not None and v.action == "block"
    assert v.overridable is False, (
        "the beat is overridable again — measured 72/72 trials escaped this way"
    )
    assert g.accept_override("the review was not needed here, trust me", _ctx({})) is False, (
        "accept_override returned True: a prose reason still releases the beat"
    )


# ── (B) structural slots ───────────────────────────────────────────────────
def test_empty_reissue_is_still_blocked():
    g = SystemPromptRemindGuard()
    g.check_pre(_ctx(_step_done()))
    v2 = g.check_pre(_ctx(_step_done(verification=["done"], notes="all good")))
    assert v2 is not None and v2.action == "block", (
        "an empty re-issue passed — the beat is a one-shot formality again"
    )
    assert "W0" in v2.message and "W3" in v2.message, (
        "the second block must name the missing slots so the agent can fill them"
    )


def test_filled_review_releases_the_beat():
    g = SystemPromptRemindGuard()
    g.check_pre(_ctx(_step_done()))
    v2 = g.check_pre(_ctx(_step_done(verification=FILLED)))
    assert v2 is None, f"a fully filled review was still blocked: {v2.message[:200]}"


def test_structurally_filled_but_wrong_content_passes():
    """The guard checks FORM. Wrong-but-filled must pass, or it is grading content."""
    g = SystemPromptRemindGuard()
    g.check_pre(_ctx(_step_done()))
    v2 = g.check_pre(_ctx(_step_done(verification=FILLED_BUT_WRONG)))
    assert v2 is None, (
        "a filled-in review was blocked because its CONTENT was judged wrong — "
        "the guard must not do semantic judgement"
    )


def test_labels_may_live_in_notes_instead_of_verification():
    g = SystemPromptRemindGuard()
    g.check_pre(_ctx(_step_done()))
    v2 = g.check_pre(_ctx(_step_done(notes="\n".join(FILLED))))
    assert v2 is None, "notes-only review was blocked; both fields are the agent's own text"


def test_missing_one_label_blocks():
    g = SystemPromptRemindGuard()
    g.check_pre(_ctx(_step_done()))
    partial = [x for x in FILLED if not x.startswith("RISK")]
    v2 = g.check_pre(_ctx(_step_done(verification=partial)))
    assert v2 is not None and v2.action == "block"
    assert "RISK" in v2.message


def test_w3_without_arrow_blocks():
    """W3 asks for '<command> → <output>'. A heading with no reading is not a probe."""
    g = SystemPromptRemindGuard()
    g.check_pre(_ctx(_step_done()))
    no_arrow = [x.replace("python3 /tmp/probe.py → 42 rows", "python3 /tmp/probe.py printed 42 rows")
                for x in FILLED]
    v2 = g.check_pre(_ctx(_step_done(verification=no_arrow)))
    assert v2 is not None and v2.action == "block", (
        "a W3 label with no arrow passed — the two slots are the point of W3"
    )
    assert "W3" in v2.message


# ── scope: nothing else is ever blocked ────────────────────────────────────
def test_other_plan_actions_are_untouched():
    g = SystemPromptRemindGuard()
    for args in (
        {"action": "step_doing", "step_id": 1},
        {"action": "add_steps", "new_steps": ["x"]},
        {"action": "predict", "prediction_statement": "s", "prediction_expected": "1"},
    ):
        assert g.check_pre(_ctx(args)) is None, f"{args['action']} was blocked"


def test_other_tools_are_untouched():
    g = SystemPromptRemindGuard()
    assert g.check_pre(GuardContext(tool_name="shell", tool_args={"command": "ls"})) is None


def test_badly_bounded_label_does_not_count():
    """'sw3' must not satisfy W3 by substring, and 'W33' must not either."""
    r = _review_labels_present({"verification": ["sw3 x → y", "W33 z → q"]})
    assert "W3" in r.missing, f"an out-of-bound W3 substring counted as coverage: {r.missing}"


def test_complete_action_is_gated_too():
    """action=complete declares the whole plan finished — the strongest declaration."""
    g = SystemPromptRemindGuard()
    v = g.check_pre(_ctx({"action": "complete"}))
    assert v is not None and v.action == "block"
    v2 = g.check_pre(_ctx({"action": "complete", "notes": "\n".join(FILLED)}))
    assert v2 is None


def test_override_cannot_release_the_beat():
    """Measured (job 2026-11-10__00-11-22): 72/72 beaten trials released the
    block with a 5+ char `_override_reason` — the beat averaged 3.2 fires per
    trial while no review was ever produced. So the override channel must be
    closed on BOTH sides: the verdict is non-overridable AND accept_override
    refuses every reason. Either side alone is not enough (a caller can invoke
    accept_override directly).
    """
    from flagscale_agent.react.guard import GuardContext

    g = SystemPromptRemindGuard()
    v = g.check_pre(
        GuardContext(tool_name="plan_update", tool_args={"action": "step_done", "step_id": 1}, tool_result="")
    )
    assert v is not None and v.action == "block"
    assert getattr(v, "overridable", True) is False, "the beat went back to being an advisory"

    ctx = GuardContext(tool_name="plan_update", tool_args={"action": "step_done", "step_id": 1}, tool_result="")
    for reason in ("self-review done", "I checked everything carefully", ""):
        assert g.accept_override(reason, ctx) is False, "a prose reason released the beat"

    # And the re-issue with real slots is the ONLY release path.
    v2 = g.check_pre(
        GuardContext(
            tool_name="plan_update",
            tool_args={"action": "step_done", "step_id": 1, "verification": [
                "W0 unverified layer: x",
                "W1 settled-too-early pattern: x",
                "W2 trap inventory walk: x",
                "W3 cmd=id -> output=uid=0 verdict=pass bar=<1",
                "W4 instance: x",
                "RISK SWEEP: r1 -> e1 -> side",
            ]},
            tool_result="",
        )
    )
    assert v2 is None


# --- False-positive guard: the check must not tax a real, well-filled review.
# Measured motivation (2026-11-10): the six labels are filled by hand inside a
# 900s task, so any form the agent plausibly writes must PASS. A check that
# rejects real submissions is worse than no check (it burns a retry per step).
_REAL_FORM = """W0 unverified layer: 部署树是否污染 — 未验证
W1 settled-too-early: 我三次凭记忆写数字 — 已改
W2 trap inventory walk: four layers walked
W3 `python3 -m pytest tests/x.py -q` → 24 passed
W4 instance: closure over own artifact
RISK SWEEP: 误伤 → 用消融 → side=safe"""


@pytest.mark.parametrize(
    "name,text,must_pass",
    [
        ("hand-filled long form", _REAL_FORM, True),
        ("ascii arrow in W3", _REAL_FORM.replace("→ 24 passed", "-> 24 passed"), True),
        ("fat arrow in W3", _REAL_FORM.replace("→ 24 passed", "=> 24 passed"), True),
        ("lowercase labels", _REAL_FORM.lower(), True),
        ("W3 heading only, no reading", _REAL_FORM.replace("`python3 -m pytest tests/x.py -q` → 24 passed", "已运行测试"), False),
        ("W4 omitted", "\n".join(l for l in _REAL_FORM.splitlines() if not l.startswith("W4")), False),
        ("RISK omitted", "\n".join(l for l in _REAL_FORM.splitlines() if not l.startswith("RISK")), False),
    ],
)
def test_real_world_fill_forms(name, text, must_pass):
    covers = _review_labels_present({"notes": text})
    assert bool(covers) is must_pass, f"{name}: missing={covers.missing}"


def test_guard_never_judges_content_only_form():
    """The user's hard boundary: the guard does NOT decide whether a review is
    good. This pins both directions on the SAME pair of inputs, so a later
    "improvement" that starts scoring content turns this red.

    (a) a deliberately WRONG but structurally filled review MUST pass;
    (b) confident prose with no slots MUST be refused.
    """
    wrong_but_filled = """W0 unverified layer: nothing at all (this is a lie)
W1 settled-too-early: never happened (lie)
W2 trap inventory walk: (lie)
W3 cmd=true -> output=(nothing) verdict=pass bar=<0
W4 instance: (lie)
RISK SWEEP: none -> none -> side=none"""
    assert bool(_review_labels_present({"notes": wrong_but_filled})) is True

    right_but_no_slots = (
        "I have carefully verified every layer of this deliverable and I am "
        "fully confident. All checks pass and nothing is unverified."
    )
    assert bool(_review_labels_present({"notes": right_but_no_slots})) is False


def test_known_limit_duplicate_w3_first_match_only():
    """KNOWN LIMIT, deliberate: the W3 arrow is looked up on the FIRST line that
    carries the W3 label. If an agent writes W3 twice and puts the arrow on the
    second copy, the check refuses it.

    Why leave it: the failure mode is a FALSE REJECT (one extra retry), never a
    false accept, so it cannot let an unreviewed declaration through. Pinned
    here so a future "fix" that silently broadens the search (e.g. scanning the
    whole blob, which would let RISK SWEEP's arrows satisfy W3) turns this red.
    """
    dup = "W0 x\nW1 x\nW2 x\nW3 nothing here\nW3 `c` -> o\nW4 x\nRISK r -> e"
    covers = _review_labels_present({"notes": dup})
    assert bool(covers) is False
    assert any("W3" in m for m in covers.missing)

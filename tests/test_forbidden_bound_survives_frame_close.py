"""The bound must ride the SAME surface as the list it constrains.

Why this file exists (2026-11-27, dna-assembly 9L8TfcV — second dna run):

Run 1 attributions put the clause in prompt E9 (the system prompt). Still failed:
8/8 primers had clamp=0. The trace showed the agent citing its OWN plan
forbidden entry #1 ("do not add extra bases 5' of the BsaI site ... WITHOUT
VERIFYING a tail parser strips it") twice (:1329, :1913) to refuse the NEB
requirement.

The mechanism, measured on the deploy tree:
  agent.py _build_plan_context  injects a step's `acceptance` ONLY while the
  step is live (status pending/doing). Once a step closes it renders identity +
  evidence and DROPS acceptance. The Frame step closes early — so a rule placed
  in the Frame's acceptance is read once at planning time and is GONE by the
  time the agent writes a forbidden entry at 00:32. Meanwhile _forbidden_lines
  re-renders the agent's own list at the TOP of the context every single turn.

Net effect: the self-authored prohibition outlived the constraint that bounds
it. Fix: the bound is emitted INSIDE the forbidden block, before the entries,
so the two can never be separated.

These tests fail if that bound is removed, moved out of the block, or placed
after the entries (order matters: a bound printed after the list is read as a
footnote to it, not as its frame). They also pin the consumer path — asserting
on _forbidden_lines alone would still pass if _build_plan_context stopped
calling it.
"""
import os
import sys
import tempfile
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from flagscale_agent.react.plan import TaskPlan  # noqa: E402
from flagscale_agent.react.tools.plan_create import PlanCreateTool  # noqa: E402

BOUND_MARK = "BOUND: this list is a note to MYSELF, subordinate to the task text"
BAN = "do not add X"


def _plan_with_ban():
    d = tempfile.mkdtemp()
    tp = TaskPlan(plan_dir=d)
    PlanCreateTool(tp, session_id="probe").execute(
        title="probe",
        steps=[{"title": "s1"}],
        forbidden=[{"action": BAN, "why": "my unverified guess", "instead": "Y"}],
    )
    return tp


def _per_turn_context(tp):
    """The PRODUCTION per-turn path — not context_for_prompt (no caller)."""
    from flagscale_agent.react.agent import WorkerAgent

    stub = types.SimpleNamespace(task_plan=tp)
    return WorkerAgent._build_plan_context.__get__(stub, object)()


def test_bound_is_emitted_with_an_existing_list():
    tp = _plan_with_ban()
    assert BOUND_MARK in tp._forbidden_lines(tp.get_active())[0:2][1] or BOUND_MARK in "\n".join(
        tp._forbidden_lines(tp.get_active())
    )


def test_bound_precedes_the_entries():
    """Direction, not distance: a footnote AFTER the list is not a frame."""
    tp = _plan_with_ban()
    lines = tp._forbidden_lines(tp.get_active())
    i_bound = next(i for i, l in enumerate(lines) if BOUND_MARK in l)
    i_ban = next(i for i, l in enumerate(lines) if f"✗ {BAN}" in l)
    assert i_bound < i_ban, f"bound printed after the entry it must bound ({i_bound} > {i_ban})"


def test_bound_lives_INSIDE_the_forbidden_block_not_the_step_list():
    tp = _plan_with_ban()
    ctx = _per_turn_context(tp)
    lo = ctx.index("<current-forbidden>")
    hi = ctx.index("</current-forbidden>")
    assert BOUND_MARK in ctx[lo:hi], (
        "bound escaped the forbidden block; outside it, it is subject to the "
        "step-acceptance drop that caused the original failure"
    )


def test_bound_survives_the_frame_step_closing():
    """THE regression: this is the exact state the agent was in at 00:32."""
    tp = _plan_with_ban()
    before = _per_turn_context(tp)
    assert BOUND_MARK in before, "bound missing even while Frame is live"

    tp.update_step(tp.get_active()["steps"][0]["id"], status="done", verification=["framed"])
    after = _per_turn_context(tp)

    assert BAN in after, "sanity: the agent's own entry must still render"
    assert BOUND_MARK in after, (
        "bound vanished when the Frame step closed — the original bug is back: "
        "the self-authored list then outlives the rule that bounds it"
    )


def test_bound_survives_every_single_step_closing():
    """Not just the Frame: close every step, the bound must still be there."""
    tp = _plan_with_ban()
    for s in list(tp.get_active()["steps"]):
        tp.update_step(s["id"], status="done", verification=["closed"])
    ctx = _per_turn_context(tp)
    assert BOUND_MARK in ctx, "bound disappeared once all steps were closed"


def test_bound_states_the_supersession_rule_not_a_virtue():
    """'be careful' never stopped anyone; the collision outcome must be stated."""
    tp = _plan_with_ban()
    block = "\n".join(tp._forbidden_lines(tp.get_active()))
    assert "the requirement wins" in block, "collision outcome not stated"
    assert "unverified guess" in block, "own-uncertainty case not named"
    assert "do not ship X unlabelled" in block, "the remedy (label, do not ban) missing"


def test_rule_is_task_agnostic():
    tp = _plan_with_ban()
    block = "\n".join(tp._forbidden_lines(tp.get_active())).lower()
    for bad in ("bsai", "ggtctc", "oligotm", "primer", "clamp", "gpt2", "gcode", "dna"):
        assert bad not in block, f"task literal leaked into the plan context: {bad!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

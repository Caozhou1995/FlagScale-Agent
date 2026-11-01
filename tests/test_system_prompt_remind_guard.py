"""SystemPromptRemindGuard: ONE block per completed step, then it passes.

The beat semantics (user-directed 2026-10-31): the FIRST submission of a
completion declaration is blocked once — forcing a scientific self-review —
and the re-issued SAME declaration passes untouched. The harness never grades
the review and never blocks any other agent behaviour. Tests pin both sides:
every shape that declares a step done gets exactly one beat, and every
neighbouring shape (other tools, non-declaration plan_updates, re-issues)
stays silent.
"""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.system_prompt_remind import (
    SystemPromptRemindGuard,
    _declares_completion,
)


def _ctx(tool_name="plan_update", args=None, result="Plan updated"):
    return GuardContext(
        tool_name=tool_name,
        tool_args=args or {},
        tool_result=result,
    )


def _pre(g, args, tool_name="plan_update"):
    return g.check_pre(_ctx(tool_name, args))


def test_first_step_done_is_blocked_with_beat_message():
    g = SystemPromptRemindGuard()
    v = _pre(g, {"action": "step_done", "step_id": 3})
    assert v is not None and v.action == "block"
    low = v.message.lower()
    assert "blocked once" in low
    assert "re-issue" in low
    assert "trap" in low  # the task-goal trap question is part of the beat


def test_beat_message_carries_the_four_self_audit_disciplines():
    """The beat must teach the failure classes that actually cost runs here:
    observed/inferred confusion, unprobed counterfactuals, checkers that
    cannot fail, and same-shape retry loops."""
    g = SystemPromptRemindGuard()
    v = _pre(g, {"action": "step_done", "step_id": 1})
    low = v.message.lower()
    assert "observed vs inferred" in low
    assert "counterfactual" in low
    assert "verifier must be able to fail" in low
    assert "strike count" in low
    # anti-decay: the old generic wording must NOT return (it taught nothing
    # actionable and was replaced because four real failure classes were absent)
    assert "what did this step observe from the world" not in low


def test_reissue_of_same_step_passes():
    g = SystemPromptRemindGuard()
    args = {"action": "step_done", "step_id": 3}
    assert _pre(g, args) is not None  # beat
    # The agent adds real evidence between the block and the re-issue; the
    # signature must NOT change (evidence fields are not identity).
    args2 = {"action": "step_done", "step_id": 3, "notes": "self-review done"}
    assert _pre(g, args2) is None


def test_different_step_gets_its_own_beat():
    g = SystemPromptRemindGuard()
    assert _pre(g, {"action": "step_done", "step_id": 1}) is not None
    assert _pre(g, {"action": "step_done", "step_id": 1}) is None  # beaten
    assert _pre(g, {"action": "step_done", "step_id": 2}) is not None  # new beat


def test_complete_is_one_beat_for_whole_plan():
    g = SystemPromptRemindGuard()
    assert _pre(g, {"action": "complete"}) is not None
    assert _pre(g, {"action": "complete"}) is None
    assert _pre(g, {"action": "complete"}) is None


def test_batch_beat_keys_on_done_row_set():
    g = SystemPromptRemindGuard()
    b1 = {
        "action": "batch",
        "updates": [
            {"step_id": 1, "status": "doing"},
            {"step_id": 2, "status": "done"},
        ],
    }
    assert _pre(g, b1) is not None  # beat on {2}
    assert _pre(g, b1) is None  # same done-set re-issue passes
    # A batch that declares a NEW done row is a different beat.
    b2 = {
        "action": "batch",
        "updates": [
            {"step_id": 2, "status": "done"},
            {"step_id": 3, "status": "done"},
        ],
    }
    assert _pre(g, b2) is not None


def test_silent_on_step_doing_and_step_skip():
    g = SystemPromptRemindGuard()
    assert _pre(g, {"action": "step_doing", "step_id": 3}) is None
    assert _pre(g, {"action": "step_skip", "step_id": 3}) is None


def test_silent_on_batch_without_done_rows():
    g = SystemPromptRemindGuard()
    args = {
        "action": "batch",
        "updates": [{"step_id": 1, "status": "doing"}],
    }
    assert _pre(g, args) is None


def test_silent_on_other_tools():
    g = SystemPromptRemindGuard()
    args = {"action": "step_done", "step_id": 1}
    assert _pre(g, args, tool_name="shell") is None
    assert _pre(g, args, tool_name="edit_file") is None


def test_check_post_never_injects():
    """The beat moved to pre-side; post must be silent (no double reminder)."""
    g = SystemPromptRemindGuard()
    args = {"action": "step_done", "step_id": 1}
    ctx = _ctx("plan_update", args)
    assert g.check_post(ctx) is None
    assert g.check_post(ctx) is None


def test_declares_completion_ignores_unknown_actions():
    assert not _declares_completion({"action": "add_steps"})
    assert not _declares_completion({})
    assert _declares_completion({"action": "batch", "updates": [{"status": "DONE"}]})

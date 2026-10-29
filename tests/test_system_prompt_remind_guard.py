"""SystemPromptRemindGuard: fires on every completion DECLARATION, and only then.

The guard's whole value is its trigger boundary, so the tests pin the boundary
from both sides: every shape that declares a step done must fire, and every
neighbouring shape that does NOT declare one must stay silent. A guard that
fires on `step_doing` trains the agent to ignore it; a guard that misses
`action=batch` with done rows leaves the most common declaration unguarded.
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


def _verdict(args, result="Plan updated", tool_name="plan_update"):
    return SystemPromptRemindGuard().check_post(_ctx(tool_name, args, result))


def test_fires_on_step_done():
    v = _verdict({"action": "step_done", "step_id": 3})
    assert v is not None
    text = v.message
    low = text.lower()
    # Pointer-only: it must ASK about the protocol, never restate its content.
    assert "S1 locate the producer" in text
    assert "verbatim" in low
    assert "negative constraints" in low


def test_fires_on_complete():
    assert _verdict({"action": "complete"}) is not None


def test_fires_on_batch_containing_a_done_row():
    args = {
        "action": "batch",
        "updates": [
            {"step_id": 1, "status": "doing"},
            {"step_id": 2, "status": "done"},
        ],
    }
    assert _verdict(args) is not None


def test_silent_on_step_doing_and_step_skip():
    assert _verdict({"action": "step_doing", "step_id": 3}) is None
    assert _verdict({"action": "step_skip", "step_id": 3}) is None


def test_silent_on_batch_without_done_rows():
    args = {
        "action": "batch",
        "updates": [{"step_id": 1, "status": "doing"}],
    }
    assert _verdict(args) is None


def test_silent_on_notes_only_update():
    assert _verdict({"action": "step_done", "step_id": 1}, result="") is None
    # ...and a different tool never fires it at all.
    assert _verdict({"action": "step_done"}, tool_name="shell") is None


def test_silent_when_the_declaration_was_blocked():
    """A blocked/errored plan_update changed no state: nothing to interrogate."""
    assert (
        _verdict(
            {"action": "step_done", "step_id": 1},
            result="[BLOCKED BY GUARD] plan_update refused",
        )
        is None
    )
    assert (
        _verdict({"action": "step_done", "step_id": 1}, result="ERROR: bad args")
        is None
    )


def test_no_cooldown_second_declaration_fires_again():
    """Per-declaration, not per-session: a stale 'I checked last time' is the bug."""
    g = SystemPromptRemindGuard()
    args = {"action": "step_done", "step_id": 1}
    assert g.check_post(_ctx(args=args)) is not None
    assert g.check_post(_ctx(args=args)) is not None


def test_never_blocks():
    """Inject-only: check_pre is a no-op, so it can never block a tool call."""
    assert SystemPromptRemindGuard().check_pre(_ctx()) is None


def test_declares_completion_ignores_unknown_actions():
    assert not _declares_completion({"action": "add_steps"})
    assert not _declares_completion({})
    assert _declares_completion({"action": "batch", "updates": [{"status": "DONE"}]})

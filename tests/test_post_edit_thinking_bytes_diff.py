"""The post-edit inject must demand a THINKING↔BYTES diff, on BOTH write tools.

Evidence (2026-11-27, official DeepSeek API, dna-assembly trial): the agent's
reasoning wrote the correct primer formula three times including the `[clamp]`
slot -- and then its `write_file` payload dropped it. Its own gate listed eight
checks, none of them clamp, and its verifier inherited the same omission, so
every self-check read GREEN while the deliverable was missing a graded
constraint. A reminder to verify the FAR END does not catch that class: the far
end was reached, the CONTENT was already wrong. These tests pin the reminder
that addresses it, and pin that it reaches write_file and edit_file alike.
"""

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.post_edit_far_end import PostEditFarEndGuard

KEY = "THINKING↔BYTES DIFF"


@pytest.fixture
def guard():
    return PostEditFarEndGuard()


def _ctx(tool_name, path="/tmp/x.txt"):
    return GuardContext(
        tool_name=tool_name,
        tool_args={"path": path, "content": "x\n", "old_string": "x", "new_string": "y"},
        tool_result=f"File written: {path}",
    )


@pytest.mark.parametrize("tool", ["write_file", "edit_file"])
def test_both_write_tools_demand_the_thinking_bytes_diff(guard, tool):
    """Coverage requirement: the reminder must not reach only one write tool."""
    v = guard.check_post(_ctx(tool))
    assert v is not None, f"{tool} must inject a far-end reminder"
    assert KEY in v.message, f"{tool} inject must demand the thinking↔bytes diff"


def test_the_reminder_names_the_failure_it_exists_for(guard):
    """A reminder that only says 'check consistency' is ignorable; this one must
    name the DROPPED decision and the self-check that inherits the omission."""
    msg = guard.check_post(_ctx("write_file")).message
    low = msg.lower()
    assert "absent from" in low or "dropped" in low
    assert "green" in low, "must name the false-green self-check it defeats"


def test_it_is_inject_only_and_never_blocks(guard):
    """User requirement: inject level only — no mechanical check, no block."""
    v = guard.check_post(_ctx("write_file"))
    assert v is not None
    assert getattr(v, "blocked", False) is False
    assert getattr(v, "action", None) in (None, "inject")

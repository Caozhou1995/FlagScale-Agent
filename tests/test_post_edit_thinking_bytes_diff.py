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


# ── The baseline of the diff: THIS turn vs the EARLIER decision ──────────────
# Evidence (2026-11-27, dna-assembly__9JscHae, official DeepSeek API): the agent
# recorded the NEB requirement ("at least 1 base 5' of the recognition site") at
# its line 117, then wrote the construction formula at its line 1585 --
# hundreds of turns later -- as `GGTCTC+'a'+OH+anneal`, with no slot at all for
# that 5' base. The design was correct and the implementation dropped it, and
# the two sat a whole run apart.
#
# The guard only has THIS turn's thinking (agent.py accumulates `thinking_text`
# per iteration and never carries it across turns), so an instruction that says
# "re-read the reasoning behind THIS write" structurally cannot see that gap:
# by the time the write happens, the deciding reasoning is not in this turn any
# more. The reminder must therefore widen its BASELINE, not its severity.

BASELINE_ANCHOR = "THEN GO BACK FURTHER"
BASELINE_OBJECT = "THAT EARLIER DECISION"
BASELINE_FAIL = "cannot point to the turn"


@pytest.mark.parametrize("tool", ["write_file", "edit_file"])
def test_the_diff_baseline_extends_past_this_turn(guard, tool):
    """The reminder must say the baseline can be an EARLIER decision.

    Without this, an implementation write is diffed against a thinking span
    that no longer contains the design — the check runs, passes, and misses
    exactly the dropped-slot class it exists for.
    """
    msg = guard.check_post(_ctx(tool)).message
    assert BASELINE_ANCHOR in msg, (
        f"{tool}: reminder must send the agent back past this turn"
    )
    assert BASELINE_OBJECT in msg, (
        f"{tool}: reminder must name the earlier decision as the diff object"
    )


def test_failing_to_locate_the_baseline_is_stated_as_not_run(guard):
    """An unlocatable baseline must read as 'the check has not run', not pass.

    This is the same self-check-inherits-the-omission trap one level up: if the
    agent cannot point to where the design was decided, it has nothing to diff
    against and must not report the write as verified.
    """
    msg = guard.check_post(_ctx("write_file")).message
    assert BASELINE_FAIL in msg, (
        "the reminder must forbid passing the check without a located baseline"
    )


def test_widening_did_not_weaken_the_original_demands(guard):
    """The old teeth must survive the widening (add, never replace)."""
    msg = guard.check_post(_ctx("write_file")).message
    for kept in (
        "MANDATORY, not optional",
        "is a DROPPED decision",          # exact case, as the bytes carry it
        "wearing a check's clothes",
        "NOT verified",
    ):
        assert kept in msg, f"widening dropped an existing demand: {kept!r}"

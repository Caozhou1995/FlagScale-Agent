"""Obligations must come from the ACTIVE PLAN, not the conversation history.

Why this exists (user ruling, 2026-10-28): the obligation set was read off the
session's conversation -- `task_text_fn` was wired to `_mech_original_task()`
(first user turn) plus the last user turns. That is wrong by construction: the
first turn is frozen while the work evolves, so every later clarification
("do not add another gate on task_complete", "let the plan carry the
obligations") is invisible to the gate. Measured cost: the guard kept
demanding a superseded obligation drawn from the origin turn while the agent's
own current plan had already replaced it -- rounds burned on a requirement that
no longer existed.

The active plan is the only always-current structure: it is rewritten as the
work changes, its steps carry the acceptance criteria actually in force, and
its notes carry the user's later rulings. So the obligation axis reads the plan,
while the GROUNDING axis (`deliverable_reading.source` quotes) keeps reading
the original task text -- those quotes are only quotable if they come from the
text the task actually stated.
"""
import pytest

from flagscale_agent.react.guard.verification import VerificationGuard


class _Plan:
    """Plan stub mirroring the surface the guard reads: get_active() -> dict."""

    def __init__(self, plan=None):
        self._plan = plan

    def get_active(self):
        return self._plan


def _plan(steps=None, thinking="", forbidden=None, title="Ship the thing"):
    return {
        "title": title,
        "thinking": thinking,
        "forbidden": forbidden or [],
        "steps": steps if steps is not None else [
            {
                "id": 1,
                "title": "build the deliverable",
                "status": "doing",
                "acceptance": ["the deliverable parses", "it is under 4 KiB"],
                "notes": "",
            }
        ],
    }


BOSS_TEXT = "ORIGINAL TASK (frozen first turn): write a report in latin."


def _guard(plan=None, task_text=BOSS_TEXT):
    return VerificationGuard(plan=plan, task_text_fn=lambda: task_text)


def test_obligations_read_the_plan_not_the_task_text():
    """The obligation corpus contains plan content and omits conversation text."""
    g = _guard(_Plan(_plan()))
    text = g._obligations_text()
    assert "build the deliverable" in text
    assert "the deliverable parses" in text
    assert "it is under 4 KiB" in text
    assert "write a report in latin" not in text, (
        "the frozen first turn must not enter the obligation set"
    )


def test_obligations_include_later_rulings_held_in_plan_notes_and_thinking():
    """A later user ruling lands in plan notes/thinking and IS then obligatory."""
    p = _plan(steps=[{
        "id": 1, "title": "build", "status": "doing",
        "acceptance": ["x"],
        "notes": "user: do not add another gate on task_complete",
    }], thinking="obligations are carried by the plan, not the conversation")
    text = _guard(_Plan(p))._obligations_text()
    assert "do not add another gate on task_complete" in text
    assert "obligations are carried by the plan" in text


def test_obligations_include_forbidden_list():
    p = _plan(forbidden=[
        {"action": "re-run the same command with a bigger timeout",
         "why": "it burns the wall", "instead": "read partial output"},
        "hand-tune constants to chase a metric",
    ])
    text = _guard(_Plan(p))._obligations_text()
    assert "re-run the same command with a bigger timeout" in text
    assert "hand-tune constants to chase a metric" in text


def test_non_due_steps_still_contribute_their_acceptance():
    """A pending step's criteria are obligations too, not only the doing ones."""
    p = _plan(steps=[
        {"id": 1, "title": "a", "status": "done", "acceptance": ["a1"]},
        {"id": 2, "title": "b", "status": "pending", "acceptance": ["b1"]},
    ])
    text = _guard(_Plan(p))._obligations_text()
    assert "a1" in text and "b1" in text


def test_no_active_plan_fails_open_to_task_text():
    """No plan => fall back to the task text; must not silently empty the set."""
    assert _guard(_Plan(None))._obligations_text() == BOSS_TEXT
    assert _guard(None)._obligations_text() == BOSS_TEXT


def test_empty_plan_rendering_fails_open_to_task_text():
    """A plan with no content must not disable the gate by rendering empty."""
    g = _guard(_Plan({"title": "", "thinking": "", "forbidden": [], "steps": []}))
    assert g._obligations_text() == BOSS_TEXT


def test_broken_plan_manager_fails_open():
    """A raising plan manager must not take the guard down with it."""
    class Boom:
        def get_active(self):
            raise RuntimeError("disk gone")

    assert _guard(Boom())._obligations_text() == BOSS_TEXT


def test_grounding_axis_still_reads_the_original_task_text():
    """`deliverable_reading.source` quotes are grounded against the TASK TEXT.

    This is the axis split: obligations read the plan, but a quote must occur in
    the task's own words. Conflating them would let a plan rewording make a
    legal task quote unquotable (and vice versa).
    """
    g = _guard(_Plan(_plan()))
    assert g._task_text() == BOSS_TEXT
    assert "build the deliverable" not in g._task_text()

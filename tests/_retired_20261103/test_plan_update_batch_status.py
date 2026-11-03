"""Batch status must fail loudly, never silently no-op.

Measured defect: a batch update whose ``status`` was ``"skip"`` (the
single-step spelling) was silently skipped by the handler, so the batch
returned a summary with NOTHING applied and the plan state did not move.
An invalid enum value must come back as an explicit ERROR naming the
expected values, and a batch that applied zero updates must not read as
success either.
"""
import tempfile

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.tools.plan_update import PlanUpdateTool


def _step_status(plan, sid):
    steps = plan.get_active().get("steps", [])
    for s in steps:
        if s.get("id") == sid or (steps.index(s) + 1) == sid:
            return s.get("status")
    return None


def _tool():
    d = tempfile.mkdtemp()
    plan = TaskPlan(d)
    plan.create(title="t", steps=[{"title": "s1"}, {"title": "s2"}],
                session_id="t-status")
    return plan, PlanUpdateTool(plan)


def test_unknown_status_is_loud_error_not_silent_drop():
    plan, tool = _tool()
    out = tool.execute(action="batch", updates=[
        {"step_id": 1, "status": "skip", "notes": "superseded"}])
    assert out.startswith("ERROR"), out[:200]
    assert "skipped" in out  # names the correct spelling
    assert _step_status(plan, 1) != "skipped"


def test_valid_batch_still_applies():
    plan, tool = _tool()
    out = tool.execute(action="batch", updates=[
        {"step_id": 1, "status": "skipped", "notes": "n/a"},
        {"step_id": 2, "status": "done", "notes": "ok"},
    ])
    assert not out.startswith("ERROR"), out[:200]
    assert _step_status(plan, 1) == "skipped"
    assert _step_status(plan, 2) == "done"


def test_missing_step_id_is_error_not_drop():
    plan, tool = _tool()
    out = tool.execute(action="batch", updates=[
        {"step_id": None, "status": "done"},
    ])
    assert out.startswith("ERROR"), out[:200]
    assert "step_id" in out


def test_partial_batch_applies_valid_rows_before_error():
    """A row with a good step_id and a bad status errors WITHOUT applying
    that row — fail-loud means the caller retries with corrected values,
    so the batch must not half-apply when a row is malformed."""
    plan, tool = _tool()
    out = tool.execute(action="batch", updates=[
        {"step_id": 1, "status": "skipped", "notes": "n/a"},
        {"step_id": 2, "status": "skip", "notes": "bad"},
    ])
    assert out.startswith("ERROR"), out[:200]
    # row 1 was applied before the malformed row aborted the batch
    assert _step_status(plan, 1) == "skipped"
    assert _step_status(plan, 2) != "skipped"

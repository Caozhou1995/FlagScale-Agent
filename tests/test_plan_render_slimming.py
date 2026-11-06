"""Slim plan rendering: closed steps cost one line, live steps carry acceptance.

Measured motivation (session b3f101eb, a 32-step all-done plan):
  * per-turn plan context: 205,422 chars, of which 199,157 were the notes of
    CLOSED steps (100% of the notes budget) while a live step's acceptance
    criteria were never rendered at all;
  * one plan_update receipt: 254,400 chars, because every mutating branch
    ended in `return self._plan.summary()`.

The slimming has to keep three things true, and these tests pin them:
  1. a closed step still shows its EVIDENCE (never dropped: it is the record
     of what was checked before the close);
  2. a live step shows its ACCEPTANCE (what it will be judged against);
  3. closed-step NOTES are not lost — plan_status(step_id=N) returns them
     verbatim, so dropping them from the render is a move, not a delete.
"""

import pytest

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.tools.plan_status import PlanStatusTool
from flagscale_agent.react.tools.plan_update import PlanUpdateTool


@pytest.fixture
def tp(tmp_path):
    return TaskPlan(str(tmp_path))


def _closed(tp, step_id, notes="", verification=None):
    tp.update_step(step_id, "done", notes, verification=verification or [])


# ── 1. closed step: one line, evidence kept, notes dropped ──────────────────


def test_closed_step_renders_one_line_with_evidence(tp):
    tp.create("T", [{"title": "Alpha", "acceptance": ["crit-1", "crit-2"]}])
    _closed(tp, 1, notes="x" * 500, verification=["Proof A"])
    out = tp.summary()
    assert "Proof A" in out
    # Acceptance and notes of a CLOSED step are not reprinted.
    assert "crit-1" not in out
    assert "x" * 50 not in out
    # One line for the step itself, not a multi-line block.
    step_lines = [l for l in out.split("\n") if "Alpha" in l]
    assert len(step_lines) == 1


def test_skipped_step_also_collapses(tp):
    tp.create("T", ["A", "B"])
    tp.skip_step(1, "not needed")
    out = tp.summary()
    assert "not needed" not in out
    assert out.count("A") >= 1


def test_closed_step_keeps_only_first_evidence_plus_count(tp):
    tp.create("T", ["A"])
    _closed(tp, 1, verification=["ev-one", "ev-two", "ev-three", "ev-four"])
    out = tp.summary()
    assert "ev-one" in out
    assert "+3 more" in out


def test_long_evidence_is_truncated_with_marker(tp):
    tp.create("T", ["A"])
    _closed(tp, 1, verification=["e" * 400])
    out = tp.summary()
    assert "…" in out
    assert "e" * 400 not in out


# ── 2. live step: acceptance IS rendered ───────────────────────────────────


def test_live_step_keeps_acceptance_in_full_render(tp):
    tp.create("T", [{"title": "Live", "acceptance": ["must-be-shown"]}])
    assert "must-be-shown" in tp.summary()


def test_doing_step_keeps_acceptance_and_notes(tp):
    tp.create("T", [{"title": "Live", "acceptance": ["crit-live"]}])
    tp.update_step(1, "doing", notes="working note")
    out = tp.summary()
    assert "crit-live" in out
    assert "working note" in out


def test_full_mode_is_the_untrimmed_render(tp):
    tp.create("T", [{"title": "Alpha", "acceptance": ["crit-1"]}])
    _closed(tp, 1, notes="the long note", verification=["Proof A"])
    full = tp.summary(compact=False)
    assert "crit-1" in full          # acceptance back
    assert "the long note" in full   # notes back
    assert "Proof A" in full         # evidence still there
    assert len(full) > len(tp.summary())


# ── 3. notes are recoverable, not deleted ──────────────────────────────────


def test_step_id_reads_full_notes_back(tp):
    tp.create("T", [{"title": "Alpha", "acceptance": ["crit-1"]}])
    long_note = "recoverable note " * 30
    _closed(tp, 1, notes=long_note, verification=["Proof A"])
    # The render drops it...
    assert long_note not in tp.summary()
    # ...the recovery path returns it verbatim.
    one = tp.summary(step_id=1)
    assert long_note.strip() in one
    assert "crit-1" in one
    assert "Proof A" in one


def test_plan_status_tool_threads_step_id(tp):
    tp.create("T", [{"title": "Alpha"}])
    _closed(tp, 1, notes="tool-recoverable")
    tool = PlanStatusTool(tp)
    assert "tool-recoverable" not in tool.execute()
    assert "tool-recoverable" in tool.execute(step_id=1)


def test_plan_status_rejects_bad_step_id(tp):
    tp.create("T", ["A"])
    assert "ERROR" in PlanStatusTool(tp).execute(step_id="nope")


def test_plan_status_unknown_step_is_an_error_not_a_crash(tp):
    tp.create("T", ["A"])
    assert "ERROR" in PlanStatusTool(tp).execute(step_id=99)


# ── 4. receipt is a delta, and still answers the caller ────────────────────


def test_step_done_receipt_is_small_and_reports_progress(tp):
    tp.create("T", ["A", "B"])
    up = PlanUpdateTool(tp)
    out = up.execute(action="step_done", step_id=1, verification=["Proof A"])
    assert len(out) < 600
    assert "Progress: 1/2" in out
    assert "Proof A" in out
    assert "Next: Step 2" in out


def test_receipt_does_not_reprint_other_steps_notes(tp):
    tp.create("T", ["A", "B"])
    tp.update_step(2, "doing", notes="other-step-secret-note")
    out = PlanUpdateTool(tp).execute(action="step_done", step_id=1)
    assert "other-step-secret-note" not in out


def test_update_acceptance_echoes_what_it_stored(tp):
    tp.create("T", ["A"])
    out = PlanUpdateTool(tp).execute(
        action="update_acceptance", step_id=1, acceptance=["A1", "A2"]
    )
    assert "A1" in out and "A2" in out


def test_set_thinking_still_shows_the_whole_model(tp):
    """A delta that omitted the model would answer a different question."""
    tp.create("T", ["A"])
    out = PlanUpdateTool(tp).execute(
        action="set_thinking", thinking="bottleneck is IO"
    )
    assert "bottleneck is IO" in out
    assert "Model recorded" in out


def test_complete_receipt_reports_no_open_steps(tp):
    tp.create("T", ["A"])
    up = PlanUpdateTool(tp)
    up.execute(action="step_done", step_id=1)
    out = up.execute(action="complete")
    assert "No open steps left" in out or "Progress: 1/1" in out


# ── 5. the per-turn context: the 97% path ─────────────────────────────────


def test_per_turn_context_drops_closed_notes_and_keeps_live_acceptance():
    """The renderer agent.py actually calls each turn."""
    from flagscale_agent.react.agent import WorkerAgent

    import tempfile

    tp = TaskPlan(tempfile.mkdtemp())
    tp.create(
        "T",
        [
            {"title": "Done one", "acceptance": ["closed-crit"]},
            {"title": "Live one", "acceptance": ["live-crit"]},
        ],
    )
    tp.update_step(1, "done", "closed step note " * 20, verification=["Proof A"])
    agent = WorkerAgent.__new__(WorkerAgent)
    agent.task_plan = tp
    ctx = agent._build_plan_context()
    assert "closed step note" not in ctx   # notes of a closed step are gone
    assert "Proof A" in ctx                # its evidence survives
    assert "live-crit" in ctx              # live acceptance is present

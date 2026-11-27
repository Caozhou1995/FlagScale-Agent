"""Standing acceptance must survive its step closing — on BOTH render surfaces.

prop_7b57bc84. The failure this pins (2026-11-27, dna-assembly):

  agent.py `_build_plan_context` injects a step's `acceptance` ONLY while the
  step is live (status pending/doing). A closed step renders identity +
  evidence and DROPS acceptance — deliberate slimming (a 32-step plan cost
  199KB/turn because of closed-step commentary). The Frame step is worked
  FIRST and closed during planning, so any rule installed in its `acceptance`
  was gone by the time the agent acted. Meanwhile the agent's own forbidden
  list re-renders at the top of every turn. Net: the self-authored prohibition
  outlived the constraint meant to bound it, and the agent cited its own ban
  (:1329, :1913) to refuse the task's named authority — 8/8 primers with
  clamp=0, twice.

  Fix: a `standing_acceptance` key holding the subset that must keep
  rendering after the close. Cost stays proportional to that subset.

Both surfaces are pinned because they are SEPARATE code — agent.py keeps its
own inline copy of the closed-step line and does not call plan.py's
`_closed_step_line`. A test that only exercised one would pass while the live
per-turn path stayed broken, which is precisely how this bug survived a
previous round of "fixes".
"""
import os
import sys
import tempfile
import types

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from flagscale_agent.react.plan import TaskPlan, _norm_standing  # noqa: E402
from flagscale_agent.react.tools.plan_create import (  # noqa: E402
    PlanCreateTool,
    _FRAME_STEP_ACCEPTANCE,
    _frame_standing_acceptance,
)

MARK = "A forbidden entry binds only what YOU control."


def _plan():
    d = tempfile.mkdtemp()
    tp = TaskPlan(plan_dir=d)
    PlanCreateTool(tp, session_id="probe").execute(
        title="probe",
        steps=[{"title": "s1"}],
        forbidden=[{"action": "do not add X", "why": "guess", "instead": "Y"}],
    )
    return tp


def _agent_context(tp):
    """The PRODUCTION per-turn path (agent.py), not plan.py's compact one."""
    from flagscale_agent.react.agent import WorkerAgent

    stub = types.SimpleNamespace(task_plan=tp)
    return WorkerAgent._build_plan_context.__get__(stub, object)()


# ── the marker set is minimal and correct ────────────────────────────────


def test_only_the_forbidden_bound_is_marked_standing():
    """Marking all 7 would re-create the 199KB/turn blowup."""
    standing = _frame_standing_acceptance()
    assert len(standing) == 1, f"expected exactly 1 standing clause, got {len(standing)}"
    assert standing[0].startswith(MARK)


def test_marker_matches_by_text_not_index():
    """An index would silently mark the wrong clause after the next insertion."""
    src = open(
        os.path.join(REPO, "flagscale_agent", "react", "tools", "plan_create.py"),
        encoding="utf-8",
    ).read()
    assert "_FRAME_STANDING_MARKERS" in src
    assert "startswith(m)" in src, "marker must be matched by prefix, not position"


# ── the key actually persists (create() whitelists step keys) ────────────


def test_standing_key_survives_create_and_yaml_roundtrip():
    tp = _plan()
    frame = tp.get_active()["steps"][0]
    assert frame.get("standing_acceptance"), (
        "create() rebuilds step_dict from an explicit whitelist — a key omitted "
        "there is silently dropped, so the marker would never reach the renderer"
    )
    # Re-read from disk: the plan is persisted as YAML.
    reloaded = tp._load(tp.get_active()["id"])
    assert reloaded["steps"][0].get("standing_acceptance"), "lost across YAML round-trip"


def test_plain_steps_do_not_gain_standing_entries():
    tp = _plan()
    assert [len(s.get("standing_acceptance") or []) for s in tp.get_active()["steps"]] == [
        1, 0, 0, 0,
    ], "only the Frame step declares standing clauses"


# ── surface 1: agent.py per-turn context ─────────────────────────────────


def test_agent_context_keeps_standing_after_frame_closes():
    tp = _plan()
    assert MARK in _agent_context(tp), "clause missing while Frame is live"
    tp.update_step(tp.get_active()["steps"][0]["id"], status="done", verification=["framed"])
    assert MARK in _agent_context(tp), (
        "THE regression: standing clause vanished when Frame closed"
    )


def test_agent_context_still_drops_the_bulk():
    """Cost control must survive the fix, or we re-create the 199KB/turn blowup."""
    tp = _plan()
    tp.update_step(tp.get_active()["steps"][0]["id"], status="done", verification=["framed"])
    ctx = _agent_context(tp)
    for dropped in (
        "FRAMING. State in one line",
        "FIRST CUT AND CONTEXT",
        "Acceptance-facet manifest",
        "Reading ledger",
    ):
        assert dropped not in ctx, f"non-standing clause re-rendered after close: {dropped!r}"


# ── surface 2: plan.py compact status ────────────────────────────────────


def test_compact_status_keeps_standing_after_frame_closes():
    """The OTHER renderer — it is separate code and can regress independently."""
    tp = _plan()
    tp.update_step(tp.get_active()["steps"][0]["id"], status="done", verification=["framed"])
    compact = tp._format_plan(tp.get_active(), compact=True)
    assert MARK in compact, "plan_status surface lost the standing clause"


# ── the helper's contract ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, []),
        ([], []),
        ("a clause", ["a clause"]),
        (["a", "a", "  ", "b"], ["a", "b"]),
        (["  padded  "], ["padded"]),
        ([1, {"x": 1}, None, "ok"], ["ok"]),
        ({"not": "a list"}, []),
    ],
)
def test_norm_standing_degrades_safely(value, expected):
    """Junk from an LLM tool call must not break plan creation."""
    assert _norm_standing(value) == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

"""ReviewerAuditGuard — the audit beat at completion declarations.

Two phases, and both are load-bearing:
  * check_post  (declaration landed) snapshots the transcript and dispatches a
    reviewer. It must NEVER return a verdict: post verdicts cannot block
    (kernel.py), and returning one would silently do nothing while looking
    like it worked.
  * check_pre   (next tool call) blocks ONCE when the reviewer has produced
    findings nobody has read. This is the "read the reviewer's report
    promptly" requirement expressed in the only mechanism that can enforce it.

Plus the fail-silent contract: no session dir / no launcher / thin budget /
errored declaration -> silence, so the guard can never break tool execution.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.reviewer_audit import (
    AUDIT_DIRNAME,
    AUDIT_STATE_FILENAME,
    FINDINGS_FILENAME,
    SNAPSHOT_FILENAME,
    ReviewerAuditGuard,
    make_spawn_fn,
)


def _ctx(tmp: Path, tool="plan_update", args=None, result="Plan updated.", messages=None):
    return GuardContext(
        tool_name=tool,
        tool_args=args if args is not None else {"action": "step_done", "step_id": 1},
        tool_result=result,
        messages=messages if messages is not None else [
            {"role": "system", "content": "S1 locate the producer. G1 bytes not model."},
            {"role": "assistant", "content": "I will re-serialize the document."},
        ],
    )


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return "reviewer pid=1234 session=auditor"


def _guard(tmp: Path, spawn=None, stats=None, enabled=True):
    return ReviewerAuditGuard(
        session_dir_fn=lambda: str(tmp),
        spawn_fn=spawn or _Recorder(),
        stats_fn=stats,
        enabled_fn=lambda: enabled,
    )


# ── check_post: dispatch, and it must not be a verdict ──────────────────────

def test_step_done_snapshots_transcript_and_dispatches(tmp_path):
    spawn = _Recorder()
    g = _guard(tmp_path, spawn=spawn)
    assert g.check_post(_ctx(tmp_path)) is None, "post phase must never block"
    snap = tmp_path / AUDIT_DIRNAME / SNAPSHOT_FILENAME
    assert snap.is_file()
    rows = [json.loads(l) for l in snap.read_text().splitlines() if l.strip()]
    assert rows[0]["role"] == "system"
    assert "G1 bytes not model" in rows[0]["content"], (
        "the doctrine must be IN the file the reviewer reads, not recalled")
    assert len(spawn.calls) == 1
    assert spawn.calls[0]["goal"].count("VIOLATION:") == 1


def test_snapshot_is_bounded(tmp_path):
    g = _guard(tmp_path)
    big = "A" * 50000
    g.check_post(_ctx(tmp_path, messages=[{"role": "assistant", "content": big}]))
    snap = (tmp_path / AUDIT_DIRNAME / SNAPSHOT_FILENAME).read_text()
    assert len(snap) < 5000, "an auditor needs intent+outcome, not megabytes"


def test_non_completion_does_not_dispatch(tmp_path):
    spawn = _Recorder()
    g = _guard(tmp_path, spawn=spawn)
    g.check_post(_ctx(tmp_path, args={"action": "step_doing", "step_id": 1}))
    assert spawn.calls == []


def test_errored_declaration_does_not_dispatch(tmp_path):
    spawn = _Recorder()
    g = _guard(tmp_path, spawn=spawn)
    g.check_post(_ctx(tmp_path, result="ERROR: bad"))
    assert spawn.calls == []


def test_thin_budget_suppresses_audit(tmp_path):
    spawn = _Recorder()
    thin = lambda: {"elapsed": 850.0, "budget": 900.0, "remaining": 50.0, "pct": 94.0}
    g = _guard(tmp_path, spawn=spawn, stats=thin)
    g.check_post(_ctx(tmp_path))
    assert spawn.calls == [], "the audit must never eat the tail of the run"


def test_disabled_inside_a_reviewer(tmp_path):
    spawn = _Recorder()
    g = _guard(tmp_path, spawn=spawn, enabled=False)
    g.check_post(_ctx(tmp_path))
    g.check_pre(_ctx(tmp_path, tool="shell"))
    assert spawn.calls == []
    assert not (tmp_path / AUDIT_DIRNAME).exists(), "no audit of an audit"


# ── check_pre: surface findings exactly once ────────────────────────────────

def _arm(tmp_path, g, findings_text):
    g.check_post(_ctx(tmp_path))                     # dispatch (writes state)
    d = tmp_path / AUDIT_DIRNAME
    (d / FINDINGS_FILENAME).write_text(findings_text, encoding="utf-8")
    st = json.loads((d / AUDIT_STATE_FILENAME).read_text())
    st["findings_mtime"] = 0.0                        # reviewer wrote it after
    (d / AUDIT_STATE_FILENAME).write_text(json.dumps(st))


def test_unread_findings_block_the_next_call_once(tmp_path):
    g = _guard(tmp_path)
    report = ("VIOLATION: G1 requires reading the delivered bytes.\n"
              "EVIDENCE: record 1 — 'I will re-serialize the document.'\n"
              "FIX: read the file back and compare it byte for byte.\n")
    _arm(tmp_path, g, report)
    v = g.check_pre(_ctx(tmp_path, tool="shell"))
    assert v is not None and v.action == "block"
    assert "G1 requires reading the delivered bytes" in v.message
    assert g.check_pre(_ctx(tmp_path, tool="shell")) is None, (
        "one report must not block the same call twice")


def test_clean_report_does_not_block(tmp_path):
    g = _guard(tmp_path)
    _arm(tmp_path, g, "NO_VIOLATIONS")
    assert g.check_pre(_ctx(tmp_path, tool="shell")) is None


def test_no_findings_file_does_not_block(tmp_path):
    g = _guard(tmp_path)
    g.check_post(_ctx(tmp_path))
    assert g.check_pre(_ctx(tmp_path, tool="shell")) is None


def test_override_demands_adjudication_not_a_bare_retry(tmp_path):
    g = _guard(tmp_path)
    _arm(tmp_path, g, "VIOLATION: x\nEVIDENCE: y\nFIX: z\n")
    g.check_pre(_ctx(tmp_path, tool="shell"))
    assert not g.accept_override("ok fine", _ctx(tmp_path)), "bare retry released it"
    assert g.accept_override(
        "accepting the reviewer finding and re-measuring the delivered bytes",
        _ctx(tmp_path))


# ── fail-silent: a guard must never break execution ────────────────────────

def test_no_session_dir_is_silent():
    g = ReviewerAuditGuard(session_dir_fn=lambda: None,
                           spawn_fn=_Recorder(), enabled_fn=lambda: True)
    assert g.check_post(_ctx(Path("/tmp"))) is None
    assert g.check_pre(_ctx(Path("/tmp"))) is None


def test_launcher_error_is_silent(tmp_path):
    def boom(**_kw):
        return "ERROR: reviewer launch failed"
    g = _guard(tmp_path, spawn=boom)
    assert g.check_post(_ctx(tmp_path)) is None
    assert g.check_pre(_ctx(tmp_path, tool="shell")) is None


def test_snapshot_total_bytes_are_capped(tmp_path):
    """A real agent transcript is ~3700 records / 550KB (measured on a live
    trial). Shipping that into a reviewer's context spends its whole wall on
    reading and buries the decision lines. The cap must hold on the far end —
    the FILE size, not a count of records."""
    g = _guard(tmp_path)
    msgs = [{"role": "system", "content": "G1 read the bytes."}]
    msgs += [{"role": "assistant", "content": f"record {i} " + "x" * 1500}
             for i in range(400)]
    g.check_post(_ctx(tmp_path, messages=msgs))
    snap = tmp_path / AUDIT_DIRNAME / SNAPSHOT_FILENAME
    assert snap.is_file()
    assert snap.stat().st_size <= 95_000, (
        f"snapshot {snap.stat().st_size}B exceeds the readable cap")
    text = snap.read_text()
    assert "G1 read the bytes." in text, "doctrine must survive the cap"
    assert "record 399" in text, "the NEWEST records must survive the cap"
    assert "omitted for length" in text, "the omission must be explicit"


def test_continuity_reuses_the_same_reviewer_session(tmp_path):
    """Audit N+1 must continue audit N's reviewer, not start a fresh one."""
    import json as _json
    from flagscale_agent.react.guard.reviewer_audit import make_spawn_fn
    seen = {}

    def shim(**kw):
        seen.update(kw)
        return "reviewer pid=1 session=auditor"

    fn = make_spawn_fn(session_dir_fn=lambda: str(tmp_path), agent_bin="/bin/true")
    # first launch: no session yet -> full goal is passed as the prompt
    assert "reviewer" in fn(goal="G", deadline_minutes=1)
    assert not (tmp_path / AUDIT_DIRNAME / "reviewer" / "auditor" / "resume.prompt").exists()
    (tmp_path / AUDIT_DIRNAME / "reviewer" / "auditor").mkdir(parents=True)
    (tmp_path / AUDIT_DIRNAME / "reviewer" / "auditor" / "conversation.json").write_text("{}")
    assert "reviewer" in fn(resume_task_id="auditor", message="M2")
    rp = tmp_path / AUDIT_DIRNAME / "reviewer" / "auditor" / "resume.prompt"
    assert rp.read_text() == "M2", "continuity needs the resume pointer on disk"

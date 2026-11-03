import json

from tests.test_reviewer_audit_guard import _ctx, _guard


def _state(tmp_path):
    return json.loads(
        (tmp_path / "audit" / "audit_state.json").read_text(encoding="utf-8"))


def test_dispatch_records_the_snapshot_size(tmp_path):
    """snapshot_records must record what the reviewer received.

    The original wrote `state.get("snapshot_records")` — reading the field it
    was about to write — so the value round-tripped null forever (production
    seq=7 was still null). A reviewer/auditor reconciling how much transcript it
    saw had nothing to compare against.
    """
    g = _guard(tmp_path)
    ctx = _ctx(tmp_path)
    g.check_post(ctx)
    st = _state(tmp_path)
    n_msgs = len(ctx.messages)
    assert st["snapshot_records"] == n_msgs, st
    assert st["snapshot_records"] > 0


def test_snapshot_records_is_not_a_round_trip_of_itself(tmp_path):
    # A pre-seeded bogus value must be OVERWRITTEN by the real count, not
    # echoed back — that is the exact defect being closed.
    d = tmp_path / "audit"
    d.mkdir(parents=True)
    (d / "audit_state.json").write_text(
        json.dumps({"snapshot_records": 999}), encoding="utf-8")
    g = _guard(tmp_path)
    g.check_post(_ctx(tmp_path, messages=[{"role": "system", "content": "S"}]))
    st = _state(tmp_path)
    assert st["snapshot_records"] == 1, st


def test_time_tick_dispatch_also_records_the_size(tmp_path):
    # Both dispatch sites (declaration hook + the 40% time tick) must record it;
    # fixing only one leaves the other path null exactly as before.
    stats = lambda: {"budget": 1000.0, "remaining": 500.0}  # noqa: E731
    g = _guard(tmp_path, stats=stats)
    g._maybe_tick(_ctx(tmp_path, tool="shell", args={"command": "ls"}))
    st = _state(tmp_path)
    assert st["seq"] == 1
    assert st["snapshot_records"] > 0, st

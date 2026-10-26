"""Regression: a progress rung must not block the very act that answers it.

WHY (measured, 2026-10-27, two independent episodes): FirstE2eRunGuard's 25%
rung asks "is a minimal usable deliverable already written this turn?". Its
`check_pre` ran BEFORE any tool executed, so a `write_file` — the exact landing
the rung demands — was blocked by the rung itself, and the agent could only get
past it by pasting a prose reason into `_override_reason` (one wasted
round-trip per occurrence; observed directly in a review worker writing its own
report deliverable, and in an earlier trial's plan_create deadlock).

The fix reads the kernel's EXECUTION ledger (ctx.recent_tool_history), which the
kernel populates only with calls that really ran, so:
  * a rung whose question the ledger already answers is released, not asked;
  * the call that IS the landing passes through;
  * a BLOCKED or ERRORED write does not count (a promise is not a write).

The 75% rung is deliberately NOT covered by this release: its question is
"have you RUN the thing", which a write cannot answer.
"""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.first_e2e_run import FirstE2eRunGuard


def _stats(pct: float) -> dict:
    return {"elapsed": 300.0, "remaining": 700.0, "pct": pct}


def _guard(pct: float = 30.0) -> FirstE2eRunGuard:
    return FirstE2eRunGuard(lambda: _stats(pct))


def test_write_call_itself_passes_the_25pct_rung():
    """The landing the rung asks for must not be blocked by that rung."""
    v = _guard().check_pre(GuardContext(tool_name="write_file"))
    assert v is None, f"write_file should pass through, got {v.reason}"


def test_edit_call_itself_passes_the_25pct_rung():
    v = _guard().check_pre(GuardContext(tool_name="edit_file"))
    assert v is None, f"edit_file should pass through, got {v.reason}"


def test_non_write_still_blocked_when_nothing_landed():
    """The rung keeps its teeth: no landing yet -> exploration is blocked."""
    v = _guard().check_pre(GuardContext(tool_name="shell"))
    assert v is not None and v.reason == "first_e2e_run_25pct"


def test_non_write_passes_after_a_landed_write():
    hist = [{"name": "write_file", "exec_error": None}]
    v = _guard().check_pre(
        GuardContext(tool_name="shell", recent_tool_history=hist)
    )
    assert v is None, f"rung should be released by a real landing, got {v.reason}"


def test_errored_write_does_not_release():
    """`exec_error` marks a call that did not land — it must not count."""
    hist = [{"name": "write_file", "exec_error": "ENOENT"}]
    v = _guard().check_pre(
        GuardContext(tool_name="shell", recent_tool_history=hist)
    )
    assert v is not None and v.reason == "first_e2e_run_25pct"


def test_non_dict_ledger_entries_do_not_crash():
    v = _guard().check_pre(
        GuardContext(tool_name="shell", recent_tool_history=["write_file", None])
    )
    assert v is not None and v.reason == "first_e2e_run_25pct"


def test_75pct_rung_not_released_by_a_write():
    """75% asks 'have you RUN it'; a write cannot answer that question."""
    hist = [{"name": "write_file", "exec_error": None}]
    v = _guard(80.0).check_pre(
        GuardContext(tool_name="shell", recent_tool_history=hist)
    )
    assert v is not None and v.reason == "first_e2e_run_75pct"


def test_prereq_tools_still_pass_every_rung():
    """plan_update/memory_write can never land the deliverable, so they are
    never gated (the original deadlock this list exists to prevent)."""
    for tool in ("plan_create", "plan_update", "memory_write"):
        v = _guard(80.0).check_pre(GuardContext(tool_name=tool))
        assert v is None, f"{tool} should pass through, got {v.reason}"

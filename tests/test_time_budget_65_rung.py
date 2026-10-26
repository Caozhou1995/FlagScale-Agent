"""65% rung: pre-completion falsification checkpoint.

Measured basis: 313/578 historical losses (54%) ended BEFORE the 75% rung with a
self-reported TASK_COMPLETE — the 75% "final check" entry point was unreachable
for over half of all losses. The 65% rung reaches those agents while they are
still working. Red phase: reverting either the threshold tuple or the message
branch makes these tests fail.
"""
from flagscale_agent.react.guard.time_budget import TimeBudgetGuard
from flagscale_agent.react.guard import GuardContext


def _msg(thr, pct):
    g = TimeBudgetGuard(lambda: None)
    return g._message(thr, {"elapsed": 18 * pct, "remaining": 1800 - 18 * pct, "pct": float(pct)})


def test_thresholds_include_65():
    assert 65 in TimeBudgetGuard._THRESHOLDS
    # ordering still high->low so the most severe pending threshold wins
    assert TimeBudgetGuard._THRESHOLDS == (100, 90, 80, 75, 65, 50, 25)


def test_65_message_demands_falsification_before_completion():
    m = _msg(65, 66)
    assert "PROVE it" in m, "65 rung must demand proof before declaring completion"
    assert "decisive check" in m, "65 rung must name the verifier's own decisive check"
    assert "self-check" in m.lower(), "65 rung must warn that an artifact agrees with its own generator"


def test_65_message_covers_the_not_done_branch():
    m = _msg(65, 66)
    assert "NOT nearly done" in m or "re-planning" in m, "65 rung must guide the still-working agent too"


def test_older_rungs_unchanged_by_65():
    # form contract: inserting 65 must not alter the 75/50 texts
    assert "Most of the budget is spent" in _msg(75, 77)
    assert "PROVE it" not in _msg(75, 77)
    assert "HEALTH CHECK" in _msg(50, 50)
    assert "PROVE it" not in _msg(50, 50)


def test_65_fires_between_50_and_75():
    g = TimeBudgetGuard(lambda: None)
    ctx = GuardContext()
    ctx.tool_name = "read_file"
    for pct in (50.0, 66.0):
        g._stats_fn = lambda p=pct: {"elapsed": 18 * p, "remaining": 1800 - 18 * p, "pct": p}
        g.check_post(ctx)
    assert 65 in g._fired, "65 rung must fire when pct crosses it"
    assert 50 in g._fired and 75 not in g._fired, "65 fires before 75, not after"

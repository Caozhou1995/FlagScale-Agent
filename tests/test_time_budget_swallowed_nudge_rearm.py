"""Regression: a generic (non attach_on_block) post-inject swallowed by a
concurrent block must be RE-ARMABLE — the guard's consumed one-shot state is
handed back via block_not_surfaced(), so the nudge fires on a later resolve.

Observed failure this guards against: a raman trial ran to 84% of its wall with
only the 25% pacing advisory ever delivered — every 50/75/80/90% nudge was
either consumed by a check_post verdict that a same-resolve block suppressed,
or consumed by a check_pre block the registry outranked. _pending_block_fired
covered only the block channel and had no registry call site at all.
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardRegistry, GuardVerdict
from flagscale_agent.react.guard.time_budget import TimeBudgetGuard


class _HardBlock(Guard):
    """A high-priority non-overridable block that wins every resolve."""

    name = "hard_block"
    priority = 5

    def check_pre(self, ctx):
        if ctx.tool_name == "shell":
            return GuardVerdict.block(
                message="hard stop", reason="hard_block", overridable=False
            )
        return None


class _PostBlock(Guard):
    """A post-phase block that wins the same resolve as a generic inject."""

    name = "post_block"
    priority = 5

    def check_post(self, ctx):
        if ctx.tool_name == "shell":
            return GuardVerdict.block(
                message="hard stop", reason="post_block",
                category="post_block", overridable=False
            )
        return None


def _ctx(tool: str = "shell", override: str | None = None) -> GuardContext:
    return GuardContext(tool_name=tool, override_reason=override)


def _stats(pct: float) -> dict:
    return {"pct": pct, "elapsed": pct, "budget": 100.0, "remaining": 100 - pct}


def test_swallowed_post_inject_thresholds_are_returned():
    tb = TimeBudgetGuard(stats_fn=lambda: _stats(55.0))
    reg = GuardRegistry()
    reg.register(tb)
    reg.register(_HardBlock())

    # The tool call executes (post-phase). TimeBudget builds the 50% advisory
    # (consuming 50 into _fired). With no concurrent block the inject surfaces
    # directly — control case.
    post = reg.check_post(_ctx("shell"))
    assert post is not None and post.action == "inject"
    assert 50 in tb._fired
    tb.reset_turn()

    # Now the suppression path: a check_pre block wins the SAME resolve in a
    # real turn, so rebuild the state — the advisory is consumed (pending) but
    # suppressed. block_not_surfaced() must hand it back.
    tb._pending_inject_fired.add(50)
    tb._fired.add(50)
    tb.block_not_surfaced()
    assert 50 not in tb._fired
    assert tb._pending_inject_fired == set()


def test_rearmed_nudge_fires_on_next_call():
    tb = TimeBudgetGuard(stats_fn=lambda: _stats(55.0))
    reg = GuardRegistry()
    reg.register(tb)
    reg.register(_HardBlock())

    reg.check_post(_ctx("shell"))  # advisory consumed then suppressed
    tb._pending_inject_fired.add(50)
    tb.block_not_surfaced()  # registry returned the consumed threshold

    # Without the hard block now, the SAME pct must still yield the 50% nudge.
    v = tb.check_post(_ctx("read_file"))
    assert v is not None and v.action == "inject"
    assert "time_budget_50pct" in v.reason


def test_block_channel_still_returns_via_block_not_surfaced():
    tb = TimeBudgetGuard(stats_fn=lambda: _stats(92.0))
    reg = GuardRegistry()
    reg.register(tb)
    reg.register(_HardBlock())

    # At 92% the critical gate (a block) is outranked by _HardBlock's block.
    pre = reg.check_pre(_ctx("read_file"))
    assert pre is not None and pre.action == "block"
    assert 90 in tb._fired

    tb.block_not_surfaced()
    assert 90 not in tb._fired

    # With the rival block gone, the gate must be able to fire again.
    v = tb.check_pre(_ctx("read_file"))
    assert v is not None and v.action == "block" and "CRITICAL" in v.message



def test_registry_notifies_suppressed_inject_guard():
    """The call site this bug lived at: when a block wins a resolve that also
    carried a generic inject, the registry must call block_not_surfaced() on
    the inject's guard — the previously-dead path (grep showed zero call
    sites). Simulate via a TimeBudgetGuard subclass with a notified flag."""

    class _Tracking(TimeBudgetGuard):
        notified = False

        def block_not_surfaced(self):
            _Tracking.notified = True
            super().block_not_surfaced()

    tb = _Tracking(stats_fn=lambda: _stats(55.0))
    reg = GuardRegistry()
    reg.register(tb)
    reg.register(_PostBlock())

    # A generic inject is built post-phase while a post-phase block wins the
    # SAME resolve → the inject is suppressed; the guard must be notified so
    # its consumed threshold is handed back.
    verdict = reg.check_post(_ctx("shell"))
    assert verdict is not None
    assert _Tracking.notified, (
        "registry must call block_not_surfaced() on suppressed-inject guards"
    )

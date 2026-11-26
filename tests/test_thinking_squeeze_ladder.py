"""Half-life squeeze ladder (user directive, 2026-11-22).

The escalation used to jump STRAIGHT to THINKING_SQUEEZE_TOKENS (2048) on its
first arm, so a model that needed only a nudge lost 8x of its configured
thinking budget in one step. Measured on write-compressor trial
2026-11-22__19-19-50/8vCLyeT: all 17 runtime-cap events printed `↑0 ↓0` — the
2048 client-side cap aborted every forced stream BEFORE the mandatory
`tool_choice={"type":"any"}` call could be emitted, so the turn ended with zero
tool calls and the kernel re-armed forever.

The ladder must walk DOWN from the configured thinking_budget — b/2, b/4, b/8 —
and stop at the floor, never below it.
"""
import re
from types import SimpleNamespace

import pytest


def _load_kernel_class():
    from flagscale_agent.react.kernel import AgentKernel as Kernel
    return Kernel


def _make_kernel(thinking_budget):
    K = _load_kernel_class()
    k = object.__new__(K)
    k.deps = SimpleNamespace(config=SimpleNamespace(thinking_budget=thinking_budget))
    k._squeeze_steps = 0
    return k


def test_ladder_halves_from_configured_budget():
    """16384 -> 8192 -> 4096 -> 2048, then stays at the floor."""
    k = _make_kernel(16384)
    seq = [k._next_squeeze_budget() for _ in range(5)]
    assert seq == [8192, 4096, 2048, 2048, 2048], seq


def test_ladder_never_goes_below_floor():
    """Whatever the configured budget, the returned value is >= the floor."""
    floor = _load_kernel_class().THINKING_SQUEEZE_TOKENS
    k = _make_kernel(16384)
    for _ in range(20):
        assert k._next_squeeze_budget() >= floor


def test_first_arm_is_not_the_floor():
    """The whole point: the FIRST squeeze keeps headroom above the floor.

    The old behaviour returned the floor immediately; that is what starved the
    forced tool call. With a configured budget of 16384 the first arm must be
    strictly greater than the floor.
    """
    floor = _load_kernel_class().THINKING_SQUEEZE_TOKENS
    k = _make_kernel(16384)
    first = k._next_squeeze_budget()
    assert first > floor, f"first arm returned the floor ({first}) — regression"
    assert first == 16384 // 2


def test_small_or_unset_budget_falls_back_to_floor():
    """budget=0 (thinking disabled) and budgets at/below the floor stay there."""
    floor = _load_kernel_class().THINKING_SQUEEZE_TOKENS
    for budget in (0, 1024, floor):
        k = _make_kernel(budget)
        assert [k._next_squeeze_budget() for _ in range(3)] == [floor] * 3


def test_ladder_resets_when_state_lands_on_disk():
    """A successful write/edit restores the full budget AND the ladder top."""
    k = _make_kernel(16384)
    assert [k._next_squeeze_budget() for _ in range(3)] == [8192, 4096, 2048]
    # Simulate what the kernel does when a write/edit produced state on disk.
    k._thinking_cap_override = None
    k._squeeze_steps = 0
    assert k._next_squeeze_budget() == 8192


def test_ladder_is_per_turn_not_cross_turn():
    """run_turn() resets the ladder, so debt does not leak across turns.

    Source-level assertion because constructing a full kernel needs every dep;
    the invariant that matters is that run_turn initialises the counter.
    """
    import inspect
    from flagscale_agent.react.kernel import AgentKernel as Kernel
    src = inspect.getsource(Kernel.run_turn)
    assert "_squeeze_steps = 0" in src, "run_turn no longer resets the ladder"


def test_all_squeeze_call_sites_use_the_ladder():
    """No call site may assign the raw constant — all must ask the ladder."""
    import inspect
    from flagscale_agent.react import kernel as kernel_mod
    src = inspect.getsource(kernel_mod)
    bad = re.findall(r"_thinking_cap_override\s*=\s*self\.THINKING_SQUEEZE_TOKENS\b", src)
    assert not bad, f"{len(bad)} call site(s) still bypass the ladder"
    good = re.findall(r"_thinking_cap_override\s*=\s*self\._next_squeeze_budget\(\)", src)
    assert len(good) >= 5, f"expected >=5 ladder call sites, found {len(good)}"

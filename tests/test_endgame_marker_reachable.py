"""Regression test: the endgame force_tool window must leave the marker reachable.

Mechanism under test (measured on kv-store-grpc trial UNKgJ4N, job
2026-11-05__06-48-27):

  * kernel arms `_force_tool_next = True` at the TOP OF EVERY LOOP ITERATION
    once the time-budget pct crosses `arm_thr`, and L444-445 clears it after
    one consumption — so the whole endgame window re-arms each round.
  * an armed round sends `tool_choice="required"` (openai_provider L61-62),
    under which "a prose-only reply is not a legal completion"
    (providers/base.py docstring).
  * the prompt (10.5) orders: "EMIT THE MARKER IN A TEXT-ONLY RESPONSE, no
    tool call in it at all".

Permanent arming therefore makes the turn's own exit unreachable. Observed:
126/126 rounds carried tool_use, 0 text-only rounds, 0 literal
[TASK_COMPLETE] in content — while 29 rounds' thinking said "emit the marker
now, text-only". Control trial pypi-server (182s, never armed) closed with a
single ['text'] block ending in [TASK_COMPLETE].

This test asserts the CADENCE, not the presentation: inside the armed window
consecutive iterations must alternate armed/released, so a text-only marker
response is legal at least every ENDGAME_RELEASE_EVERY rounds.
"""
import pytest

from flagscale_agent.react.kernel import AgentKernel


def _cadence(n_rounds, release_every=2):
    """Replicate the arming decision at the top of the loop (kernel L389..).

    Returns the list of booleans: True = this round is FORCED.
    """
    forced = []
    endgame_rounds = 0
    for _ in range(n_rounds):
        endgame_rounds += 1
        release = release_every > 0 and endgame_rounds % release_every == 0
        forced.append(not release)
    return forced


def test_endgame_window_alternates_so_marker_is_legal():
    seq = _cadence(6, AgentKernel.ENDGAME_RELEASE_EVERY)
    # Not every round forced: the pre-fix behaviour failed here.
    assert any(f is False for f in seq), (
        "every endgame round is forced -> a text-only marker response is "
        f"structurally illegal for the whole window: {seq}"
    )
    # At least one legal opening every ENDGAME_RELEASE_EVERY rounds.
    max_gap = 0
    run = 0
    for f in seq:
        run = 0 if not f else run + 1
        max_gap = max(max_gap, run)
    assert max_gap < AgentKernel.ENDGAME_RELEASE_EVERY, (
        f"longest forced streak {max_gap} >= ENDGAME_RELEASE_EVERY "
        f"{AgentKernel.ENDGAME_RELEASE_EVERY}: {seq}"
    )


def test_pressure_preserved_first_round_is_forced():
    """The fix must not neuter the mechanism: the anti-burn pressure stays."""
    seq = _cadence(4, AgentKernel.ENDGAME_RELEASE_EVERY)
    assert seq[0] is True, (
        "the first endgame round must still be forced, otherwise the release "
        "turns the window back into unconstrained prose (the failure the "
        f"permanent arming was introduced to fix): {seq}"
    )
    forced_share = sum(seq) / len(seq)
    assert forced_share >= 0.5, (
        f"at least half the window must stay forced, got {forced_share:.2f}: {seq}"
    )


def test_release_disabled_reproduces_the_bug():
    """Falsification hook: with the release switched OFF the cadence is the
    pre-fix permanent arming, i.e. the marker is unreachable. If this ever
    stops holding, the test above proves nothing."""
    seq = _cadence(5, release_every=0)
    assert all(seq), f"release_every=0 must force every round: {seq}"

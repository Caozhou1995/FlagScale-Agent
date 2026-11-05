# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Endgame force_tool cadence, driven through the REAL kernel loop.

Why this file exists (and why the replica test alone was not enough):
`test_endgame_marker_reachable.py` re-implements the arming decision and then
asserts on its own re-implementation — it shares its source of truth with the
code under test, so it cannot fail when the real kernel bypasses the arming
point (an early `continue`/`return` branch, a scope error in the phase
counter, ENDGAME_RELEASE_EVERY read off the wrong object). This file instead
drives `AgentKernel.run_turn` with a fake provider and records the ACTUAL
`force_tool` kwarg the kernel sends on each round.

Measured failure this encodes (kv-store-grpc trial UNKgJ4N,
jobs/2026-11-05__06-48-27):
  budget 900.18s -> arm_thr = max(50, 85 - 240/900*100) = 58.34%
  -> forced for the last 375.0s
  -> 126 assistant rounds, ALL 126 carried tool_use, 0 text-only rounds,
     0 literal [TASK_COMPLETE] in content, while 29 rounds' thinking said
     "emit the marker now, text-only"
  Control pypi-server: 182.39s (never armed) closed with a single ['text']
  block ending in [TASK_COMPLETE] — the only trial in the batch to emit it.
Mechanism: an armed round sends tool_choice="required", under which
"a prose-only reply is not a legal completion" (providers/base.py), while
prompt 10.5 orders the marker to be text-only. Permanent arming therefore
makes the turn's own exit unreachable.
"""

from unittest.mock import MagicMock
import types

import pytest

from flagscale_agent.react.kernel import AgentKernel, KernelDeps
from flagscale_agent.react.guard import GuardRegistry


class _History:
    def __init__(self):
        self.messages = [{"role": "user", "content": "do the task"}]

    def append(self, msg):
        self.messages.append(msg)

    def get_messages(self):
        return self.messages

    def get_context_pressure(self):
        return 0.1

    def get_evictable_indexes(self):
        return []

    def report_actual_tokens(self, n):
        pass


def _drive(pct, n_rounds, response_fn, release_every=None):
    """Run the real kernel loop with a time budget already at *pct*.

    Returns (force_sequence, KernelResult). force_sequence[i] is the
    force_tool kwarg the kernel actually passed on call i.
    """
    history = _History()
    registry = GuardRegistry()
    config = types.SimpleNamespace(
        max_iterations=n_rounds, max_continuations=200, mode="auto",
        _turn_count=0,
    )

    force_seq = []
    calls = {"n": 0}

    def call_llm_fn(messages, schemas, **kw):
        force_seq.append(bool(kw.get("force_tool", False)))
        calls["n"] += 1
        return response_fn(calls["n"] - 1), {"input_tokens": 1, "output_tokens": 1}

    provider = MagicMock()
    provider.format_assistant_message.side_effect = (
        lambda resp: {"role": "assistant", "content": resp.get("content", "")})

    deps = KernelDeps(
        provider=provider,
        history=history,
        tool_registry=MagicMock(),
        judge=MagicMock(),
        guard_registry=registry,
        config=config,
        display=MagicMock(),
        get_schemas_fn=lambda: [],
        inject_message_fn=lambda msg: history.append(
            {"role": "user", "content": msg}),
        append_tool_results_fn=lambda results: None,
        format_tool_result_fn=lambda tid, r: {},
        execute_tools_fn=lambda tcs: ["ok"] * len(tcs),
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=call_llm_fn,
        # Budget already past the arm point: pct=94 > arm_thr for a 900s budget.
        time_budget_stats_fn=lambda: {
            "elapsed": 0.94 * 900.0, "budget": 900.0,
            "remaining": 0.06 * 900.0, "pct": pct,
        },
    )
    kernel = AgentKernel(deps)
    if release_every is not None:
        kernel.ENDGAME_RELEASE_EVERY = release_every
    result = kernel.run_turn()
    return force_seq, result


def _tool_call_response(i):
    """Every round: one shell call (the observed shape of the real trial)."""
    return {"content": "", "tool_calls": [
        {"id": f"c{i}", "name": "shell", "arguments": {"command": "echo hi"}}]}


def test_real_kernel_alternates_force_tool_in_endgame_window():
    """THE regression: inside the armed window the requests must alternate."""
    seq, _ = _drive(pct=94.0, n_rounds=6, response_fn=_tool_call_response)
    seq = seq[:6]
    assert len(seq) >= 4, f"kernel stopped early, only {len(seq)} rounds: {seq}"
    assert any(f is False for f in seq), (
        "every endgame request was forced -> a text-only marker response is "
        f"structurally illegal for the whole window: {seq}"
    )
    # first round still forced, then alternating
    assert seq[0] is True, f"first endgame round must stay forced: {seq}"
    max_run = run = 0
    for f in seq:
        run = run + 1 if f else 0
        max_run = max(max_run, run)
    assert max_run < AgentKernel.ENDGAME_RELEASE_EVERY, (
        f"longest forced streak {max_run} >= ENDGAME_RELEASE_EVERY "
        f"{AgentKernel.ENDGAME_RELEASE_EVERY}: {seq}"
    )


def test_real_kernel_below_arm_threshold_never_forces():
    """Counter-test: the arming must not fire before the arm point."""
    seq, _ = _drive(pct=30.0, n_rounds=4, response_fn=_tool_call_response)
    assert not any(seq), f"forced below the arm threshold: {seq}"


def test_real_kernel_release_disabled_reproduces_the_bug():
    """Falsification hook on the REAL path: release=0 => permanent arming.

    If this ever stops holding, the test above proves nothing.
    """
    seq, _ = _drive(pct=94.0, n_rounds=5, response_fn=_tool_call_response,
                    release_every=0)
    seq = seq[:5]
    assert seq and all(seq), (
        "ENDGAME_RELEASE_EVERY=0 must reproduce the pre-fix permanent arming "
        f"(every request forced): {seq}"
    )


def test_text_only_marker_ends_the_turn():
    """A released round must actually accept a text-only marker as completion."""
    def resp(i):
        if i == 0:
            return _tool_call_response(0)
        return {"content": "done.\n\n[TASK_COMPLETE]", "tool_calls": []}

    seq, result = _drive(pct=94.0, n_rounds=6, response_fn=resp)
    # Round 2 must be a released round (alternation), and it must close the turn.
    assert seq[1] is False, f"round 2 should be released: {seq}"
    assert result.stop_reason, (
        "a text-only [TASK_COMPLETE] did not end the turn "
        f"(stop_reason={result.stop_reason!r})"
    )
    assert len(seq) == 2, (
        f"turn should have closed on round 2, but {len(seq)} requests were "
        f"made: {seq}"
    )

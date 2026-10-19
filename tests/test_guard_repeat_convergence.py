# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for repeat-block convergence (M6, block channel).

Root cause observed in four_m4/mteb-retrieve: a task FINISHED its work, then
burned its whole wall clock on 42 blocks of the SAME (guard, reason) slug —
the block message names a category but the agent cannot localize which key or
nesting level is wrong, so it re-emits nearly-identical declarations until the
timeout. The registry had NO mechanism to detect or escalate a repeated
identical block.

Fix: GuardRegistry counts consecutive resolves surfacing the SAME
(owner-guard, reason) block. At REPEAT_CONVERGENCE_AT it appends a convergence
directive (change the FORM, not the wording); at _REPEAT_CONVERGENCE_HARD_AT it
appends the hard directive (stop re-issuing; take the alternative exit). The
streak resets ONLY via reset_repeat_convergence(), which the kernel calls when
a tool actually executes (real progress), and when the surfaced (owner, reason)
CHANGES (a different gate now fires).
"""

import flagscale_agent.react.guard as guard_mod
from flagscale_agent.react.guard import (
    Guard, GuardVerdict, GuardContext, GuardRegistry,
)


class _Repeat(Guard):
    name = "stub"
    priority = 50

    def check_pre(self, ctx):
        return GuardVerdict.block("[stub] blocked", reason="same_slug",
                                  category="c", overridable=False)


class _Switching(Guard):
    """Alternate between two reasons to prove the counter tracks (owner, reason)."""

    name = "switch"
    priority = 50

    def __init__(self):
        self.n = 0

    def check_pre(self, ctx):
        self.n += 1
        reason = "alpha" if self.n % 2 else "beta"
        return GuardVerdict.block("[switch] blocked", reason=reason,
                                  category="c", overridable=False)


def _ctx():
    return GuardContext(tool_name="shell", tool_args={"command": "x"})


def test_repeat_convergence_fires_at_threshold_with_soft_directive():
    reg = GuardRegistry()
    reg.register(_Repeat())
    ctx = _ctx()
    msgs = []
    for i in range(1, guard_mod.REPEAT_CONVERGENCE_AT + 2):
        v = reg.check_pre(ctx)
        msgs.append(v.message)
    # below threshold: no directive
    for m in msgs[:guard_mod.REPEAT_CONVERGENCE_AT - 1]:
        assert "CONVERGENCE:" not in m
    # at threshold: soft directive naming the count and the reason slug
    at = msgs[guard_mod.REPEAT_CONVERGENCE_AT - 1]
    assert "CONVERGENCE:" in at
    assert "same_slug" in at
    assert str(guard_mod.REPEAT_CONVERGENCE_AT) in at
    # past threshold but below hard: still the soft directive (no HARD)
    assert "HARD" not in at


def test_repeat_convergence_hard_directive_at_hard_threshold():
    reg = GuardRegistry()
    reg.register(_Repeat())
    ctx = _ctx()
    v = None
    for _ in range(guard_mod._REPEAT_CONVERGENCE_HARD_AT):
        v = reg.check_pre(ctx)
    assert "HARD" in v.message
    assert "same_slug" in v.message
    # hard threshold message must forbid re-issuing the same call
    assert "Do NOT emit" in v.message or "NOT emit" in v.message


def test_repeat_counter_resets_when_reason_changes():
    reg = GuardRegistry()
    g = _Switching()
    reg.register(g)
    ctx = _ctx()
    # alpha, beta, alpha, beta... — the (owner, reason) key alternates, so the
    # count for each specific reason restarts at 1 every time it reappears.
    for _ in range(2 * guard_mod.REPEAT_CONVERGENCE_AT):
        v = reg.check_pre(ctx)
    assert "CONVERGENCE:" not in v.message


def test_explicit_reset_clears_streak():
    reg = GuardRegistry()
    reg.register(_Repeat())
    ctx = _ctx()
    for _ in range(guard_mod.REPEAT_CONVERGENCE_AT + 1):
        reg.check_pre(ctx)
    reg.reset_repeat_convergence()
    v = reg.check_pre(ctx)
    assert "CONVERGENCE:" not in v.message
    # and the counter starts over from 1 after the reset
    v2 = None
    for _ in range(guard_mod.REPEAT_CONVERGENCE_AT - 2):
        v2 = reg.check_pre(ctx)
    assert "CONVERGENCE:" not in v2.message


def test_kernel_reset_is_gated_on_tool_execution():
    """The kernel must reset the streak ONLY when a tool actually executed.

    Behavioural (not a source grep): this reproduces the kernel's per-iteration
    ordering — [existing streak] then check_pre — and asserts the streak still
    accumulates across turns when NOTHING resets it (the all-blocked case), and
    that an explicit reset (the a-tool-executed case) does clear it. A lossy
    wiring that calls reset unconditionally every turn would pin the count at 1
    and never reach REPEAT_CONVERGENCE_AT; this test fails in that world.
    """
    import flagscale_agent.react.guard as gm

    reg = GuardRegistry()
    reg.register(_Repeat())
    ctx = _ctx()

    # Simulate 4 turns of an ALL-BLOCKED loop: NO reset (no tool executed).
    for _ in range(gm.REPEAT_CONVERGENCE_AT):
        v = reg.check_pre(ctx)
    assert reg._repeat_count >= gm.REPEAT_CONVERGENCE_AT
    assert v is not None and "CONVERGENCE:" in v.message

    # Simulate a turn where a tool EXECUTED: the kernel calls reset once.
    reg.reset_repeat_convergence()
    assert reg._repeat_count == 0


def test_kernel_all_blocked_turns_accumulate_convergence():
    """Integration (the defect F1 closed): drive AgentKernel.run_turn through
    several ALL-BLOCKED tool turns and assert the CONVERGENCE directive reaches
    history. In the buggy wiring the kernel reset the streak unconditionally at
    the top of every iteration — including all-blocked ones — so the count was
    pinned at 1 and NO convergence ever fired. This test exercises the REAL
    kernel ordering, so it fails in that world (unlike the old source-grep)."""
    import types
    from unittest.mock import MagicMock
    from flagscale_agent.react.kernel import AgentKernel, KernelDeps

    class _Block(Guard):
        name = "blocker"
        priority = 50

        def check_pre(self, ctx):
            # Only block the TOOL CALL (the real mteb site), not the top-of-loop
            # empty-ctx consultation — a guard that also blocks there would
            # `continue` before tool execution and never exercise the reset path.
            if not ctx.tool_name:
                return None
            return GuardVerdict.block("[blocker] nope", reason="same_slug",
                                      category="c", overridable=False)

    class _Hist:
        def __init__(self):
            self.messages = [{"role": "user", "content": "go"}]

        def append(self, m):
            self.messages.append(m)

        def get_messages(self):
            return self.messages

        def get_context_pressure(self):
            return 0.1

        def get_evictable_indexes(self):
            return []

        def report_actual_tokens(self, n):
            pass

        def pop_last_assistant(self):
            for i in range(len(self.messages) - 1, -1, -1):
                if self.messages[i].get("role") == "assistant":
                    self.messages.pop(i)
                    return

    reg = GuardRegistry()
    reg.register(_Block())
    hist = _Hist()
    injected = []
    cfg = types.SimpleNamespace(max_iterations=20, max_continuations=200,
                                mode="auto", max_output_tokens=100)
    provider = MagicMock()
    provider.format_assistant_message.side_effect = (
        lambda r: {"role": "assistant", "content": r.get("content", "")}
    )

    def call_llm_fn(messages, schemas):
        # Every turn: the SAME blocked tool call (a guess-retry loop).
        return ({"content": "", "tool_calls": [
            {"id": "1", "name": "plan_update", "arguments": {}}
        ]}, {"input_tokens": 1, "output_tokens": 5})

    deps = KernelDeps(
        provider=provider, history=hist, tool_registry=MagicMock(),
        judge=MagicMock(), guard_registry=reg, config=cfg, display=MagicMock(),
        get_schemas_fn=lambda: [], inject_message_fn=lambda m: injected.append(m),
        append_tool_results_fn=lambda r: None,
        format_tool_result_fn=lambda t, r: {}, execute_tools_fn=lambda tcs: [],
        is_context_limit_error_fn=lambda e: False, call_llm_fn=call_llm_fn,
    )
    AgentKernel(deps).run_turn()
    joined = "\n".join(injected)
    assert "CONVERGENCE" in joined, "M6 never fired on an all-blocked loop"



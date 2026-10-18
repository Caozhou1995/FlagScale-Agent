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


def test_kernel_progress_reset_is_wired():
    """The kernel calls reset_repeat_convergence() when tools execute."""
    import inspect
    from flagscale_agent.react import kernel as kernel_mod
    src = inspect.getsource(kernel_mod.AgentKernel)
    assert "reset_repeat_convergence()" in src

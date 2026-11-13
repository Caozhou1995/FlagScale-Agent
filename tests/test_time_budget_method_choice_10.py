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

"""The 10% METHOD-CHOICE checkpoint.

Pins the three things that make this rung worth having, and the two things it
must NOT do:

  * it is a BLOCK (check_pre), not an inject, so the agent must spend one
    response answering instead of reading it as pacing chatter in the same
    breath as the 25% nudge;
  * it is released by ANY override_reason — the prose is the forcing function,
    not a credential check;
  * its text states BOTH halves of "a good method" — satisfies the task's
    stated constraints AND finishes inside the budget — because a method that
    fails either half scores 0 either way;
  * it carries NO regex, NO bound extraction and NO task-name logic. The whole
    point of this rung (after an extractor-based design was rejected) is that
    the model reads its own task text and judges; the prompt only supplies the
    criterion. A future edit that reintroduces matching here should fail these
    tests;
  * it fires only in the 10..25% window — never late. A run that jumps
    9% -> 60% between calls must not be told "switch now, it is still cheap".
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.time_budget import TimeBudgetGuard


def _stats(pct: float, budget: float = 900.0):
    elapsed = budget * pct / 100.0
    return {
        "elapsed": elapsed,
        "budget": budget,
        "remaining": budget - elapsed,
        "pct": float(pct),
    }


def _ctx(tool: str = "write_file"):
    return GuardContext(messages=[], tool_name=tool)


def _block_at(pct: float, tool: str = "write_file"):
    g = TimeBudgetGuard(lambda: _stats(pct))
    return g.check_pre(_ctx(tool))


def _msg(thr: int, pct: float) -> str:
    g = TimeBudgetGuard(lambda: _stats(pct))
    return g._message(thr, {**_stats(pct), "pct": float(pct)})


# --------------------------------------------------------------------------
# the rung exists, in the right place in the ladder
# --------------------------------------------------------------------------


def test_10_is_the_lowest_rung():
    assert TimeBudgetGuard._THRESHOLDS[-1] == 10
    # highest -> lowest, so the most severe pending threshold wins
    assert TimeBudgetGuard._THRESHOLDS == (100, 90, 80, 75, 65, 50, 25, 10)


def test_10_percent_blocks_a_tool_call():
    v = _block_at(10.0)
    assert v is not None
    assert v.action == "block", "the rung must BLOCK, not inject"
    assert v.reason == "time_budget_10pct_method_choice_block"


def test_block_is_overridable_by_any_reason():
    """Any override releases it — this is a forcing function, not a credential
    check. An extractor-based gate that demanded specific evidence was rejected
    because it turned into ceremony; the whole design here is that the model
    judges, and the gate only buys the response in which it judges."""
    v = _block_at(10.0)
    assert v.overridable is True


def test_blocks_every_tool_kind_not_just_observations():
    """Unlike the 90% wrap-up gate (which only stops observations so it cannot
    block the write that IS the deliverable), this gate's subject is the method
    itself, so it must stop production calls too — otherwise the agent writes
    more of a wrong method and the checkpoint never lands."""
    for tool in ("write_file", "edit_file", "shell", "read_file"):
        assert _block_at(10.0, tool) is not None, tool


def test_fires_once_only():
    g = TimeBudgetGuard(lambda: _stats(11.0))
    assert g.check_pre(_ctx()) is not None
    assert g.check_pre(_ctx()) is None


def test_silent_before_10_percent():
    assert _block_at(9.9) is None
    assert _block_at(0.0) is None


def test_silent_after_the_window_closes():
    """The advice "switching is still cheap" is FALSE later on. This is the
    reason the rung sits at 10 and not 75; if it also fired at 60 it would
    reintroduce exactly the late advice it exists to replace."""
    for pct in (25.0, 40.0, 60.0, 76.0, 95.0):
        assert _block_at(pct) is None, f"must stay silent at {pct}%"


def test_check_post_does_not_also_inject_the_10_rung():
    """A block AND an inject of the same 1550-character lecture in the same
    turn is one lecture too many."""
    g = TimeBudgetGuard(lambda: _stats(10.5))
    v = g.check_post(_ctx())
    assert v is None, f"10% must not inject, got {v.reason if v else None}"
    # the neighbouring rung still injects normally
    g2 = TimeBudgetGuard(lambda: _stats(26.0))
    v2 = g2.check_post(_ctx())
    assert v2 is not None and v2.reason == "time_budget_25pct"


# --------------------------------------------------------------------------
# the text carries the criterion (this is the entire payload)
# --------------------------------------------------------------------------


def test_10_rung_states_both_halves_of_a_good_method():
    """A method must satisfy the constraints AND finish in time. Both halves
    are the whole content of this rung; dropping either leaves the rung with
    nothing to say that the 25% pacing nudge does not already cover.

    The conjunction is load-bearing: (a) AND (b), not OR. Under OR the rung
    would license shipping a method that finishes fast but violates the bound
    — the exact failure this rung exists to prevent. Measured: an earlier
    version of this test passed after AND was ablated to OR, because it only
    substring-matched the two clauses; the MATCHER is now pinned too.
    """
    m = _msg(10, 10.5)
    assert "hardest constraint the task itself states" in m
    assert "INCLUDING the time to verify" in m
    # both failure directions must be named, and both must read as zero
    assert "never delivered, scores 0" in m
    assert "violates a stated bound" in m
    assert "fails EITHER one scores ZERO" in m
    assert "scores 0" in m


def test_10_rung_names_the_coordinate_both_failures_share():
    """Both failure directions must be named, and they must be the SAME two —
    too slow vs. violating a bound. Naming only one turns the rung into a
    deadline nudge, which the 25/75 rungs already do."""
    m = _msg(10, 10.5)
    assert "too slow to finish" in m
    assert "violates a stated bound" in m
    assert "METHOD is wrong" in m


def test_10_rung_is_not_only_a_deadline_reminder():
    """The rung must not degrade into 'you are running out of time'. It must
    force a number out of the model, not a vibe."""
    m = _msg(10, 10.5)
    assert "should be fine" in m, "the rung must explicitly reject a vibe"
    assert "the number your method implies" in m
    assert "unit cost x units" in m


def test_10_rung_contains_no_bound_extraction():
    """Guard against a future re-introduction of the extractor design: no
    regex, no task-name special-casing, no per-task constants. This is what
    the user rejected — the rung must stay semantic."""
    m = _msg(10, 10.5)
    # no task names
    for name in ("write-compressor", "write_compressor", "data.comp", "2500",
                 "compression"):
        assert name not in m, f"task-specific token leaked: {name}"
    # no numeric bound table / regex machinery in this rung's own text
    assert "regex" not in m.lower()
    assert re.search(r"\d{3,} bytes", m) is None, "a hardcoded byte bound leaked"


def test_10_rung_states_release_path():
    """The message must say how to get past it, or the block reads as a wall."""
    m = _msg(10, 10.5)
    assert "any reasoning releases this gate" in m
    assert "if both answers are comfortable" in m.lower() or "comfortable" in m

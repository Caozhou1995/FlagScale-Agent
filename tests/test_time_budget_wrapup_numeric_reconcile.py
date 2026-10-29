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

"""The WRAP-UP / TIME-IS-UP messages must make the agent reconcile the limit the
task states against its own last measured value.

Rationale (measured losses): the commonest late failure is not a missing result
but a result the agent had ALREADY measured as violating a stated bound and
shipped anyway (a shape counter measured at 9 against a stated max of 8; a code
size measured at 5533 against a stated max of 5000). The wrap-up copy previously
only demanded "a complete valid result is banked", which such a run satisfies.
"""

import re

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.time_budget import TimeBudgetGuard


def make_ctx(tool_name="read_file"):
    ctx = GuardContext()
    ctx.tool_name = tool_name
    return ctx


class _Stats:
    def __init__(self, pct):
        self.pct = pct

    def __call__(self):
        budget = 1200.0
        return {
            "elapsed": budget * self.pct / 100.0,
            "budget": budget,
            "remaining": budget * (1.0 - self.pct / 100.0),
            "pct": self.pct,
        }


def _flat(text):
    return re.sub(r"\s+", " ", text)


def test_80_wrapup_demands_limit_vs_measured_reconciliation():
    g = TimeBudgetGuard(stats_fn=_Stats(82.0))
    v = g.check_post(make_ctx())
    assert v is not None
    flat = _flat(v.message)
    assert "WRAP-UP WINDOW" in flat
    # the decisive pair must be named: task's own limit vs own last measurement
    assert "reconcile the ONE number that decides the grade" in flat
    assert "the limit the TASK ITSELF states" in flat
    assert "your OWN last measurement" in flat
    # and the two escapes the loss runs actually used must be pre-empted
    assert "do not reinterpret the task's limit" in flat


def test_90_and_100_demand_numbers_side_by_side_and_no_self_invented_budget():
    for pct in (92.0, 100.0):
        g = TimeBudgetGuard(stats_fn=_Stats(pct))
        v = g.check_post(make_ctx())
        assert v is not None, pct
        flat = _flat(v.message)
        assert "put the numbers side by side" in flat, pct
        assert "the limit the TASK ITSELF states" in flat, pct
        # a self-chosen limit is explicitly disallowed as a substitute
        assert "self-invented budget is not the task's budget" in flat, pct

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

"""The 90% CRITICAL checkpoint must distinguish "nothing at the delivery path
yet" from "something is there but it violates a stated limit", and must say
that a compliant copy living outside the delivery path is not a deliverable.

Rationale (measured loss, gpt2-codegolf): the agent's artifact at the graded
path was 5232 bytes against a stated max of 5000. In the final 30 seconds it
computed a compliant 4983-byte version but wrote it to /tmp/x.c and never
copied it onto the graded path (0 hits for `cp /tmp/x.c`, `cat > <path>`,
`mv /tmp/x.c` across the whole trajectory). The pre-change copy only demanded
"write the deliverable to that path NOW", which a run whose good answer is one
`cp` away has already satisfied in its own mind.
"""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.time_budget import TimeBudgetGuard


def make_ctx(tool_name="read_file"):
    ctx = GuardContext()
    ctx.tool_name = tool_name
    return ctx


class _Stats:
    """Minimal stats_fn stand-in matching the guard's expected shape."""

    def __init__(self, pct):
        self.pct = pct

    def __call__(self):
        return {
            "pct": self.pct,
            "budget": 900.0,
            "elapsed": 900.0 * self.pct / 100.0,
            "remaining": 900.0 * (1.0 - self.pct / 100.0),
            "rounds": 4,
            "per_round": 225.0,
        }


def _critical_message() -> str:
    g = TimeBudgetGuard(stats_fn=_Stats(93.0))
    v = g.check_pre(make_ctx())
    assert v is not None and v.action == "block"
    assert v.reason == "time_budget_90pct_block"
    return v.message


class TestCriticalCheckpointDeliveryShapes:
    def test_distinguishes_missing_from_violating(self):
        msg = _critical_message()
        # Shape (a): nothing at the path yet.
        assert "NOTHING is at that path yet" in msg
        # Shape (b): something is there but it violates a stated limit.
        assert "Something IS at that path" in msg
        assert "VIOLATES a limit the task itself states" in msg

    def test_says_measure_at_the_delivery_path(self):
        msg = _critical_message()
        assert "Measure it AT THAT PATH" in msg
        # The "patch THAT file, do not restart from scratch" reading is the
        # load-bearing half: a vague "patch a file" invites a rewrite. Assert the
        # deictic THAT, not just the verb — an ablation proved this assertion
        # was missing when only the VERB was pinned.
        assert "patch THAT file" in msg
        assert "Do NOT restart from scratch" in msg
        assert "do not re-derive it" in msg

    def test_says_out_of_path_copy_is_not_a_deliverable(self):
        msg = _critical_message()
        assert "A compliant copy sitting in /tmp" in msg
        assert "The grader reads only the path the task names" in msg
        # and the concrete cheap escape (copy it over) is named
        assert "copying it onto that path is a single call" in msg

    def test_missing_shape_still_points_at_write_file(self):
        # The original shape-(a) advice must survive verbatim enough that a run
        # with nothing on disk still gets told to write, not to measure.
        msg = _critical_message()
        assert "crude-but-complete" in msg
        assert "write_file" in msg


class TestWrapupRungsShareTheSameRule:
    """The same "measure at the path, a copy elsewhere is not delivered" rule
    must also be present on the 80% and 100% wrap-up rungs — an agent that
    reaches those rungs without the 90% gate having fired (e.g. a giant single
    generation skipping 90..100) still needs it."""

    def test_80_rung_measures_at_path(self):
        msg = TimeBudgetGuard(stats_fn=_Stats(82.0))._message(80, _Stats(82.0)())
        assert "MEASURE it there against the limit the task states" in msg
        assert "A compliant copy in /tmp or a scratch dir is NOT delivered" in msg

    def test_100_rung_measures_at_path(self):
        msg = TimeBudgetGuard(stats_fn=_Stats(100.0))._message(100, _Stats(100.0)())
        assert "MEASURE " in msg
        assert "A compliant copy in /tmp or a scratch dir is NOT" in msg

    def test_90_postinject_fallback_measures_at_path(self):
        # check_post is the fallback path for the 90 rung when check_pre was
        # suppressed by a concurrent block; it must carry the same rule.
        msg = TimeBudgetGuard(stats_fn=_Stats(95.0))._message(90, _Stats(95.0)())
        assert "AT THE DELIVERY PATH" in msg
        assert "a compliant copy in /tmp or a scratch dir is NOT delivered" in msg

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

"""FirstE2eRunGuard — progress-ORDER checkpoints at 25% and 50% of budget.

TB failure review found a progress-order inversion that pure time awareness
never catches: runs that spend most of the budget on intermediates and only
touch the deliverable at the very end. Two observed faces:
  • run-late — the first end-to-end exercise of the pipeline happens when
    most of the budget is gone (a run at 80% budget that had never executed
    its own deliverable end-to-end; another whose first full pipeline run
    started with ~1/30 of the budget left). Integration bugs discovered that
    late have no budget left to be fixed in.
  • write-late — the delivery path stays empty through the design/implementation
    phase (zero writes through the first half of the run; the first landing
    only after the quarter mark). If the run dies late, every completed step
    scores zero because nothing sits at the required path.

TimeBudgetGuard already injects pacing advice at these rungs, but advice
does not force a decision (observed: agents read it and keep exploring).
This guard uses the lever the 90% block uses — a block that costs one
override — at the rungs where changing the ORDER of work still costs
minutes instead of the task.

Key design decisions:
  • AWARENESS ONLY, NO bookkeeping. The guard does NOT count writes, does
    NOT parse delivery paths, does NOT try to detect whether an end-to-end
    run happened — any such proxy would be brittle and task-specific. It
    only blocks to force the agent to explicitly answer the ORDER question
    in its _override_reason (the LLM analyzes its own progress; the guard
    supplies the consciousness, not the audit).
  • The ONLY state is the per-turn fired set, mirroring TimeBudgetGuard.
  • 25% → write-late checkpoint: is a minimal usable deliverable already at
    its path? If not, land a crude-but-valid skeleton THIS STEP and iterate
    on completeness — never complete-then-write.
  • 50% → run-late checkpoint: compile/build success proves an intermediate,
    not the deliverable. Exercise the simplest end-to-end path NOW; a bug
    found now costs minutes, found at 90% it costs the task.
  • Both blocks are overridable and each fires at most once per turn; a
    jump straight past both fires only the 50% one (the later, more urgent
    question subsumes the earlier one).
  • Silent when no external wall is enforced (stats_fn → None), exactly
    like TimeBudgetGuard.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

_MSG_25 = (
    "[ProgressOrder] 25% of the enforced wall-clock budget is spent — this is "
    "the WRITE-THROUGH checkpoint, and it is an ORDER question, not a pacing "
    "tip: does a minimal usable version of the deliverable ALREADY sit at its "
    "required delivery path? If nothing is there yet — the design is done, "
    "the primitives work, but the path is empty — land a crude-but-valid "
    "skeleton at the path THIS STEP (the required file, correct name/format, "
    "worst-acceptable content) and iterate on completeness afterward. "
    "Complete-then-write is the inverted order: a run that dies after the "
    "design phase scores zero even though every step worked. Before "
    "executing this tool, state in _override_reason either (a) the path "
    "already holds a usable skeleton (cite the path and what is in it) or "
    "(b) a delivery path does not apply to this task, with the reason, or "
    "(c) the tool call you are making now WRITES the first version to the "
    "path."
)

_MSG_50 = (
    "[ProgressOrder] 50% of the enforced wall-clock budget is spent — this is "
    "the FIRST-EXERCISE checkpoint: a compile or build that succeeds proves "
    "an intermediate artifact, NOT a working deliverable. Has the simplest "
    "end-to-end path of your deliverable actually been EXECUTED yet (run, "
    "invoked, or exercised against a real input)? If not — pipeline never "
    "assembled, output never produced, correctness never observed — run the "
    "cheapest end-to-end exercise NOW: a bug found here costs minutes; the "
    "same bug found at 90% costs the task. Before executing this tool, "
    "state in _override_reason either (a) the deliverable has already been "
    "exercised end-to-end (name the run and its observed output) or (b) "
    "this task has no meaningful end-to-end exercise, with the reason, or "
    "(c) the tool call you are making now RUNS that first exercise."
)


class FirstE2eRunGuard(Guard):
    """Block once each at 25% and 50% to force an explicit progress-ORDER
    answer (deliverable written? deliverable exercised?) before the next
    tool call."""

    name = "first_e2e_run"
    priority = 93  # Next to time_budget (92) — advisory-family, low priority.

    # Ordered high→low: the FIRST crossed-but-unfired threshold handled is the
    # most severe pending question.
    _THRESHOLDS = (50, 25)

    def __init__(self, stats_fn):
        """stats_fn() -> dict|None with keys elapsed/budget/remaining/pct.

        None means no external wall is enforced; the guard stays silent.
        Same contract as TimeBudgetGuard's stats_fn.
        """
        self._stats_fn = stats_fn
        self._fired: set[int] = set()

    def reset_turn(self):
        # A new user turn restarts the per-turn budget accounting on the
        # agent side; clear which checkpoints have already fired so the
        # ORDER question is asked again in the new turn.
        self._fired = set()

    def _stats(self) -> dict | None:
        try:
            return self._stats_fn() if self._stats_fn else None
        except Exception:
            # A stats_fn hiccup must never break tool execution — fail silent.
            return None

    def _message(self, thr: int, stats: dict) -> str:
        from flagscale_agent.react.guard.time_budget import _fmt

        elapsed = _fmt(stats.get("elapsed", 0.0))
        remaining = _fmt(stats.get("remaining", 0.0))
        head = (
            f"({elapsed} used, ~{remaining} left before the harness terminates "
            f"the whole task)."
        )
        return (_MSG_25 if thr <= 25 else _MSG_50) + " " + head

    def _verdict_for(self, stats: dict) -> GuardVerdict | None:
        """Shared crossing logic for check_pre and check_post.

        Returns the block for the most severe crossed-but-unfired threshold,
        marking every crossed threshold as spent (a jump straight past both
        fires only the later checkpoint — the earlier question is subsumed).
        """
        pct = stats.get("pct", 0.0)
        for thr in self._THRESHOLDS:  # (50, 25): high→low
            if pct >= thr and thr not in self._fired:
                for t in self._THRESHOLDS:
                    if pct >= t:
                        self._fired.add(t)
                return GuardVerdict.block(
                    message=self._message(thr, stats),
                    reason=f"first_e2e_run_{thr}pct",
                    category="progress_order",
                    overridable=True,
                )
        return None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None
        stats = self._stats()
        if not stats:
            return None
        return self._verdict_for(stats)

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # check_pre handles the blocks; stay silent post so a crossing never
        # doubles up with a separate post-phase verdict.
        return None

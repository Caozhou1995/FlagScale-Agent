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

"""FirstE2eRunGuard — progress-ORDER checkpoints at 25%, 50% and 75% of budget.

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
  • 75% → cold-consumer checkpoint: the finish line has no room for a
    PREDICTED outcome. If a path was never run, the agent's own "this will
    fail" is an unverified claim about to ship — run it and paste the command
    + output (evidence binding, not a yes/no answer).
  • All blocks are overridable and each fires at most once per turn; a
    jump straight past several fires only the highest (most urgent) one
    (the later question subsumes the earlier ones).
  • Silent when no external wall is enforced (stats_fn → None), exactly
    like TimeBudgetGuard.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

_MSG_25 = (
    "[ProgressOrder] 25% of the enforced wall-clock budget is spent — this is "
    "the WRITE-THROUGH checkpoint, an ORDER question not a pacing tip: does a "
    "minimal usable version of the deliverable ALREADY sit at its required "
    "delivery path? If the design is done and primitives work but the path is "
    "empty, land a crude-but-valid skeleton at the path THIS STEP (required file, "
    "correct name/format, worst-acceptable content) and iterate afterward. Before "
    "deep-diving a hard sub-problem, run the CHEAPEST acceptance probe first: one "
    "command that touches the verifier's actual check (a stub input, a dry-run of "
    "the delivery path) so a wrong assumption about what is graded surfaces at "
    "minutes, not at the wall. Complete-then-write is the inverted order: a run "
    "that dies after design scores zero though every step worked. Before this "
    "tool, state in _override_reason either (a) the path already holds a usable "
    "skeleton (cite path and contents), or (b) a delivery path does not apply, "
    "with the reason, or (c) this call WRITES the first version to the path."
)

_MSG_50 = (
    "[ProgressOrder] 50% of the enforced wall-clock budget is spent — this is "
    "the FIRST-EXERCISE checkpoint: a compile or build that succeeds proves an "
    "intermediate artifact, NOT a working deliverable. Has the simplest "
    "end-to-end path of your deliverable actually been EXECUTED yet (run, "
    "invoked, exercised against a real input)? If not — pipeline never assembled, "
    "output never produced, correctness never observed — run the cheapest "
    "end-to-end exercise NOW: a bug found here costs minutes; at 90% it costs the "
    "task. Before this tool, state in _override_reason either (a) the deliverable "
    "has already been exercised end-to-end (name the run and its observed "
    "output), or (b) this task has no meaningful end-to-end exercise, with the "
    "reason, or (c) this call RUNS that first exercise."
)

_MSG_75 = (
    "[ProgressOrder] 75% of the enforced wall-clock budget is spent — this is "
    "the COLD-CONSUMER checkpoint, an EVIDENCE question not a yes/no one: for "
    "the end-to-end path of your deliverable, have you ACTUALLY RUN it and READ "
    "the result — or are you about to finish having only PREDICTED the outcome? "
    "A prediction is not a run: \"this should work\" / \"this edge case will "
    "fail\" written but never executed leaves you one commit from shipping an "
    "unverified claim. Run it NOW — cold, against the artifact at its delivery "
    "path, not an in-memory copy. Before this tool, state in _override_reason, "
    "for the applicable end-to-end path, the CONCRETE ANCHOR you observed: (a) "
    "the exact command the CONSUMER will run to grade or use your artifact, "
    "COLD-RELOADED from its delivery path, and the OUTPUT it produced (paste "
    "both — \"I ran it\" with no output names nothing), or (b) an explicit "
    "\"none apply\" with the reason — (b) holds only when NO end-to-end path "
    "exists; \"not yet reached it\" is not \"none apply\", that is what (a)/(c) "
    "are for, or (c) this call IS that run — name the exact command it will "
    "execute AND the observable you expect."
)


class FirstE2eRunGuard(Guard):
    """Block once each at 25%, 50% and 75% to force an explicit progress-ORDER
    answer (deliverable written? deliverable exercised?) before the next
    tool call."""

    name = "first_e2e_run"
    priority = 93  # Next to time_budget (92) — advisory-family, low priority.

    # Ordered high→low: the FIRST crossed-but-unfired threshold handled is the
    # most severe pending question.
    _THRESHOLDS = (75, 50, 25)

    def __init__(self, stats_fn):
        """stats_fn() -> dict|None with keys elapsed/budget/remaining/pct.

        None means no external wall is enforced; the guard stays silent.
        Same contract as TimeBudgetGuard's stats_fn.
        """
        self._stats_fn = stats_fn
        self._fired: set[int] = set()
        # Threshold whose block is currently pending release (the one the
        # registry surfaced last). accept_override reads it to bind the
        # evidence check to the 75% checkpoint only.
        self._last_pending_thr: int | None = None

    # ── 75% cold-consumer override: evidence binding ─────────────────────
    # The 75% block message demands the agent paste the command it ran against
    # the delivery artifact and the output it produced. A reason that only
    # SAYS "I ran it" was the observed escape: without a check, prose released
    # the gate in 133/134 losing trials. The release is therefore bound to
    # EVIDENCE the agent already produced: the pasted command must appear,
    # verbatim (normalized), in the text the agent itself emitted or read back
    # during the trial (ctx.messages — the full history the kernel already
    # hands to every guard; no new ledger, no LLM judge).

    @staticmethod
    def _norm(s: str) -> str:
        """Case/whitespace-insensitive comparison form."""
        return " ".join(s.split()).lower()

    @classmethod
    def _recorded_text(cls, ctx: GuardContext) -> str:
        """The AGENT'S OWN ACTIONS, not the guard's echo.

        Scope = recent_tool_history entries (each executed tool call's result)
        PLUS every message whose role is NOT assistant (user/task text and
        tool results carried in messages). Assistant text and guard block
        text are EXCLUDED on purpose: the 75% block message itself names the
        vocabulary an honest answer needs ("run it and read the result"), so
        a reason that merely parrots the guard's own words would otherwise
        self-certify — the loop-echo failure measured in the 2026-10-22
        replay (47/47 historical overrides accepted via guard-echo tokens).
        """
        parts: list[str] = []
        for entry in ctx.recent_tool_history or []:
            if isinstance(entry, dict):
                text = entry.get("result")
                if isinstance(text, str):
                    parts.append(text)
        for msg in ctx.messages or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str):
                            parts.append(text)
        return cls._norm("\n".join(parts))

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
        if thr >= 75:
            body = _MSG_75
        elif thr >= 50:
            body = _MSG_50
        else:
            body = _MSG_25
        return body + " " + head

    def _verdict_for(self, stats: dict) -> GuardVerdict | None:
        """Shared crossing logic for check_pre and check_post.

        Returns the block for the most severe crossed-but-unfired threshold,
        marking every crossed threshold as spent (a jump straight past both
        fires only the later checkpoint — the earlier question is subsumed).
        """
        pct = stats.get("pct", 0.0)
        for thr in self._THRESHOLDS:  # (75, 50, 25): high→low
            if pct >= thr and thr not in self._fired:
                for t in self._THRESHOLDS:
                    if pct >= t:
                        self._fired.add(t)
                # Remember which checkpoint's block is being surfaced so
                # accept_override can bind its evidence check to 75% only.
                self._last_pending_thr = thr
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

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """75% (cold-consumer) releases are EVIDENCE-bound; 25%/50% stay prose.

        A reason must contain a command-shaped token (a backticked span, or a
        line starting with a shell-like invocation) that appears — normalized
        (case/whitespace-insensitive) — in the trial's own recorded text. A
        reason that merely claims a run was made is rejected, and rejecting
        RE-ARMS the 75% checkpoint so the gate can re-block next call (without
        this, one rejected reason would release forever, since _verdict_for
        marks thresholds spent when it fires).
        """
        if not reason or len(reason.strip()) <= 5:
            return False
        if self._last_pending_thr != 75:
            # Non-75% overrides keep the base behavior (prose is enough at the
            # write-through / first-exercise checkpoints — the cost of a wrong
            # answer there is minutes, not the task).
            return True
        recorded = self._recorded_text(ctx)
        reason_norm = self._norm(reason)
        # Candidate commands: backticked spans first, then bare lines that
        # look like invocations.
        candidates: list[str] = []
        import re as _re
        candidates.extend(m.strip("` ") for m in _re.findall(r"`([^`\n]+)`", reason))
        for line in reason.splitlines():
            ls = line.strip()
            if ls.startswith(("$ ", "python", "python3", "./", "bash ", "sh ", "cd ", "make ", "git ", "pip ", "cargo ", "npm ", "java ", "gcc ", "g++ ", "cat ", "ls ", "grep ", "wc ", "pytest", "cmake ")):
                candidates.append(ls)
        if "none apply" in reason_norm:
            return True
        if not candidates:
            # Rejected: re-arm the 75% threshold so the gate can re-block.
            self._fired.discard(75)
            return False
        for cand in candidates:
            c = self._norm(cand.lstrip("$ ").strip())
            if len(c) >= 6 and c in recorded:
                return True
        # Rejected: re-arm the 75% threshold so the gate can re-block.
        self._fired.discard(75)
        return False

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # check_pre handles the blocks; stay silent post so a crossing never
        # doubles up with a separate post-phase verdict.
        return None

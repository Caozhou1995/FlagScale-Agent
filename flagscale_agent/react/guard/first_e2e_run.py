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


def _none_apply(reason: str) -> bool:
    """True when the reason asserts the rung's option (b): no delivery path
    applies (a data file, a passive artifact). Kept narrow on purpose — a
    blanket "none apply" would be a universal escape hatch, so the phrasing
    must be explicit."""
    r = " ".join(reason.split()).lower()
    return ("none apply" in r) or ("no delivery path" in r) or ("no end-to-end path" in r)

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
    "with the reason, or (c) this call WRITES the first version to the path. "
    "The release is MECHANICAL: a write must have EXECUTED this turn (a "
    "write_file/edit_file that ran, or a shell call that landed bytes). An "
    "intention to write later — \"once the design settles I will write it\" — "
    "does NOT release this checkpoint; write the skeleton first, then think."
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

    # How many times one rung may BLOCK before it hands back a non-blocking
    # demand. Two is enough to ask twice (the agent gets a second look after
    # fixing its reason) without letting an unsatisfiable form consume the
    # rest of the trial. Measured 2026-10-26: 8 and 9 consecutive blocks.
    _MAX_DENIALS = 2

    def __init__(self, stats_fn, task_text_fn=None):
        """stats_fn() -> dict|None with keys elapsed/budget/remaining/pct.

        None means no external wall is enforced; the guard stays silent.
        Same contract as TimeBudgetGuard's stats_fn.

        task_text_fn() -> str with the CURRENT task's instruction text (the
        same source VerificationGuard's task_text_fn reads). When the text
        itself states a preservation requirement, the 75% release additionally
        demands identity evidence (see _PRESERVATION_TERMS). Never a
        task-name table — detection runs on the task's own wording at
        runtime.
        """
        self._stats_fn = stats_fn
        self._task_text_fn = task_text_fn
        self._fired: set[int] = set()
        # Rungs whose release was ACCEPTED. A rung's block marks every crossed
        # threshold spent at fire time (`_fired`); only a rung the agent
        # actually released may stay spent. Without this set a rejected reason
        # silently ends the checkpoint for the whole trial — the observed
        # 2026-10-26 escape, where a single-turn task (no reset_turn) let one
        # prose sentence retire the 25% and 50% rungs forever.
        self._released: set[int] = set()
        # Threshold whose block is currently pending release (the one the
        # registry surfaced last). accept_override reads it to bind the
        # evidence check to the 75% checkpoint only.
        self._last_pending_thr: int | None = None
        # Per-rung rejection counts feeding the escape valve in `_rearm`.
        self._denials: dict[int, int] = {}
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

    # ── byte-preservation demand (75% only) ───────────────────────────────
    # Terms are read from the CURRENT task's own instruction text at runtime
    # (task_text_fn); there is NO task-name table. The terms themselves are
    # generic English verbs/adjectives drawn from the phrasing real task
    # instructions use for preservation requirements ("must remain
    # unchanged", "byte-for-byte", "preserve as much ... as possible").
    _PRESERVATION_TERMS = (
        "remain unchanged",
        "remains unchanged",
        "should remain unchanged",
        "must remain",
        "stay unchanged",
        "byte-for-byte",
        "byte for byte",
        "unchanged",
        "unmodified",
        "preserve",
        "must not modify",
        "do not modify",
        "without modifying",
        "keep the",
        "leave the",
    )

    # Command tokens that constitute IDENTITY evidence for preserved files.
    _IDENTITY_CMDS = (
        "md5sum", "sha256sum", "sha1sum", "cmp ", "cksum", "diff -q",
        "git diff --no-index",
    )

    def _demands_preservation(self) -> bool:
        fn = self._task_text_fn
        if not fn:
            return False
        try:
            text = fn() or ""
        except Exception:
            return False
        t = self._norm(text)
        if not t:
            return False
        return any(term in t for term in self._PRESERVATION_TERMS)

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

    # The harness's own writers. This is the AUTHORITATIVE signal for "a file
    # landed": the tool ran and the kernel ledger recorded it. Deliberately NOT
    # a list of shell idioms (heredoc / tee / sed -i ...) — enumerating the
    # shapes a write can take is an open-ended detector that always misses the
    # next one, and a `grep` whose OUTPUT contains such a token would
    # false-pass. A shell call is counted only through its own args (the
    # command it ran), never through its output text.
    _WRITE_TOOLS = ("write_file", "edit_file")

    # Tools whose ONLY purpose is to satisfy another guard's prerequisite.
    # Gating one of these behind a rung the agent can only clear by calling
    # this tool is a closed loop: measured 2026-10-26 (write-compressor),
    # `plan` blocked write_file ("call plan_create first"), `first_e2e_run`'s
    # 25% rung blocked plan_create ("a write must have EXECUTED this turn"),
    # and neither could ever run — 11 consecutive identical blocks while the
    # agent reasoned "deadlock... circular" and burned 5% of the trial. A
    # gate may never block the call that is its own prerequisite.
    _PREREQ_TOOLS = ("plan_create", "plan_update", "memory_write")

    def _has_write_evidence(self, ctx: GuardContext) -> bool:
        """True when the harness's own writer actually executed this turn (or
        this call IS that write).

        Reads the kernel's execution ledger (ctx.recent_tool_history), which
        records only calls that really ran — blocked calls are excluded — so a
        promise to write cannot manufacture an entry. A call that errored does
        not count either: `exec_error` is set by the kernel for those.
        """
        for entry in ctx.recent_tool_history or []:
            if not isinstance(entry, dict) or entry.get("exec_error"):
                continue
            if str(entry.get("name", "")) in self._WRITE_TOOLS:
                return True
        # This very call may be the first landing (the rung's option (c)).
        return ctx.tool_name in self._WRITE_TOOLS

    def _release(self, thr: int):
        """Accept a rung AND every lower rung it subsumes.

        Evidence that satisfies a later checkpoint also answers the earlier
        question (a write seen at 50% is exactly what the 25% write-through
        rung asks about), so an agent that gets one real landing should not be
        blocked again for the rung it already satisfied.
        """
        for t in self._THRESHOLDS:
            if t <= thr:
                self._released.add(t)
        self._last_pending_thr = None

    def _rearm(self, thr: int):
        """Un-spend a rung so a REJECTED release re-blocks instead of silently
        releasing forever. _verdict_for marks the rung spent at fire time; only
        an accepted release may keep it spent.

        Every rejection also counts against the rung's escape valve: a rung the
        agent structurally CANNOT satisfy (the 75% rung asks for a run of an
        artifact that is not finished yet; a task whose environment lacks the
        runtime the artifact needs can never produce the pasted output) must not
        degrade into an unbounded block. Measured 2026-10-26: two trials were
        blocked 8 and 9 times on the 75% rung until the wall-clock ran out, the
        gate converting from a speed bump into the thing that killed the run.
        After _MAX_DENIALS the rung stops blocking and speaks once more as an
        advisory-shaped demand instead, so the agent spends its remaining
        budget shipping what it has rather than answering an unanswerable form.
        """
        if thr in self._fired:
            self._fired.discard(thr)
        self._denials[thr] = self._denials.get(thr, 0) + 1
        self._last_pending_thr = None

    def _pending_thr_for(self, verdict) -> int | None:
        """Recover which rung a verdict belongs to from its reason slug
        (`first_e2e_run_<thr>pct[_exhausted]`).

        The rung number is already encoded in the reason this guard itself
        emitted, so read that naming instead of keeping a parallel map that can
        drift out of sync with `_verdict_for`.
        """
        reason = str(getattr(verdict, "reason", "") or "")
        for thr in self._THRESHOLDS:
            if reason.startswith(f"first_e2e_run_{thr}pct"):
                return thr
        return None

    def note_merged_refusal(self, verdict):
        """Count a refusal that was DELIVERED but merged under another guard's
        message.

        This rung is lower priority than the plan guard, so whenever both fire
        the plan guard's block is the surfaced one and this verdict rode along
        as `also_failing` — seen by the agent, refused by it, yet never counted
        because accept_override is only called for the surfaced block. Measured
        2026-10-26 (write-compressor): 11 consecutive identical 25% blocks, the
        valve never fired, and the plan/write loop stayed deadlocked until the
        wall-clock expired. Counting the merged refusal spends the valve on
        schedule and lets the rung degrade to its non-blocking demand.
        """
        thr = self._pending_thr_for(verdict)
        if thr is None:
            return
        self._last_pending_thr = thr
        self._rearm(thr)

    def reset_turn(self):
        # A new user turn restarts the per-turn budget accounting on the
        # agent side; clear which checkpoints have already fired so the
        # ORDER question is asked again in the new turn.
        self._fired = set()
        self._released = set()
        self._denials = {}
        self._last_pending_thr = None

    def _stats(self) -> dict | None:
        try:
            return self._stats_fn() if self._stats_fn else None
        except Exception:
            # A stats_fn hiccup must never break tool execution — fail silent.
            return None

    def _exhausted_message(self, thr: int, stats: dict) -> str:
        """Non-blocking replacement after a rung's denials are used up.

        Names the honest exit explicitly: the rung stops being a gate, and the
        remaining budget belongs to shipping and stating what was not verified —
        not to satisfying a form the environment may make unsatisfiable.
        """
        from flagscale_agent.react.guard.time_budget import _fmt

        remaining = _fmt(stats.get("remaining", 0.0))
        return (
            f"This checkpoint ({thr}% of the wall-clock) has now blocked {self._MAX_DENIALS} "
            f"times without being released for the same call — it will NOT block again, so "
            f"do not spend more turns on its form. Treat it as settled and switch to the "
            f"honest exit: (1) if your artifact is INCOMPLETE, land the missing bytes first — "
            f"a partial deliverable on disk at the delivery path beats a perfect one that "
            f"never got written; (2) if it exists but cannot be executed here (missing "
            f"runtime/dependency/corpus), STATE plainly in your final message which command "
            f"you tried, its real error output, and what remains unverified — an honest "
            f"'ran it, it failed like this' is an accepted answer, a claim of success is not; "
            f"(3) do not re-issue the same declaration. (~{remaining} left before the harness "
            f"terminates the whole task.)"
        )

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
            if pct >= thr and thr not in self._released:
                # Mark ONLY this rung as surfaced; the higher rungs stay
                # pending until they are themselves crossed AND released, so a
                # jump straight past several thresholds asks the most severe
                # one now and the earlier ones later (they are not subsumed —
                # an un-released rung is a question still owed an answer).
                self._fired.add(thr)
                self._last_pending_thr = thr
                if self._denials.get(thr, 0) >= self._MAX_DENIALS:
                    # Escape valve: this rung has been answered and rejected
                    # as many times as it is allowed to block. Stop spending
                    # the remaining budget on it — hand back a non-blocking
                    # demand naming the honest exit. _release (not a bare add)
                    # also retires every LOWER rung: an agent already past 75%
                    # has been asked the most severe question there is, so
                    # re-asking the 50%/25% rungs afterwards is pure noise that
                    # burns the budget the valve is trying to protect.
                    self._release(thr)
                    return GuardVerdict.inject(
                        message=self._exhausted_message(thr, stats),
                        reason=f"first_e2e_run_{thr}pct_exhausted",
                        category="progress_order",
                    )
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
        # A gate must never block the call that is its own prerequisite. The
        # plan guard requires plan_create before any write, while the 25% rung
        # requires a write before anything else — gating plan_create here makes
        # the two mutually unsatisfiable (measured deadlock, see _PREREQ_TOOLS).
        # These tools cannot land the deliverable, so letting them through
        # costs the checkpoint nothing: the write it asks for is still demanded
        # on the very next call.
        if ctx.tool_name in self._PREREQ_TOOLS:
            return None
        stats = self._stats()
        if not stats:
            return None
        return self._verdict_for(stats)

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """75% (cold-consumer) releases are EVIDENCE-bound; 25%/50% are
        PROGRESS-bound.

        Observed escape this closes (2026-10-26, two independent trials): the
        25% and 50% rungs used to release on ANY prose reason, and both losing
        runs wrote a sentence like "next step: write the encoder" and kept
        reasoning for another 45% of the wall-clock without a single tool call
        — one run's first delivery was at 76% of budget, the other's real
        implementation only appeared at 60%. A promise to write is not a
        write. So the release is bound to the trial's own EXECUTION LEDGER
        (ctx.recent_tool_history): the rung releases only when a write-shaped
        tool actually executed (or this very call IS that write), or the agent
        states "none apply" for a deliverable with no path. Rejection RE-ARMS
        the rung so the gate can block again instead of staying silent.

        The 75% rung keeps its stricter evidence binding: the pasted command
        must appear in the trial's own recorded text.
        """
        if not reason or len(reason.strip()) <= 5:
            # A too-short reason is still a DENIAL: it must count against the
            # escape valve, otherwise an agent that re-issues a five-character
            # filler is blocked forever (the exact livelock the valve exists to
            # stop). Count via _rearm so the budget is spent either way.
            if self._last_pending_thr is not None:
                self._rearm(self._last_pending_thr)
            return False
        thr = self._last_pending_thr
        if thr is None:
            return True
        # A real landing answers EVERY rung, including the 75% one. The 75%
        # question is "have you run the thing you are about to ship?" — but a
        # run needs the artifact to exist first, and gating the landing on a
        # run of the not-yet-landed artifact is a closed loop. Measured cost of
        # missing this (2026-10-26, gpt2-codegolf): the first write landed a
        # 1200-byte skeleton, every APPEND that would have completed it was
        # blocked 9 times on the 75% rung, the agent diagnosed it itself
        # ("guards keep blocking appends") and died at 91% budget with the
        # skeleton still the deliverable. So: progress evidence releases.
        if self._has_write_evidence(ctx):
            self._release(thr)
            self._denials.pop(thr, None)
            return True
        if thr != 75:
            # 25% / 50%: progress evidence, not prose.
            if _none_apply(reason):
                self._release(thr)
                return True
            self._rearm(thr)
            return False
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
            self._release(75)
            return True
        if not candidates:
            # Rejected: re-arm the 75% threshold so the gate can re-block.
            self._rearm(75)
            return False
        for cand in candidates:
            c = self._norm(cand.lstrip("$ ").strip())
            if len(c) >= 6 and c in recorded:
                # 75% + a preservation-demanding task: the pasted command
                # must additionally be IDENTITY evidence (md5/cmp/sha/diff -q)
                # for the files the task asks to keep unchanged — "I ran the
                # pipeline" is not identity proof, the byte-preservation miss
                # shipped precisely because the final exercise never compared
                # the untouched inputs.
                if (self._last_pending_thr == 75
                        and self._demands_preservation()
                        and not any(cmd.strip() in c
                                    for cmd in self._IDENTITY_CMDS)):
                    self._rearm(75)
                    return False
                self._release(75)
                return True
        # Rejected: re-arm the 75% threshold so the gate can re-block.
        self._rearm(75)
        return False

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # check_pre handles the blocks; stay silent post so a crossing never
        # doubles up with a separate post-phase verdict.
        return None

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

"""Guard system — behavioral constraints for the agent.

Guards fire at two points:
- pre: Before tool execution (can block or inject)
- post: After tool execution (can inject context)

Three action levels:
- inject: advisory message appended to context (does not block)
- block: prevents tool execution, LLM can override with _override_reason
- escalate: prevents tool execution, cannot be overridden
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from flagscale_agent.react import display
from typing import Literal, Any


@dataclass
class GuardContext:
    """Read-only snapshot passed to guards."""

    # Tool context
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str | None = None
    turn_count: int = 0
    recent_tool_names: list[str] = field(default_factory=list)
    recent_tool_history: list[dict] = field(default_factory=list)
    context_pressure: float = 0.0
    evictable_indexes: list[int] = field(default_factory=list)

    # Full message history
    messages: list[dict] = field(default_factory=list)

    # LLM response text
    assistant_text: str = ""

    # True only when assistant_text is the text the LLM produced THIS turn's
    # current iteration (completion-path consultation), not stale text scanned
    # back out of history at the top of a fresh iteration. Completion gates that
    # match on assistant_text.endswith("[TASK_COMPLETE]") MUST require this, or a
    # prior turn's trailing sentinel re-triggers them at every new turn's top.
    llm_responded: bool = False

    # LLM classify function
    classify_fn: Any = None

    # The unified LLM judge (flagscale_agent.react.judge.Judge). Guards use it
    # for the SEMANTIC half of a check -- reading prose that has no fixed shape
    # (e.g. "which requirements does this statement state?", "which requirement
    # does this claim address?"). None when no provider is configured; every
    # caller must fail open on None so a dead provider cannot manufacture a
    # block.
    judge: Any = None

    # Override reason from LLM
    override_reason: str = ""


@dataclass
class GuardVerdict:
    """What the guard wants the agent to do."""

    action: Literal["allow", "block", "inject", "escalate"]
    message: str
    reason: str
    category: str  # For inject deduplication
    # Declare that this inject carries data tied to THIS invocation's trigger
    # (e.g. a fresh execution-domain profile built for the exact command being
    # checked). If a block surfaces in the same resolve, the registry merges
    # this inject into the block message instead of dropping it. Generic
    # periodic injects keep the default (False): they
    # are suppressed by blocks to keep the override turn focused on the block's
    # own criteria (test_block_suppresses_injects contract).
    attach_on_block: bool = False
    # Owner guard name, stamped by the registry when a block/escalate is selected
    # for presentation. Used both to name the owner in the message and to scope
    # override: an _override_reason only releases the guard whose name matches the
    # currently presented block, so a reason written for guard A can never release
    # guard B.
    guard_name: str = ""
    # A block is overridable by default (_override_reason releases it). Set
    # overridable=False for a block that must be released by a concrete corrective
    # ACTION (a specific tool call), not by a text/tool-arg override. Unlike
    # escalate (a dead-end that tells the agent to stop and ask the user), a
    # non-overridable block RE-PROMPTS: the agent can recover on its own by taking
    # the required action. Used by the single-shot completion gate, whose only exit
    # is calling plan_create — a text-inline override the model cannot reliably
    # emit was letting bare [TASK_COMPLETE] livelock the loop.
    overridable: bool = True

    @classmethod
    def block(cls, message: str, reason: str, category: str,
              overridable: bool = True) -> GuardVerdict:
        return cls(action="block", message=message, reason=reason,
                   category=category, overridable=overridable)

    @classmethod
    def inject(cls, message: str, reason: str, category: str,
               attach_on_block: bool = False) -> GuardVerdict:
        return cls(action="inject", message=message, reason=reason, category=category,
                   attach_on_block=attach_on_block)

    @classmethod
    def escalate(cls, message: str, reason: str, category: str) -> GuardVerdict:
        return cls(action="escalate", message=message, reason=reason, category=category)


_OVERRIDE_HINT = (
    '\n\n⚠️ OVERRIDE REQUIRED: Add "_override_reason" to your next tool call to proceed.\n'
    "DO: re-call the same tool with an extra parameter:\n"
    '  {"command": "...", "_override_reason": "reason why this is safe/justified"}\n'
    "DON'T: explain in text. Only _override_reason in tool_args works."
)

_TEXT_OVERRIDE_HINT = (
    '\n\n⚠️ OVERRIDE REQUIRED: To override this gate and complete, add an inline '
    '_override_reason to your [TASK_COMPLETE] message.\n'
    "DO: re-emit with a reason on the same line or next line:\n"
    '  [TASK_COMPLETE]\n  _override_reason: <reason why no plan was warranted>\n'
    "DON'T: re-emit a bare [TASK_COMPLETE] — it will be blocked again."
)

_ESCALATE_HINT = (
    "\n\n🚫 ESCALATED: This tool call is blocked and cannot be overridden.\n"
    "DO NOT retry the same tool call — it will be blocked again.\n"
    "Choose a different approach. If you must proceed this way, stop and ask the user."
)

# A block message is a template that enumerates EVERY sub-rule a structured
# declaration could violate; the failing check category is carried only in
# verdict.reason (internal, used for override matching). Without that slug the
# agent can only GUESS which kind of rule it tripped and may re-fire the same
# declaration until the budget runs out. Naming the slug is generic (it describes
# the agent's OWN declaration, never the task's answer) and narrows the guess-retry
# loop to a bounded corrective edit. Appended to every surfaced block/escalate.
_REASON_DIAGNOSTIC = (
    "\n\n[failing check: {reason}] — this slug names the check CATEGORY the "
    "declaration failed. Every failing check evaluated at this decision point is "
    "listed above, so you can fix them together instead of one at a time."
)

# Multi-block header. The registry merges EVERY failing check at a decision
# point into ONE message: a guard cascade (surface one block, wait for the fix,
# surface the next) makes the agent pay one full round trip per violation and
# leaves the remaining violations invisible until it has already spent the fix
# on the first. Reporting all of them at once costs the same single round trip
# and lets one corrective edit satisfy the whole set. Nothing is dropped: each
# block keeps its full message and its own failing-check slug.
_MULTI_BLOCK_HEADER = (
    "{n} independent requirements failed at this decision point. Fix them ALL in "
    "one pass — they are reported together so you do not have to discover them "
    "one round trip at a time:"
)

# Convergence escalation for a REPEATED block. The block channel has no
# mechanism to break a guess-retry loop: when the agent keeps re-emitting a
# declaration that trips the SAME guard with the SAME reason, the registry
# surfaces the same message every turn and the agent re-fires it until the wall
# clock dies (observed: a task finished its WORK but burned its whole budget on
# 42 identical blocks of one hedge sub-rule). This is the block-channel analogue
# of the output-saturation breaker in the kernel: after N consecutive identical
# (owner, reason) blocks with no intervening progress, ATTACH a directive that
# names the repetition and tells the agent to change the FORM, not the wording.
# Generic: it describes the agent's OWN declaration and the slug the block
# already carries — never the task's answer, never a regex, never a task name.
REPEAT_CONVERGENCE_AT = 3
_REPEAT_CONVERGENCE_HARD_AT = 6
_REPEAT_CONVERGENCE = (
    "\n\n🔁 CONVERGENCE: the SAME block ({reason}) has now been surfaced {n} "
    "times in a row with no progress past it. Re-emitting the same declaration "
    "again will be blocked the SAME way — STOP rewording it and change its FORM:\n"
    "  1. Every required key is NAMED in the requirement above. Put each named "
    "key LITERALLY at the nesting level the block shows (a key read from the "
    "inner object but written at the outer level, or vice-versa, re-fails).\n"
    "  2. Match each required value's JSON TYPE to what the block shows — an "
    "object where a string is required (or a string where an object/dict is "
    "required) re-fails the same check.\n"
    "  3. If you cannot construct a conforming declaration, take the "
    "ALTERNATIVE EXIT the block message names instead of retrying this one."
)
_REPEAT_CONVERGENCE_HARD = (
    "\n\n🛑 CONVERGENCE (HARD): {n} identical blocks ({reason}) — you are stuck "
    "in a retry loop that will NOT clear by re-issuing this call. Do NOT emit "
    "this same tool call again. Either (a) make a STRUCTURALLY DIFFERENT edit "
    "that satisfies the named keys exactly as the block shows, or (b) abandon "
    "this path and take the alternative exit the block names. Retrying the "
    "identical declaration is the one action guaranteed to fail again."
)


class Guard(abc.ABC):
    """Base class for all guards.

    Three action levels:
    - inject: advisory, does not block tool execution
    - block: blocks execution, LLM can override with _override_reason
    - escalate: blocks execution, cannot be overridden
    """

    name: str = "unnamed"
    priority: int = 50  # lower = higher priority

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Pre-execution check. Return block/escalate to prevent, inject to warn."""
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Post-execution check. Return inject to add context."""
        return None

    def check_all_pre(self, ctx: GuardContext) -> list[GuardVerdict]:
        """Every pre-check violation of this guard, not just the first.

        A guard that internally evaluates several INDEPENDENT sub-rules (e.g. one
        scanner per command family) naturally short-circuits: the first rule that
        fires returns and the rest are never evaluated, so the agent fixes one
        violation only to discover the next on the following round trip. Override
        this to run the whole rule set and return EVERY violation found; the
        registry merges them into a single message.

        Default wraps check_pre so single-rule guards need no change.
        """
        verdict = self.check_pre(ctx)
        if verdict is None:
            return []
        return [verdict]

    def check_all_post(self, ctx: GuardContext) -> list[GuardVerdict]:
        """Every post-check violation of this guard. See check_all_pre."""
        verdict = self.check_post(ctx)
        if verdict is None:
            return []
        return [verdict]

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Validate LLM's override reason. Only called for block verdicts.

        Default: accept any reason longer than 5 chars.
        Override for stricter validation.
        """
        return bool(reason and len(reason.strip()) > 5)

    def reset_turn(self):
        """Called at the start of each new user message. Clear per-turn state."""
        pass

    def block_not_surfaced(self):
        """Called by the registry when THIS guard returned a block/escalate
        verdict that was NOT the one surfaced to the agent — a higher-ranked
        block won the same resolve. A guard whose block has a one-shot side
        effect (e.g. consuming a pending-demand flag when the block is shown)
        must UNDO that side effect here, so the demand survives and re-fires
        once the winning block clears. Default: no-op."""
        pass


class GuardRegistry:
    """Manages all guards, runs them in priority order, deduplicates injects."""

    def __init__(self):
        self._guards: list[Guard] = []
        # Name of the guard whose block was surfaced to the agent on the most
        # recent resolve. An override reason may release ONLY this guard's block
        # (you can only override a block you were actually shown). Prevents a
        # reason written for guard A from silently releasing guard B that fired
        # for the first time this turn — the phantom-override crosstalk (Bug C).
        self._last_surfaced: str | None = None
        self._last_surfaced_reason: str | None = None
        # Repeat-convergence tracker (block channel). Counts CONSECUTIVE resolves
        # that surfaced the SAME (owner, reason) block with no intervening progress
        # (no override release, no different block). A repeated identical block is
        # the tell of a guess-retry loop: the agent re-emits a declaration that
        # re-fails the same way, and without an escalating directive it does so
        # until the wall clock dies (observed: 42 identical hedge blocks). At the
        # thresholds below, an escalating convergence directive is appended. It
        # resets the moment anything changes (override released a block, the
        # surfaced (owner, reason) differs, or a different guard wins).
        self._repeat_key: tuple[str, str] | None = None
        self._repeat_count: int = 0

    def register(self, guard: Guard):
        self._guards.append(guard)
        self._guards.sort(key=lambda g: g.priority)

    @staticmethod
    def _severity_rank(verdict: GuardVerdict) -> int:
        """Hardest-first ordering key (lower = surfaced first).

        escalate (0) > non-overridable block (1) > overridable block (2).
        A non-overridable block MUST be surfaced before an overridable one at the
        same decision point: otherwise the agent overrides the soft block, retries,
        and only then hits the hard wall it could have addressed first — the
        crosstalk/deadlock (Bug B).
        """
        if verdict.action == "escalate":
            return 0
        if verdict.action == "block" and not verdict.overridable:
            return 1
        return 2  # overridable block

    def _resolve(self, ctx: GuardContext, phase: str) -> GuardVerdict | None:
        """Shared block/escalate resolution for pre/post checks.

        collect-all + severity-rank + owner-scoped override:
        - Collect ALL block/escalate verdicts paired with their owning guard
          (injects still merge, deduplicated by category). Guards that evaluate
          several independent sub-rules contribute EVERY violation they find via
          check_all_pre/check_all_post, not just their first.
        - Rank hardest-first (escalate > non-overridable > overridable); ties
          broken by guard priority (guards are pre-sorted, so stable sort keeps
          priority order within a severity tier).
        - Present EVERY failing requirement in ONE message: the top-ranked block
          leads, and the rest are merged beneath it with their own owner tag and
          failing-check slug. They are delivered, not deferred — an agent that
          fixes only the first and retries pays one full round trip per violation
          for information it could have had at once.
        - Offer _override_reason ONLY to the leading block's owning guard via
          guard.accept_override — a reason written for one guard can never release
          another (Bug B root cause: global override_reason fed to whichever guard
          happened to win).
        - Stamp guard_name so the message names the owner.
        """
        blocks: list[tuple[Guard, GuardVerdict]] = []
        inject_messages: list[str] = []
        inject_categories_seen: set[str] = set()
        # (guard, verdict) pairs for trigger-bound injects that must survive a
        # concurrent block.
        inject_pairs: list[tuple[Guard, GuardVerdict]] = []
        first_reason = ""

        check = (lambda g: g.check_pre(ctx)) if phase == "pre" else (lambda g: g.check_post(ctx))
        # Collect EVERY violation per guard, not just its first. A guard with
        # several independent sub-rules short-circuits check_pre at the first hit;
        # reporting only that one makes the agent fix a violation, retry, and
        # discover the next on a later round trip. check_all_* lets one guard
        # contribute all of its violations to the same message.
        collect = ((lambda g: g.check_all_pre(ctx)) if phase == "pre"
                   else (lambda g: g.check_all_post(ctx)))

        for guard in self._guards:
            seen_rule: set[tuple[str, str]] = set()
            for verdict in collect(guard):
                if verdict is None:
                    continue

                if verdict.action in ("block", "escalate"):
                    # One rule can be detected by more than one layer inside a
                    # guard (e.g. a broad-grep regex layer AND the lexical
                    # scanner confirm the same violation). Reporting it twice
                    # from the same guard is noise, not extra information: the
                    # agent would fix one thing and be told it twice. Keep the
                    # first (richer) report per rule; distinct rules still all
                    # appear.
                    _key = (guard.name, verdict.reason)
                    if _key in seen_rule:
                        continue
                    seen_rule.add(_key)
                    verdict.guard_name = guard.name
                    blocks.append((guard, verdict))
                elif verdict.action == "inject":
                    cat = verdict.category
                    if cat and cat in inject_categories_seen:
                        continue
                    if cat:
                        inject_categories_seen.add(cat)
                    inject_messages.append(verdict.message)
                    inject_pairs.append((guard, verdict))
                    if not first_reason:
                        first_reason = verdict.reason

        if blocks:
            # Stable sort by severity; priority ties already ordered by _guards sort.
            blocks.sort(key=lambda gv: self._severity_rank(gv[1]))

            # Walk hardest-first. The reason (if any) is offered ONLY to the guard
            # currently surfaced — never carried to another guard. If the surfaced
            # guard accepts, it is released and we drop to the next hardest block;
            # any other block requires its OWN reason next turn. accept_override is
            # called at most once per guard here, so no double-eval side effects.
            surfaced: GuardVerdict | None = None
            surfaced_guard: Guard | None = None
            # Blocks that ALSO failed this decision point and are merged into the
            # surfaced message. They are DELIVERED, not deferred: the agent sees
            # every failing requirement in one message and can satisfy them in one
            # corrective pass.
            also_failing: list[tuple[Guard, GuardVerdict]] = []
            for idx, (guard, verdict) in enumerate(blocks):
                # An override reason may release a block ONLY if that block is the
                # one this registry surfaced to the agent last turn. You cannot
                # override a block you were never shown. A guard that fires for the
                # FIRST time this turn (e.g. BackupGuard on the turn's first shell)
                # was never presented, so a reason authored for a DIFFERENT guard
                # must not fall through and release it — that is the phantom
                # override crosstalk (Bug C: printed "override: backup" though no
                # backup block was ever displayed).
                if (
                    verdict.action == "block"
                    and verdict.overridable
                    and ctx.override_reason
                    and guard.name == self._last_surfaced
                    and verdict.reason == self._last_surfaced_reason
                    and guard.accept_override(ctx.override_reason, ctx)
                ):
                    display.guard_overridden(guard.name, ctx.override_reason)
                    # Intra-guard gate-crosstalk prevention: a single guard may
                    # have cascading internal gates (e.g. VerificationGuard's 5
                    # gates on plan_update(complete)). When the first gate's block
                    # is released by an override, the guard's check_pre may now
                    # return a DIFFERENT block (the next gate). Re-call check_pre
                    # and if it returns a new block with a different reason, report
                    # THAT instead of silently releasing. This prevents an
                    # override reason written for Gate 1 from releasing Gate 5.
                    new_verdict = check(guard)
                    if (
                        new_verdict is not None
                        and new_verdict.action in ("block", "escalate")
                        and new_verdict.reason != verdict.reason
                    ):
                        new_verdict.guard_name = guard.name
                        if surfaced is None:
                            surfaced, surfaced_guard = new_verdict, guard
                        else:
                            also_failing.append((guard, new_verdict))
                    continue  # released; fall through to next block
                if surfaced is None:
                    surfaced, surfaced_guard = verdict, guard
                else:
                    also_failing.append((guard, verdict))

            if surfaced is not None:
                # Name the owner guard so the agent knows WHICH guard is blocking
                # and whose criteria an override reason must satisfy — a reason for
                # another guard will not release this one.
                owner = surfaced.guard_name or surfaced_guard.name if surfaced_guard else surfaced.guard_name
                owner_tag = f"\n\n[blocked by guard: {owner}]" if owner else ""
                # Name the failing sub-rule (verdict.reason) so the agent does not
                # have to guess WHICH clause of the template block it tripped. The
                # template lists every sub-rule; without this the agent re-fires the
                # same declaration and burns budget in a guess-retry loop. The slug
                # describes the agent's OWN declaration — no task/grader information.
                reason_tag = (
                    _REASON_DIAGNOSTIC.format(reason=surfaced.reason)
                    if surfaced.reason else ""
                )
                # Attach the appropriate hint to the surfaced (still-blocking) block.
                if surfaced.action == "block" and surfaced.overridable:
                    surfaced.message += owner_tag + reason_tag
                    # Add the override hint ONLY when no reason was supplied this
                    # turn. If a reason WAS given but did not release this block
                    # (rejected by accept_override, or written for a different
                    # guard), do not re-nag with the hint — the guard's own message
                    # explains what it needs, and the owner_tag names it.
                    if not ctx.override_reason:
                        if not ctx.tool_name:
                            surfaced.message += _TEXT_OVERRIDE_HINT
                        else:
                            surfaced.message += _OVERRIDE_HINT
                elif surfaced.action == "escalate":
                    surfaced.message += owner_tag + reason_tag + _ESCALATE_HINT
                else:
                    # Non-overridable block: no override hint, but still name the
                    # owner so the agent addresses the right guard's required action.
                    surfaced.message += owner_tag + reason_tag
                # ── Merge every OTHER failing requirement into the same message ──
                # A cascade (surface one block, wait for the fix, surface the next)
                # costs one full round trip PER violation and hides the rest until
                # the agent has already spent a fix on the first. Report them all
                # here instead: nothing is dropped, each keeps its full message and
                # its own failing-check slug, and one corrective pass can satisfy
                # the whole set. Ordering stays hardest-first.
                if also_failing:
                    extra = []
                    for ex_guard, ex_verdict in also_failing:
                        ex_owner = ex_verdict.guard_name or ex_guard.name
                        ex_text = ex_verdict.message
                        if ex_owner:
                            ex_text += f"\n\n[blocked by guard: {ex_owner}]"
                        if ex_verdict.reason:
                            ex_text += _REASON_DIAGNOSTIC.format(reason=ex_verdict.reason)
                        if ex_verdict.action == "escalate":
                            ex_text += _ESCALATE_HINT
                        extra.append(ex_text)
                    surfaced.message += (
                        "\n\n" + "=" * 62 + "\n"
                        + _MULTI_BLOCK_HEADER.format(n=len(also_failing) + 1)
                        + "\n" + "=" * 62 + "\n\n"
                        + "\n\n" + "-" * 62 + "\n\n".join(extra)
                    )
                # Injects marked attach_on_block carry data built for THIS
                # invocation (e.g. an execution-domain profile for the exact
                # command under check). Dropping them on a concurrent block
                # loses numbers the agent needs at the decision point. Merge
                # those into the block message; generic periodic injects stay
                # suppressed (see test_block_suppresses_injects contract: the
                # override turn must focus on the block's own criteria).
                attached = [
                    ov.message for (og, ov) in inject_pairs
                    if ov.attach_on_block and og.name != surfaced_guard.name
                ]
                if attached:
                    surfaced.message += (
                        "\n\n[concurrent trigger-bound guard advisory]\n"
                        + "\n\n".join(attached)
                    )
                # Remember which guard (and which internal gate/reason) we surfaced.
                # Next turn, only THIS guard's block with THIS exact reason may be
                # released by an override reason (see loop above). Tracking the
                # reason prevents intra-guard gate-crosstalk: an override written
                # for Gate 1 (e.g. task_complete_premise_recheck) must not release
                # Gate 2 (e.g. task_complete_observation_demand) even though both
                # belong to the same VerificationGuard.
                self._last_surfaced = owner or (surfaced_guard.name if surfaced_guard else None)
                self._last_surfaced_reason = surfaced.reason
                # ── Repeat-block convergence (block channel) ──
                # Track CONSECUTIVE resolves surfacing the SAME (owner, reason).
                # A repeated identical block is a guess-retry loop: the agent
                # re-emits a declaration that re-fails identically and, with no
                # escalating signal, keeps doing so to the wall (observed: 42
                # identical blocks). Past the thresholds, append a directive that
                # names the repetition and demands a FORM change (or the
                # alternative exit). The streak is reset ONLY by real progress —
                # an explicit reset_repeat_convergence() the kernel calls when a
                # tool actually EXECUTES (see kernel.py). It must NOT reset on a
                # "no block this resolve": the top-of-loop check_pre runs with an
                # empty ctx every iteration between real blocks and returns None
                # there, which would zero the streak before it could accumulate.
                _key = (surfaced.guard_name or "", surfaced.reason or "")
                if _key == self._repeat_key:
                    self._repeat_count += 1
                else:
                    self._repeat_key = _key
                    self._repeat_count = 1
                if (surfaced.action == "block"
                        and REPEAT_CONVERGENCE_AT > 0
                        and self._repeat_count >= REPEAT_CONVERGENCE_AT):
                    if (_REPEAT_CONVERGENCE_HARD_AT > 0
                            and self._repeat_count >= _REPEAT_CONVERGENCE_HARD_AT):
                        surfaced.message += _REPEAT_CONVERGENCE_HARD.format(
                            n=self._repeat_count, reason=surfaced.reason)
                    else:
                        surfaced.message += _REPEAT_CONVERGENCE.format(
                            n=self._repeat_count, reason=surfaced.reason)
                return surfaced

        if inject_messages:
            return GuardVerdict.inject(
                "\n\n".join(inject_messages),
                reason=first_reason or "multi_guard_inject",
                category="merged"
            )
        return None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' pre-checks. Hardest block surfaces first (severity-ranked,
        owner-scoped override); injects merge. See _resolve."""
        return self._resolve(ctx, "pre")

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' post-checks. Same resolution as check_pre. See _resolve."""
        return self._resolve(ctx, "post")

    def reset_turn(self):
        """Reset per-turn state for all guards.

        Note: _last_surfaced is intentionally NOT cleared here — it must persist
        across turns so an override reason on turn N+1 can release the block that
        was surfaced on turn N (the block and its override live in different turns).
        """
        for guard in self._guards:
            guard.reset_turn()

    def reset_repeat_convergence(self):
        """Clear the repeat-block convergence streak.

        Called by the kernel the moment a tool actually EXECUTES (real progress
        past whatever blocked before), so only a genuine attempt-to-advance —
        not the per-iteration top-of-loop no-block resolve — zeroes the streak.
        """
        self._repeat_key = None
        self._repeat_count = 0

    @property
    def guards(self) -> list[Guard]:
        return list(self._guards)

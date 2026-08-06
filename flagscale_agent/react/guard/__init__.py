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

"""Guard system — behavioral constraints with lifecycle hooks.

Guards fire at three points:
- pre: Before tool execution (can block)
- post: After tool execution (can inject messages)
- strategic: At review points (can redirect plan)

v2: Guard lifecycle and inject deduplication.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from flagscale_agent.react import display
from typing import Literal, Any




@dataclass
class GuardContext:
    """Read-only snapshot passed to guards.

    Contains tool context, state machine info, and LLM classify function.
    """

    # Tool context
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str | None = None
    turn_count: int = 0
    recent_tool_names: list[str] = field(default_factory=list)
    recent_tool_history: list[dict] = field(default_factory=list)  # [{tool, args_summary, result_summary}]
    context_pressure: float = 0.0
    evictable_indexes: list[int] = field(default_factory=list)  # Available indexes for eviction

    # Full message history (for context-aware guards)
    messages: list[dict] = field(default_factory=list)  # Current history (including evicted placeholders)

    # LLM response text (for guards that need to scan assistant replies)
    assistant_text: str = ""

    # LLM classify function
    classify_fn: Any = None  # (category: str, context: dict) -> Any

    # Experiment context
    experiment_compare_fn: Any = None
    experiment_diff_fn: Any = None
    current_experiment_name: str = ""

    # Override reason: LLM declares why a potentially-blocked call is justified
    override_reason: str = ""


@dataclass
class GuardVerdict:
    """What the guard wants the agent to do."""

    action: Literal["allow", "block", "inject_msg", "escalate", "redirect"]
    message: str = ""
    reason: str = ""
    metadata: dict = field(default_factory=dict)
    # v2: category tag for deduplication
    category: str = ""  # e.g. "read_stall", "loop", "plan_needed"

    @classmethod
    def block(cls, message: str, reason: str = "", category: str = "") -> GuardVerdict:
        return cls(action="block", message=message, reason=reason, category=category)

    @classmethod
    def inject(cls, message: str, reason: str = "", category: str = "") -> GuardVerdict:
        return cls(action="inject_msg", message=message, reason=reason, category=category)

    @classmethod
    def escalate(cls, message: str, reason: str = "", category: str = "") -> GuardVerdict:
        return cls(action="escalate", message=message, reason=reason, category=category)

    @classmethod
    def redirect(cls, message: str, reason: str = "", metadata: dict = None) -> GuardVerdict:
        return cls(action="redirect", message=message, reason=reason, metadata=metadata or {})


class Guard(abc.ABC):
    """Base class for all guards.

    Lifecycle:
    - Guards accumulate state over time (across iterations/turns).
    - Guards that block/inject must define when they are SATISFIED (concern resolved).
    - Guards must support DECAY: if N iterations pass without re-triggering, state resets.
    - All guards are overridable by default (LLM can bypass with a reason).
    - Inject messages have a MAX_INJECT_REPEATS cooldown to avoid context pollution.
    """

    name: str = "unnamed"
    priority: int = 50  # lower = higher priority
    activate_on_tools: set[str] | None = None  # None = all tools

    # Override mechanism: if True, LLM can bypass this guard's block by providing
    # a reason in tool_args["_override_reason"]. The guard's accept_override()
    # method decides whether the reason is sufficient.
    # v3: Default changed to True — all guards should be overridable to prevent
    # death spirals. Guards can set to False only for safety-critical blocks.
    overridable: bool = True

    # v3: Inject cooldown — max times the same inject message category fires
    # before going silent. Prevents context pollution from repetitive warnings.
    # v5: Escalation chain — inject → block → escalate
    # After this many injects without satisfaction, upgrade to block
    escalate_after: int = 3  # inject count before escalating to block

    # v3: Decay window — if this many iterations pass without the guard
    # re-triggering (returning a non-None verdict), persistent state is reset.
    # Set to 0 to disable decay (state persists indefinitely).
    decay_after_idle: int = 10

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # v3: Patch subclass __init__ to ensure lifecycle attrs exist even if
        # subclass doesn't call super().__init__()
        original_init = cls.__dict__.get('__init__')
        if original_init:
            def patched_init(self, *args, _original=original_init, **kw):
                # Ensure lifecycle attrs exist BEFORE subclass init
                if not hasattr(self, '_inject_counts'):
                    self._inject_counts = {}
                    self._iterations_since_trigger = 0
                _original(self, *args, **kw)
                # Also ensure they exist AFTER (in case subclass clobbers)
                if not hasattr(self, '_inject_counts'):
                    self._inject_counts = {}
                    self._iterations_since_trigger = 0
            cls.__init__ = patched_init

    def __init__(self):
        # v5: Lifecycle tracking (managed by GuardRegistry)
        self._inject_counts: dict[str, int] = {}  # category -> times fired (for escalation)
        self._iterations_since_trigger: int = 0

    def should_activate(self, ctx: GuardContext) -> bool:
        """Check if this guard should run for the current context."""
        if self.activate_on_tools and ctx.tool_name not in self.activate_on_tools:
            return False
        return True

    def is_satisfied(self, ctx: GuardContext) -> bool:
        """Return True if the guard's concern has been addressed.

        When satisfied, the guard resets its persistent state and stops firing.
        Subclasses SHOULD override this to define their satisfaction condition.
        Default: never satisfied (backward compat).
        """
        return False

    def reset_state(self):
        """Reset all persistent state to initial values.

        Called when:
        - is_satisfied() returns True
        - Decay window expires (no re-triggers for N iterations)
        - LLM successfully overrides the guard

        Subclasses MUST override this to reset their specific state.
        """
        self._inject_counts.clear()
        self._iterations_since_trigger = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Pre-execution check. Return block/inject to prevent or warn."""
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Post-execution check. Return inject to add context."""
        return None

    def check_strategic(self, ctx: GuardContext) -> GuardVerdict | None:
        """Strategic review check. Return redirect to change plan."""
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Evaluate whether the LLM's override reason is sufficient to bypass a block.

        Only called when overridable=True and the guard returned a block verdict.
        Default: accept any non-empty reason. Override for stricter validation.

        v3: When override is accepted, guard state is reset to prevent re-blocking
        on the same concern.
        """
        accepted = bool(reason and reason.strip())
        if accepted:
            self.reset_state()
        return accepted

    def notify_blocked(self, ctx: GuardContext):
        """Called when a tool call was blocked externally (e.g., by another guard)."""
        pass

    def was_inject_effective(self, ctx: GuardContext) -> bool | None:
        """Called after each tool execution to check if a previous inject changed behavior.

        Return:
        - True: the agent responded to the inject (e.g., used memory after reminder)
        - False: the agent ignored the inject (keeps doing unrelated things)
        - None: can't tell yet / not applicable

        Subclasses SHOULD override this to define their own effectiveness criteria.
        Default: None (no opinion — won't mark effective or ineffective).
        """
        return None

    def reset_iteration(self):
        """Called at the start of each iteration (LLM+tool loop) within a turn.

        A "turn" is one user message → completion (may contain many iterations).
        An "iteration" is one LLM call + one tool execution within that turn.

        Most guards should NOT reset state here — they need to track patterns
        across iterations (e.g., consecutive errors, read streaks).
        Only reset per-iteration dedup caches or similar ephemeral state.
        """
        pass

    # Backward compat: subclasses may override either name
    reset_turn = reset_iteration

    def reset_new_turn(self):
        """Called once at the start of a new turn (new user message).

        Unlike reset_iteration (called per LLM+tool loop), this is called once
        per user message. Guards that track cross-iteration patterns within a
        turn should reset here so state doesn't leak between turns.
        """
        pass

    # --- v3: Lifecycle helpers (called by GuardRegistry) ---

    def _record_trigger(self, category: str = ""):
        """Record that this guard fired (returned non-None verdict)."""
        self._iterations_since_trigger = 0
        if category:
            self._inject_counts[category] = self._inject_counts.get(category, 0) + 1

    def _get_escalation_level(self, category: str = "") -> str:
        """Determine current escalation level for this category.

        Returns:
        - "inject": normal soft advisory (default)
        - "block": inject has been ignored escalate_after times → block tool
        - "escalate": block was overridden but behavior unchanged → hard escalate

        The chain: inject × N → block → (override) → escalate
        """
        if self.escalate_after <= 0 or not category:
            return "inject"
        count = self._inject_counts.get(category, 0)
        if count >= self.escalate_after * 2:
            # Block was overridden but behavior didn't change
            return "escalate"
        elif count >= self.escalate_after:
            # Inject was ignored repeatedly → upgrade to block
            return "block"
        return "inject"

    def _tick_idle(self):
        """Called each iteration when guard did NOT fire. Manages decay."""
        self._iterations_since_trigger += 1
        if self.decay_after_idle > 0 and self._iterations_since_trigger >= self.decay_after_idle:
            self.reset_state()


# Semantic categories for inject deduplication
# Injects with the same category in the same turn are merged, not duplicated.
_INJECT_CATEGORY_PATTERNS = {
    "read_stall": re.compile(r"read.only|re.reading|gathering information|not acting", re.IGNORECASE),
    "loop": re.compile(r"loop|repeated|same tool|same call", re.IGNORECASE),
    "plan_needed": re.compile(r"plan|plan_create|organize", re.IGNORECASE),
}

# Category suppression: if a higher-priority category fires, suppress lower-priority ones.
# Key = category that suppresses, Value = set of categories it suppresses.
_CATEGORY_SUPPRESSION: dict[str, set[str]] = {
    "comprehension": {"memory_write_reminder", "memory_read_reminder"},
    "read_stall": {"memory_write_reminder"},
    "loop": {"read_stall", "memory_write_reminder"},
}


def _infer_category(verdict: GuardVerdict) -> str:
    """Infer the semantic category of an inject verdict for deduplication."""
    if verdict.category:
        return verdict.category
    # Try to infer from message content
    text = verdict.message + " " + verdict.reason
    for cat, pattern in _INJECT_CATEGORY_PATTERNS.items():
        if pattern.search(text):
            return cat
    return ""


_OVERRIDE_HINT = (
    '\n\n⚠️ OVERRIDE REQUIRED: Add "_override_reason" to your next tool call to proceed.\n'
    "DO: re-call the same tool with an extra parameter:\n"
    '  {"command": "...", "_override_reason": "reason why this is safe/justified"}\n'
    "DON'T: explain in text. Only _override_reason in tool_args works."
)


def _maybe_add_override_hint(
    verdict: GuardVerdict, blocking_guard: Guard | None, ctx: GuardContext
) -> str:
    """Append override instructions to a block message if the blocking guard is overridable.

    Only appends when:
    - The verdict is a "block"
    - The blocking guard has overridable=True
    - The LLM hasn't already provided an override_reason (avoids re-hinting on rejection)
    """
    if verdict.action != "block":
        return verdict.message
    if ctx.override_reason:
        # Override was attempted but rejected — don't re-hint
        return verdict.message
    if blocking_guard and blocking_guard.overridable:
        return verdict.message + _OVERRIDE_HINT
    return verdict.message


class GuardRegistry:
    """Manages all guards, runs them in priority order, deduplicates injects."""

    def __init__(self):
        self._guards: list[Guard] = []
    def register(self, guard: Guard):
        self._guards.append(guard)
        self._guards.sort(key=lambda g: g.priority)

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' pre-checks with inject deduplication."""
        inject_messages = []
        inject_categories_seen: set = set()
        blocking_verdicts: list = []  # v6: collect ALL blocking verdicts
        blocking_guards: list = []
        first_reason = ""
        fired_guards: set[str] = set()

        # Context pressure override: when pressure is critical and nothing can be
        # evicted, suppress all other guards' blocks to prevent deadlock
        _suppress_other_blocks = (
            ctx.context_pressure >= 0.90
            and not ctx.evictable_indexes
        )

        for guard in self._guards:
            if not guard.should_activate(ctx):
                continue

            # v3: Check satisfaction before running the guard
            # Note: do NOT call reset_state() here — that would clear the
            # satisfied condition and cause the guard to re-fire next iteration.
            # A satisfied guard simply stays silent until decay resets it.
            if hasattr(guard, 'is_satisfied') and guard.is_satisfied(ctx):
                continue

            verdict = guard.check_pre(ctx)
            if verdict is None:
                continue

            # v3: Track which guards fired
            fired_guards.add(guard.name)

            if verdict.action in ("block", "escalate", "redirect"):
                # Override mechanism: only block is overridable.
                # Escalate is the ultimate enforcement — no override path.
                if (
                    verdict.action == "block"
                    and guard.overridable
                    and ctx.override_reason
                    and guard.accept_override(ctx.override_reason, ctx)
                ):
                    # Override accepted — skip this block, log it
                    display.guard_overridden(guard.name, ctx.override_reason)
                    continue
                # Suppress non-critical blocks when context is full and un-evictable
                if _suppress_other_blocks and guard.name != "context_pressure":
                    continue
                blocking_verdicts.append(verdict)
                blocking_guards.append(guard)
                continue

            if verdict.action == "inject_msg":
                # v2: Deduplicate by semantic category
                category = _infer_category(verdict)
                if category and category in inject_categories_seen:
                    # Skip duplicate — a similar warning was already queued
                    continue
                if category:
                    inject_categories_seen.add(category)
                    # v2.1: Suppress lower-priority categories
                    suppressed = _CATEGORY_SUPPRESSION.get(category)
                    if suppressed:
                        inject_categories_seen.update(suppressed)

                # v5: Escalation uses category OR reason as tracking key
                # Category is for dedup; escalation_key is for counting fires
                escalation_key = category or verdict.reason or guard.name
                escalation_level = guard._get_escalation_level(escalation_key)

                if escalation_level == "escalate":
                    # Final level: hard escalate, LLM cannot bypass
                    # But suppress if context is critically full with nothing to evict
                    if _suppress_other_blocks and guard.name != "context_pressure":
                        continue
                    esc_msg = (
                        f"[{guard.name}] ESCALATED: {verdict.message}\n\n"
                        f"This advisory has been ignored {guard.escalate_after * 2}+ times. "
                        f"You MUST address it before continuing."
                    )
                    blocking_verdicts.append(GuardVerdict.escalate(
                        esc_msg, reason=f"escalated_{guard.name}"
                    ))
                    blocking_guards.append(guard)
                    # Record trigger (keeps counting)
                    guard._record_trigger(escalation_key)
                    fired_guards.add(guard.name)
                    continue

                elif escalation_level == "block":
                    # Mid level: block tool execution, LLM can override with reason
                    # Suppress if context is critically full with nothing to evict
                    if _suppress_other_blocks and guard.name != "context_pressure":
                        continue
                    block_msg = (
                        f"[{guard.name}] BLOCKED: {verdict.message}\n\n"
                        f"This advisory has been ignored {guard.escalate_after}+ times. "
                        f"Tool call blocked until you address the concern."
                    )
                    block_verdict = GuardVerdict.block(
                        block_msg, reason=f"escalated_block_{guard.name}"
                    )
                    # Check if LLM is overriding this escalated block
                    if guard.overridable and ctx.override_reason:
                        if guard.accept_override(ctx.override_reason, ctx):
                            display.guard_overridden(guard.name, ctx.override_reason)
                            continue
                    blocking_verdicts.append(block_verdict)
                    blocking_guards.append(guard)
                    # Record trigger (keeps counting toward escalate)
                    guard._record_trigger(escalation_key)
                    fired_guards.add(guard.name)
                    continue

                # Normal inject level
                inject_messages.append(verdict.message)
                if not first_reason:
                    first_reason = verdict.reason

                # Record trigger to track escalation progress
                guard._record_trigger(escalation_key)
                fired_guards.add(guard.name)

        # v6: If there are blocking verdicts, merge into one multi-block message
        if blocking_verdicts:
            self.tick_guard_lifecycle(fired_guards)
            if len(blocking_verdicts) == 1:
                # Single block — add override hint
                v = blocking_verdicts[0]
                v.message = _maybe_add_override_hint(v, blocking_guards[0], ctx)
                return v
            else:
                # Multiple blocks — format as list
                lines = [f"[{len(blocking_verdicts)} Guard(s) blocking]"]
                for i, (v, g) in enumerate(zip(blocking_verdicts, blocking_guards), 1):
                    override_hint = "(可 _override_reason 绕过)" if g.overridable else "(不可绕过)"
                    lines.append(f"  {i}. [{g.name}] {override_hint} {v.message}")
                lines.append("解决对应条件后重试。")
                # Use the strongest action among all verdicts
                has_escalate = any(v.action == "escalate" for v in blocking_verdicts)
                if has_escalate:
                    return GuardVerdict.escalate(
                        "\n".join(lines), reason="multi_guard_block"
                    )
                return GuardVerdict.block(
                    "\n".join(lines), reason="multi_guard_block"
                )

        # Merge all inject messages into one verdict (deduplicated)
        if inject_messages:
            # v3: Tick lifecycle for idle guards
            self.tick_guard_lifecycle(fired_guards)
            combined = "\n\n".join(inject_messages)
            return GuardVerdict.inject(
                combined,
                reason=first_reason or "multi_guard_inject"
            )

        # v3: Tick lifecycle for idle guards (no verdict case)
        self.tick_guard_lifecycle(fired_guards)
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' post-checks with inject deduplication."""
        inject_messages = []
        inject_categories_seen: set = set()
        first_hard_verdict = None
        first_hard_guard = None
        first_reason = ""

        # v4: Universal effectiveness check — ask each guard if its inject worked
        for guard in self._guards:
            result = guard.was_inject_effective(ctx)
            if result is not None:
                # v5: If effective, reset escalation counter — agent responded
                if result is True:
                    guard._inject_counts.clear()

        for guard in self._guards:
            if not guard.should_activate(ctx):
                continue
            verdict = guard.check_post(ctx)
            if verdict is None:
                continue

            if verdict.action in ("block", "escalate", "redirect"):
                # Override mechanism (same as check_pre) — only block is overridable
                if (
                    verdict.action == "block"
                    and guard.overridable
                    and ctx.override_reason
                    and guard.accept_override(ctx.override_reason, ctx)
                ):
                    display.guard_overridden(guard.name, ctx.override_reason)
                    continue
                # Suppress non-critical blocks when context is full and un-evictable
                _suppress = (
                    ctx.context_pressure >= 0.90
                    and not ctx.evictable_indexes
                    and guard.name != "context_pressure"
                )
                if _suppress:
                    continue
                if first_hard_verdict is None:
                    first_hard_verdict = verdict
                    first_hard_guard = guard
                continue

            if verdict.action == "inject_msg":
                # v2: Deduplicate by semantic category
                category = _infer_category(verdict)
                if category and category in inject_categories_seen:
                    continue
                if category:
                    inject_categories_seen.add(category)
                    # v2.1: Suppress lower-priority categories
                    suppressed = _CATEGORY_SUPPRESSION.get(category)
                    if suppressed:
                        inject_categories_seen.update(suppressed)

                # v5: Escalation uses category OR reason as tracking key
                escalation_key = category or verdict.reason or guard.name
                escalation_level = guard._get_escalation_level(escalation_key)
                if escalation_level == "escalate":
                    esc_msg = (
                        f"[{guard.name}] ESCALATED: {verdict.message}\n\n"
                        f"This advisory has been ignored repeatedly. "
                        f"You MUST address it before continuing."
                    )
                    if first_hard_verdict is None:
                        first_hard_verdict = GuardVerdict.escalate(
                            esc_msg, reason=f"escalated_{guard.name}"
                        )
                    guard._record_trigger(escalation_key)
                    continue
                elif escalation_level == "block":
                    block_msg = (
                        f"[{guard.name}] BLOCKED (post): {verdict.message}\n\n"
                        f"This advisory has been ignored repeatedly."
                    )
                    if first_hard_verdict is None:
                        first_hard_verdict = GuardVerdict.block(
                            block_msg, reason=f"escalated_block_{guard.name}"
                        )
                    guard._record_trigger(escalation_key)
                    continue

                inject_messages.append(verdict.message)
                if not first_reason:
                    first_reason = verdict.reason

                guard._record_trigger(escalation_key)

        if first_hard_verdict:
            # Don't prepend unrelated inject messages — block message is sufficient
            first_hard_verdict.message = _maybe_add_override_hint(
                first_hard_verdict, first_hard_guard, ctx
            )
            return first_hard_verdict

        if inject_messages:
            return GuardVerdict.inject(
                "\n\n".join(inject_messages),
                reason=first_reason or "multi_guard_inject"
            )

        return None

    def check_strategic(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' strategic checks."""
        for guard in self._guards:
            if guard.should_activate(ctx):
                verdict = guard.check_strategic(ctx)
                if verdict is not None:
                    return verdict
        return None

    def reset_iteration(self):
        """Reset per-iteration state for all guards.

        Called at the start of each iteration (LLM+tool loop) within a turn.
        v3: Satisfaction and decay are handled in tick_guard_lifecycle(), not here.
        """
        for guard in self._guards:
            # Call reset_turn — subclasses override this name
            guard.reset_turn()

    def reset_new_turn(self):
        """Reset per-turn state for all guards.

        Called once at the start of a new turn (new user message). This resets
        cross-iteration counters so state doesn't leak between user messages.
        """
        for guard in self._guards:
            guard.reset_new_turn()
        # Also reset shared state per-turn tracking

    # Backward compat alias
    reset_turn = reset_iteration

    def tick_guard_lifecycle(self, fired_guards: set[str]):
        """v3: Called after each check cycle to manage decay for idle guards.

        Args:
            fired_guards: set of guard names that returned non-None verdicts this cycle
        """
        for guard in self._guards:
            if guard.name in fired_guards:
                guard._record_trigger()
            else:
                guard._tick_idle()

    @property
    def guards(self) -> list[Guard]:
        return list(self._guards)

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

    # LLM classify function
    classify_fn: Any = None

    # Override reason from LLM
    override_reason: str = ""


@dataclass
class GuardVerdict:
    """What the guard wants the agent to do."""

    action: Literal["allow", "block", "inject_msg", "escalate"]
    message: str = ""
    reason: str = ""
    category: str = ""  # For inject deduplication

    @classmethod
    def block(cls, message: str, reason: str = "", category: str = "") -> GuardVerdict:
        return cls(action="block", message=message, reason=reason, category=category)

    @classmethod
    def inject(cls, message: str, reason: str = "", category: str = "") -> GuardVerdict:
        return cls(action="inject_msg", message=message, reason=reason, category=category)

    @classmethod
    def escalate(cls, message: str, reason: str = "", category: str = "") -> GuardVerdict:
        return cls(action="escalate", message=message, reason=reason, category=category)


_OVERRIDE_HINT = (
    '\n\n⚠️ OVERRIDE REQUIRED: Add "_override_reason" to your next tool call to proceed.\n'
    "DO: re-call the same tool with an extra parameter:\n"
    '  {"command": "...", "_override_reason": "reason why this is safe/justified"}\n'
    "DON'T: explain in text. Only _override_reason in tool_args works."
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

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Validate LLM's override reason. Only called for block verdicts.

        Default: accept any reason longer than 5 chars.
        Override for stricter validation.
        """
        return bool(reason and len(reason.strip()) > 5)

    def reset_turn(self):
        """Called at the start of each new user message. Clear per-turn state."""
        pass


class GuardRegistry:
    """Manages all guards, runs them in priority order, deduplicates injects."""

    def __init__(self):
        self._guards: list[Guard] = []

    def register(self, guard: Guard):
        self._guards.append(guard)
        self._guards.sort(key=lambda g: g.priority)

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' pre-checks. First block/escalate wins; injects merge."""
        inject_messages: list[str] = []
        inject_categories_seen: set[str] = set()
        first_reason = ""

        for guard in self._guards:
            verdict = guard.check_pre(ctx)
            if verdict is None:
                continue

            if verdict.action in ("block", "escalate"):
                # Override mechanism: only block is overridable
                if (
                    verdict.action == "block"
                    and ctx.override_reason
                    and guard.accept_override(ctx.override_reason, ctx)
                ):
                    display.guard_overridden(guard.name, ctx.override_reason)
                    continue
                # Add override hint for blocks
                if verdict.action == "block" and not ctx.override_reason:
                    verdict.message += _OVERRIDE_HINT
                return verdict

            if verdict.action == "inject_msg":
                # Deduplicate by category
                cat = verdict.category
                if cat and cat in inject_categories_seen:
                    continue
                if cat:
                    inject_categories_seen.add(cat)
                inject_messages.append(verdict.message)
                if not first_reason:
                    first_reason = verdict.reason

        if inject_messages:
            return GuardVerdict.inject(
                "\n\n".join(inject_messages),
                reason=first_reason or "multi_guard_inject"
            )
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Run all guards' post-checks. First block/escalate wins; injects merge."""
        inject_messages: list[str] = []
        inject_categories_seen: set[str] = set()
        first_reason = ""

        for guard in self._guards:
            verdict = guard.check_post(ctx)
            if verdict is None:
                continue

            if verdict.action in ("block", "escalate"):
                if (
                    verdict.action == "block"
                    and ctx.override_reason
                    and guard.accept_override(ctx.override_reason, ctx)
                ):
                    display.guard_overridden(guard.name, ctx.override_reason)
                    continue
                if verdict.action == "block" and not ctx.override_reason:
                    verdict.message += _OVERRIDE_HINT
                return verdict

            if verdict.action == "inject_msg":
                cat = verdict.category
                if cat and cat in inject_categories_seen:
                    continue
                if cat:
                    inject_categories_seen.add(cat)
                inject_messages.append(verdict.message)
                if not first_reason:
                    first_reason = verdict.reason

        if inject_messages:
            return GuardVerdict.inject(
                "\n\n".join(inject_messages),
                reason=first_reason or "multi_guard_inject"
            )
        return None

    def reset_turn(self):
        """Reset per-turn state for all guards."""
        for guard in self._guards:
            guard.reset_turn()

    @property
    def guards(self) -> list[Guard]:
        return list(self._guards)

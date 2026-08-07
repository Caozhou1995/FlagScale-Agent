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

"""ConstraintGuard — enforces compiled Constraints via queue-based blocking.

Design:
1. First trigger: all matching constraints are judged at once, violations collected into a queue.
2. Agent sees one violation at a time — fix it, retry the tool call.
3. On retry: re-judge the current (first) queued constraint.
   - Pass → pop it, expose next one.
   - Fail → keep blocking on the same one.
4. Queue empty → tool call is allowed through.
5. Degrade: if same constraint fails judge 5 times in a row, pop it (judge is broken).
6. inject_only constraints: never block, only inject advisory.
"""

from __future__ import annotations

from flagscale_agent.react.constraint import Constraint
from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class ConstraintGuard(Guard):
    """Enforce constraints with queue-based sequential resolution."""

    name = "constraint"
    priority = 15  # High priority — runs before most guards

    def __init__(self, constraints: list[Constraint] | None = None):
        self._constraints: list[Constraint] = constraints or []
        # Queue of violated constraint IDs (ordered), set on first trigger
        self._violation_queue: list[str] = []
        # Per-constraint consecutive fail count (for degrade)
        self._consecutive_fails: dict[str, int] = {}
        # Constraints that have been degraded (judge unreliable)
        self._degraded: set[str] = set()
        # Track whether we're in "queue mode" (processing a violation list)
        self._queue_active: bool = False

        self._DEGRADE_THRESHOLD = 5  # Pop after 5 consecutive fails on same constraint

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not self._constraints:
            return None
        # Only check during tool execution
        if not ctx.tool_name:
            return None

        # Collect all triggered constraints for this tool call
        triggered = [
            c for c in self._constraints
            if c.trigger.matches(ctx.tool_name, ctx.tool_args, ctx.tool_result)
        ]
        if not triggered:
            # No constraints match — clear queue (different tool call pattern)
            self._clear_queue()
            return None

        # Separate inject_only (advisory) from blocking constraints
        advisory = [c for c in triggered if c.inject_only]
        blocking = [c for c in triggered if not c.inject_only]

        # Handle advisory constraints: inject all at once, never block
        advisory_msg = self._process_advisory(advisory, ctx)

        # If no blocking constraints, just return advisory (or None)
        if not blocking:
            self._clear_queue()
            return advisory_msg

        # === Queue-based blocking logic ===

        if not self._queue_active:
            # First time: judge ALL blocking constraints, build violation queue
            violations = self._judge_all(blocking, ctx)
            if not violations:
                # All pass — allow through (+ advisory if any)
                self._clear_queue()
                return advisory_msg
            # Build queue from violation results
            self._violation_queue = [v_id for v_id, _ in violations]
            self._queue_active = True
            # Show first violation
            first_id, first_reason = violations[0]
            constraint = self._get_constraint(first_id)
            total = len(violations)
            return self._make_block(constraint, first_reason, position=1, total=total)
        else:
            # Queue mode: re-judge only the current (first) constraint
            if not self._violation_queue:
                # Queue exhausted — allow through
                self._clear_queue()
                return advisory_msg

            current_id = self._violation_queue[0]
            constraint = self._get_constraint(current_id)

            if constraint is None:
                # Constraint removed? Skip it.
                self._violation_queue.pop(0)
                return self._advance_or_pass(ctx, advisory_msg)

            # Re-judge current constraint
            violated, reason = self._judge_violation(ctx, constraint)

            if not violated:
                # Agent fixed it — pop and move to next
                self._violation_queue.pop(0)
                self._consecutive_fails.pop(current_id, None)
                return self._advance_or_pass(ctx, advisory_msg)
            else:
                # Still violated — check degrade
                fails = self._consecutive_fails.get(current_id, 0) + 1
                self._consecutive_fails[current_id] = fails

                if fails >= self._DEGRADE_THRESHOLD:
                    # Judge is unreliable for this constraint — degrade and skip
                    self._degraded.add(current_id)
                    self._violation_queue.pop(0)
                    return self._advance_or_pass(ctx, advisory_msg)

                # Block again on same constraint
                pos = 1
                total = len(self._violation_queue)
                return self._make_block(constraint, reason, position=pos, total=total)

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    # ── Internal helpers ──

    def _judge_all(
        self, constraints: list[Constraint], ctx: GuardContext
    ) -> list[tuple[str, str]]:
        """Judge all constraints, return list of (id, reason) for violated ones."""
        violations = []
        for c in constraints:
            if c.id in self._degraded:
                continue  # Skip permanently degraded
            violated, reason = self._judge_violation(ctx, c)
            if violated:
                violations.append((c.id, reason))
        return violations

    def _advance_or_pass(
        self, ctx: GuardContext, advisory_msg: GuardVerdict | None
    ) -> GuardVerdict | None:
        """After popping a constraint, either block on next or pass."""
        if not self._violation_queue:
            self._clear_queue()
            return advisory_msg

        # Block on next constraint in queue
        next_id = self._violation_queue[0]
        constraint = self._get_constraint(next_id)
        if constraint is None:
            self._violation_queue.pop(0)
            return self._advance_or_pass(ctx, advisory_msg)

        pos = 1
        total = len(self._violation_queue)
        return self._make_block(
            constraint, f"Queued violation (not yet re-judged)", position=pos, total=total
        )

    def _make_block(
        self, constraint: Constraint, reason: str, position: int, total: int
    ) -> GuardVerdict:
        """Create a block verdict for a constraint violation."""
        progress = f"[{position}/{total} remaining]" if total > 1 else ""
        return GuardVerdict.block(
            f"[CONSTRAINT BLOCK] {progress} Constraint [{constraint.id}] violated.\n"
            f"Reason: {reason}\n"
            f"Required action: {constraint.correction}\n"
            f"[To override: add \"_override_reason\" field to tool_args with justification.]",
            reason=f"Constraint [{constraint.id}]: {reason}",
        )

    def _process_advisory(
        self, constraints: list[Constraint], ctx: GuardContext
    ) -> GuardVerdict | None:
        """Process inject_only constraints — judge and return combined inject or None."""
        if not constraints or not ctx.classify_fn:
            return None
        violations = []
        for c in constraints:
            violated, reason = self._judge_violation(ctx, c)
            if violated:
                violations.append(f"[{c.id}] {reason}. Suggestion: {c.correction}")
        if not violations:
            return None
        return GuardVerdict.inject(
            "[CONSTRAINT ADVISORY]\n" + "\n".join(violations),
            reason="Advisory constraints triggered",
            category="constraint_advisory",
        )

    def _judge_violation(
        self, ctx: GuardContext, constraint: Constraint
    ) -> tuple[bool, str]:
        """Call LLM judge to determine if constraint is violated.

        Returns (violated: bool, reason: str).
        """
        if not ctx.classify_fn:
            return (True, "no judge available")

        judge_context = {
            "constraint": constraint.description,
            "prompt": constraint.prompt,
            "tool_name": ctx.tool_name,
            "tool_args": str(ctx.tool_args),
        }
        # Include recent tool history if available for context
        if hasattr(ctx, 'recent_tool_history') and ctx.recent_tool_history:
            judge_context["recent_history"] = str(ctx.recent_tool_history[-5:])

        try:
            result = ctx.classify_fn("is_constraint_violated", judge_context)
            if isinstance(result, dict):
                # Support both formats:
                # {"violated": bool, "reason": str}  (direct)
                # {"real": bool, "reason": str}      (judge format)
                violated = result.get("violated", result.get("real", False))
                reason = result.get("reason", "")
                return (bool(violated), reason)
            # Fallback for legacy bool return
            return (bool(result), "")
        except Exception as e:
            return (True, f"judge error: {e}")

    def _get_constraint(self, constraint_id: str) -> Constraint | None:
        """Look up a constraint by ID."""
        for c in self._constraints:
            if c.id == constraint_id:
                return c
        return None

    def _clear_queue(self):
        """Reset queue state."""
        self._violation_queue.clear()
        self._queue_active = False

    # ── Public API ──

    @property
    def violations(self) -> dict[str, int]:
        """Current consecutive fail counts per constraint."""
        return dict(self._consecutive_fails)

    @property
    def constraints(self) -> list[Constraint]:
        """Currently loaded constraints."""
        return list(self._constraints)

    def add_constraints(self, constraints: list[Constraint]):
        """Add constraints to the existing list."""
        self._constraints.extend(constraints)

    @property
    def queue_length(self) -> int:
        """Number of violations remaining in queue."""
        return len(self._violation_queue)

    def reset_turn(self):
        """Violations persist across iterations (working on same tool call)."""
        pass

    def reset_turn(self):
        """New user message — clear queue (new context, constraints re-evaluated fresh)."""
        self._clear_queue()
        self._consecutive_fails.clear()

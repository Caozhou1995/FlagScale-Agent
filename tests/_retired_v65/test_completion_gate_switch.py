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

"""The completion gate is a SWITCH, not deleted code.

Context: the [TASK_COMPLETE] gate wedged real runs (see
``pitfall/agent/completion_gate_loop``), so it is disabled by default. Disabling
must not mean deleting: the obligation logic stays live and unit-tested, so
re-enabling is a one-line flip and a regression in the gate logic still fails.

These tests pin the three properties that make that safe:

1. The flag EXISTS on the class and defaults to ``False`` (gate off).
2. With the flag off, ``_completion_obligations`` returns ``[]`` -- i.e. the
   gate really is bypassed on the hot path, not merely documented as such.
3. With the flag forced on (monkeypatched), the SAME call still produces the
   original blocks -- so the gate bodies were preserved, not stubbed out.

Property 3 is the load-bearing one: without it, a future edit could gut the
gate bodies and every currently-green test would still pass.
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.guard.verification import VerificationGuard


class _Plan:
    """Minimal plan stub matching the real PlanManager surface.

    The guard reads ``_plan.get_active()`` and inspects ``steps[*].status`` --
    NOT a convenience ``unfinished_steps()`` method. The stub must mirror that
    exactly, otherwise the test measures a shape the production code never sees.
    """

    def __init__(self, steps=None):
        self._steps = steps if steps is not None else [
            {"id": 1, "title": "do the thing", "status": "doing"},
        ]

    def get_active(self):
        return {"steps": self._steps}


class _Ctx:
    """Minimal GuardContext stub for the completion route."""

    def __init__(self, tool_args=None, override_reason=None):
        self.tool_args = tool_args or {}
        self.override_reason = override_reason
        self.messages = []
        self.task_text = "TASK: build a thing\nCONSTRAINT: must be 3 items"


def test_flag_exists_and_defaults_off():
    """The switch must exist and be OFF by default."""
    assert hasattr(VerificationGuard, "COMPLETION_GATE_ENABLED"), (
        "COMPLETION_GATE_ENABLED must exist on the class -- the gate is a "
        "switch, so re-enabling it is one line, not a code resurrection"
    )
    assert VerificationGuard.COMPLETION_GATE_ENABLED is False, (
        "the completion gate must default OFF (user instruction 2026-10-22); "
        "if this flips, real runs wedge on [TASK_COMPLETE] again"
    )


def test_gate_off_returns_no_obligations():
    """Flag off => the hot path bypasses the gate (returns [])."""
    guard = VerificationGuard(plan=_Plan())
    assert VerificationGuard.COMPLETION_GATE_ENABLED is False
    assert guard._completion_obligations(_Ctx()) == []


def test_gate_on_still_blocks_unfinished_steps(monkeypatch):
    """Flag on => the preserved gate body still fires.

    This is the mutation-catching test: gutting the gate bodies while leaving
    the flag wired would keep every other test green but fail here.
    """
    monkeypatch.setattr(VerificationGuard, "COMPLETION_GATE_ENABLED", True)
    guard = VerificationGuard(plan=_Plan())
    verdicts = guard._completion_obligations(_Ctx())
    assert verdicts, (
        "with COMPLETION_GATE_ENABLED=True the unfinished-step obligation must "
        "still block; an empty list means the gate body was gutted"
    )
    reasons = [v.reason for v in verdicts]
    assert "completion_with_unfinished_steps" in reasons, (
        f"expected the unfinished-steps obligation, got {reasons}"
    )


def test_gate_on_allows_when_plan_is_finished(monkeypatch):
    """Flag on but plan complete => no unfinished-step block (negative control).

    Without this, a gate that unconditionally blocks would pass the test above.
    """
    monkeypatch.setattr(VerificationGuard, "COMPLETION_GATE_ENABLED", True)

    guard = VerificationGuard(
        plan=_Plan([{"id": 1, "title": "do the thing", "status": "done"}])
    )
    reasons = [v.reason for v in guard._completion_obligations(_Ctx())]
    assert "completion_with_unfinished_steps" not in reasons, (
        f"a finished plan must not produce the unfinished-steps block; got {reasons}"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))

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

"""WriteParamLossGuard — post-inject when write_file loses its content arg.

Anti-pattern (observed 3+ times in one session): in a parallel tool-call
batch, the upstream truncates/clobbers the long `content` parameter of one
write_file call; the registry's pre-validation then returns only
"required argument(s) missing for tool 'write_file': content" — a message
that names the symptom but not the CAUSE. The agent cannot tell an argument
it forgot from an argument the pipeline dropped, and burns a turn guessing
(or resends into the same parallel-batch shape and loses content again).

Design:
- post-only + inject-only: fires AFTER the executed write_file call returned
  the missing-content error; never blocks (the error itself already blocked
  the write — this adds the diagnosis the error text lacks).
- injects AT MOST ONCE per assistant turn: the counter rides the guard
  instance and resets on any non-matching tool result or a fresh user turn.
- the match is anchored to the registry's exact error prefix for write_file
  (see ToolRegistry._missing_required in react/tools/__init__.py) so real
  content-bearing errors (shrink guard, protected path) never trip it.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# Exact prefix the registry emits for write_file with content absent/empty
# (react/tools/__init__.py L54). Kept specific: matching a bare substring
# like "missing" would collide with unrelated errors.
_MATCH = "required argument(s) missing for tool 'write_file'"


class WriteParamLossGuard(Guard):
    """Post-tool guard: explain WHY write_file saw an empty content arg."""

    name = "write_param_loss"
    priority = 70  # Advisory tier — never blocks

    def __init__(self):
        self._fired_this_turn = False

    def reset_turn(self):
        self._fired_this_turn = False

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None  # post-only

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if (ctx.tool_name or "") != "write_file":
            return None
        result = (ctx.tool_result or "")
        if _MATCH not in result:
            return None
        if self._fired_this_turn:
            return None
        self._fired_this_turn = True
        return GuardVerdict.inject(
            "[WriteParamLoss] write_file arrived WITHOUT its content argument. "
            "In a parallel tool-call batch the long content parameter is the "
            "most likely casualty of output truncation — the pipeline dropped "
            "it, you did not forget it. Fix: re-issue THIS call alone (not in a "
            "parallel batch), with content split into sections of <=3000 chars "
            "(first section mode='write', continuations mode='append').",
            reason="write_file_content_dropped_in_batch",
            category="write_param_loss",
        )

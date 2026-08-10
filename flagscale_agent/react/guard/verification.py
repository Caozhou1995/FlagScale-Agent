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

"""VerificationGuard — requires verification evidence when marking steps complete.

Design principles:
- Block plan_update(action="step_done") if no _override_reason provided
- Other tool calls (read_file/shell/grep) are completely unaffected
- LLM can freely perform verification operations after being blocked
- Once verified, LLM calls plan_update(step_done, _override_reason="...") to pass
- Does not check override_reason content — any non-empty reason passes
- Uses existing override mechanism — LLM already knows this pattern

Why this works:
- To pass the block, LLM must write override_reason
- Writing override_reason forces LLM to reflect "how do I know this step is done?"
- The reflection itself improves verification discipline

Execution flow:
1. LLM: plan_update(action="step_done", step_id=3)
2. Guard: BLOCK - verification evidence required

3. LLM: OK, let me verify first
4. LLM: shell("grep '<<<<<<' -r .")  ← executes normally, not blocked
5. LLM: read_file("/path/to/__init__.py")  ← executes normally
6. LLM: shell("python -m py_compile *.py")  ← executes normally

7. LLM: Verified, now I can step_done
8. LLM: plan_update(action="step_done", step_id=3, _override_reason="grep shows no conflicts, files complete, parseable")
9. Guard: Has override_reason, allow ✓
"""

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


_VERIFICATION_REQUIRED = """[VerificationGuard] Verification evidence required before step_done.

You called plan_update(action="step_done") without verification evidence.

Complex operations can fail silently. Before marking a step complete, verify the goal was achieved:
- Code changes: check syntax, imports, key functions exist
- Git operations: grep conflict markers, verify file count, check key files parseable
- Config changes: read config back, verify key fields present and valid
- Installations: test import, run basic functionality

To proceed:
1. Run verification commands (grep/read_file/shell/python -c) — these are NOT blocked
2. Once verified, call plan_update(step_done, _override_reason="verification details")

Example: _override_reason="grep shows no conflicts, __init__.py parseable, import test passed"
"""

_POST_RECOVERY_REMINDER = """[VerificationGuard] Context was just recovered via hard_reset.

Before continuing work:
1. Read key files to confirm current state
2. Check recent changes (git status, grep for markers, file checksums)
3. Verify assumptions from pre-recovery context still hold

The goal: avoid propagating stale assumptions into new work."""


class VerificationGuard(Guard):
    """Requires verification evidence when marking steps complete.
    
    Key design:
    - Only blocks plan_update(action="step_done"), other tool calls unaffected
    - LLM can freely execute verification operations after being blocked
    - Once verified, LLM calls with _override_reason to pass
    - Does not check override_reason content
    
    Also injects a reminder after hard_reset recovery.
    """
    
    name = "verification"
    priority = 55
    
    def __init__(self):
        self._post_recovery = False
        self._recovery_reminded = False
    
    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Timing 1: step_done requires verification evidence
        if ctx.tool_name == "plan_update":
            action = ctx.tool_args.get("action")
            
            # Only check on step_done, other actions (step_doing/add_steps) pass through
            if action == "step_done":
                override_reason = ctx.tool_args.get("_override_reason", "").strip()
                
                if not override_reason:
                    return GuardVerdict.block(
                        message=_VERIFICATION_REQUIRED,
                        reason="step_done_no_verification",
                        category="verification_required"
                    )
                # Has override_reason, allow — don't check content
                return None
        
        # Timing 2: post-recovery, inject reminder on first step_doing
        if self._post_recovery and not self._recovery_reminded:
            if ctx.tool_name == "plan_update":
                action = ctx.tool_args.get("action")
                if action == "step_doing":
                    self._recovery_reminded = True
                    return GuardVerdict.inject(
                        message=_POST_RECOVERY_REMINDER,
                        reason="post_recovery_reminder",
                        category="post_recovery"
                    )
        
        # All other tools (read_file/shell/edit_file) completely unaffected
        return None
    
    def notify_recovery(self):
        """Called by hard_reset logic to signal recovery."""
        self._post_recovery = True
        self._recovery_reminded = False

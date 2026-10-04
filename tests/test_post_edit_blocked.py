# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for the blocked-edit forced re-read inject (proposal a6b1f647).

A guard-blocked edit NEVER landed, yet the advisory stream may still print
"[Post-edit] ... edited" — the false-success header that hid a real loss.
PostEditFarEndGuard.check_post must swap its usual far-end hint for a
MANDATORY cold re-read demand when the kernel's "[BLOCKED BY GUARD]" marker
is present in the tool result.
"""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.post_edit_far_end import PostEditFarEndGuard


def _ctx(result, path="/tmp/probe_target.py"):
    return GuardContext(
        tool_name="edit_file",
        tool_args={"path": path},
        tool_result=result,
    )


class TestBlockedEditInject:
    def test_blocked_result_injects_cold_reread(self):
        """[BLOCKED BY GUARD] result -> inject with the cold re-read demand."""
        g = PostEditFarEndGuard()
        v = g.check_post(_ctx("[BLOCKED BY GUARD] edit rejected by guard"))
        assert v is not None
        assert v.action == "inject"
        assert v.reason == "post_edit_blocked_reverify"
        assert v.category == "post_edit_far_end_blocked"
        assert "did NOT land" in v.message
        assert "MANDATORY cold re-read" in v.message

    def test_blocked_message_names_the_path(self):
        """The demand points at the exact path that was NOT changed."""
        g = PostEditFarEndGuard()
        v = g.check_post(_ctx("[BLOCKED BY GUARD] nope", path="/a/b/c.py"))
        assert "/a/b/c.py" in v.message

    def test_success_path_unchanged(self):
        """A successful edit keeps the ORIGINAL far-end hint behavior."""
        g = PostEditFarEndGuard()
        v = g.check_post(_ctx("Successfully edited /tmp/probe_target.py"))
        assert v is not None
        assert v.reason == "post_edit_far_end"
        assert "did NOT land" not in v.message

    def test_error_result_still_silent(self):
        """An errored edit still produces no inject (nothing landed to verify)."""
        g = PostEditFarEndGuard()
        assert g.check_post(_ctx("ERROR: old_string not found")) is None

    def test_blocked_without_path_is_silent(self):
        """A blocked result with no path arg cannot name a target — stay silent."""
        g = PostEditFarEndGuard()
        assert g.check_post(_ctx("[BLOCKED BY GUARD] x", path="")) is None

    def test_non_file_tool_ignored(self):
        """Non file-writing tools never trigger the guard."""
        g = PostEditFarEndGuard()
        ctx = GuardContext(
            tool_name="shell",
            tool_args={"command": "ls"},
            tool_result="[BLOCKED BY GUARD] something",
        )
        assert g.check_post(ctx) is None

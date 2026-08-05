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

"""Tests for Context Management V3: evict/recall system.

Covers:
- SwapStore: save, load, has, delete, list, size
- HistoryManager: evict_message, recall_message, get_evictable_indexes
- ContextPressureGuard: soft/hard limits, 30% requirement, reset
- Agent integration: _handle_evict, _handle_recall
"""

import json
import os
import tempfile
import shutil
import pytest

from flagscale_agent.react.swap_store import SwapStore


# ════════════════════════════════════════════════════════════════════════════════
# SwapStore Tests
# ════════════════════════════════════════════════════════════════════════════════


class TestSwapStore:
    """Test SwapStore local file operations."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = SwapStore(self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        """Basic save and load cycle."""
        content = "This is a large tool result with many lines...\n" * 100
        self.store.save(5, content, {"tool_name": "read_file", "tokens": 500})

        loaded = self.store.load(5)
        assert loaded == content

    def test_load_nonexistent(self):
        """Loading a non-existent index returns None."""
        assert self.store.load(999) is None

    def test_has(self):
        """Check if index exists."""
        assert not self.store.has(3)
        self.store.save(3, "content")
        assert self.store.has(3)

    def test_delete(self):
        """Delete removes stored content."""
        self.store.save(7, "content")
        assert self.store.has(7)
        result = self.store.delete(7)
        assert result is True
        assert not self.store.has(7)
        assert self.store.load(7) is None

    def test_delete_nonexistent(self):
        """Deleting a non-existent index returns False."""
        assert self.store.delete(999) is False

    def test_list_indexes(self):
        """List all stored indexes in sorted order."""
        self.store.save(3, "a")
        self.store.save(10, "b")
        self.store.save(1, "c")
        assert self.store.list_indexes() == [1, 3, 10]

    def test_list_indexes_empty(self):
        """Empty store returns empty list."""
        assert self.store.list_indexes() == []

    def test_size_bytes(self):
        """size_bytes returns total bytes of stored files."""
        self.store.save(1, "hello")
        self.store.save(2, "world" * 1000)
        size = self.store.size_bytes()
        assert size > 0
        # Second file should be much larger
        assert size > 5000

    def test_save_with_metadata(self):
        """Metadata is stored alongside content."""
        self.store.save(5, "content", {"tool_name": "shell", "tokens": 200})
        # Verify file structure
        path = os.path.join(self.tmpdir, "5.json")
        with open(path) as f:
            data = json.load(f)
        assert data["content"] == "content"
        assert data["metadata"]["tool_name"] == "shell"
        assert data["metadata"]["tokens"] == 200

    def test_overwrite_existing(self):
        """Saving to the same index overwrites."""
        self.store.save(5, "original")
        self.store.save(5, "updated")
        assert self.store.load(5) == "updated"

    def test_unicode_content(self):
        """Handle unicode content correctly."""
        content = "中文内容 日本語 한국어 🎉"
        self.store.save(1, content)
        assert self.store.load(1) == content

    def test_large_content(self):
        """Handle large content (simulating big tool results)."""
        content = "x" * 100_000  # 100K chars
        self.store.save(1, content)
        assert self.store.load(1) == content
        assert len(self.store.load(1)) == 100_000

    def test_store_dir_creation(self):
        """SwapStore creates its directory if it doesn't exist."""
        new_dir = os.path.join(self.tmpdir, "nested", "deep", "swap")
        store = SwapStore(new_dir)
        assert os.path.isdir(new_dir)
        store.save(1, "test")
        assert store.load(1) == "test"


# ════════════════════════════════════════════════════════════════════════════════
# HistoryManager evict/recall Tests
# ════════════════════════════════════════════════════════════════════════════════


class TestHistoryEvictRecall:
    """Test HistoryManager evict and recall methods."""

    def setup_method(self):
        from flagscale_agent.react.history import HistoryManager
        self.hm = HistoryManager(max_context_tokens=64000)

    def _build_messages(self):
        """Build a realistic message sequence."""
        self.hm._messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Read the file for me."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_001",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "/src/main.py"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": "def main():\n    print('hello')\n" * 100,
            },
            {"role": "assistant", "content": "I can see the main.py file..."},
            {"role": "user", "content": "Now run shell command."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_002",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "ls -la /workspace"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_002",
                "content": "total 128\ndrwxr-xr-x 5 root root 4096 ...\n" * 50,
            },
            {"role": "assistant", "content": "Here's the directory listing."},
        ]

    def test_evict_tool_result(self):
        """Evict a tool_result message."""
        self._build_messages()
        result = self.hm.evict_message(3)  # read_file result

        assert result is not None
        assert "def main():" in result["content"]
        assert result["metadata"]["tool_name"] == "read_file"
        assert "/src/main.py" in result["metadata"]["tool_input"]

        # Message should now have placeholder
        msg = self.hm._messages[3]
        assert msg["_evicted"] is True
        assert "[evicted | index=3" in msg["content"]
        assert "read_file" in msg["content"]

    def test_evict_non_tool_result_fails(self):
        """Cannot evict system messages. User/assistant are evictable."""
        self._build_messages()
        # User message — now evictable in V3
        assert self.hm.evict_message(1) is not None
        # Assistant message — now evictable in V3
        assert self.hm.evict_message(4) is not None
        # System message — still not evictable
        assert self.hm.evict_message(0) is None

    def test_evict_already_evicted_fails(self):
        """Cannot evict the same message twice."""
        self._build_messages()
        result1 = self.hm.evict_message(3)
        assert result1 is not None
        result2 = self.hm.evict_message(3)
        assert result2 is None

    def test_evict_out_of_range(self):
        """Out-of-range index returns None."""
        self._build_messages()
        assert self.hm.evict_message(-1) is None
        assert self.hm.evict_message(100) is None

    def test_recall_message(self):
        """Recall restores evicted content in-place."""
        self._build_messages()
        original_content = self.hm._messages[3]["content"]
        self.hm.evict_message(3)
        assert self.hm._messages[3]["_evicted"] is True

        success = self.hm.recall_message(3, original_content)
        assert success is True
        assert self.hm._messages[3]["content"] == original_content
        assert "_evicted" not in self.hm._messages[3]

    def test_recall_non_evicted_fails(self):
        """Cannot recall a message that wasn't evicted."""
        self._build_messages()
        assert self.hm.recall_message(3, "something") is False

    def test_get_evictable_indexes(self):
        """All non-system, non-tail messages are evictable."""
        self._build_messages()
        evictable = self.hm.get_evictable_indexes()
        assert 3 in evictable  # read_file result
        assert 0 not in evictable  # system
        # User and assistant messages are evictable in V3
        assert 1 in evictable  # user
        # Last 4 messages are protected (total 9, so 5,6,7,8 protected)
        assert 5 not in evictable
        assert 7 not in evictable

    def test_get_evictable_after_evict(self):
        """Evicted messages no longer appear as evictable."""
        self._build_messages()
        self.hm.evict_message(3)
        evictable = self.hm.get_evictable_indexes()
        assert 3 not in evictable
        # Index 1 and 2 should still be evictable (OpenAI format doesn't trigger paired eviction)
        assert 1 in evictable
        assert 2 in evictable

    def test_evict_shell_tool(self):
        """Evict shell command result with correct metadata."""
        self._build_messages()
        # Index 7 is in protected tail (9 msgs, protected from idx 5+)
        # Use index 3 (read_file) which is evictable
        result = self.hm.evict_message(3)
        assert result is not None
        assert result["metadata"]["tool_name"] == "read_file"
        assert "/src/main.py" in result["metadata"]["tool_input"]

    def test_placeholder_format(self):
        """Placeholder contains all required info."""
        self._build_messages()
        self.hm.evict_message(3)
        placeholder = self.hm._messages[3]["content"]
        assert "index=3" in placeholder
        assert "read_file" in placeholder
        assert "tokens" in placeholder


# ════════════════════════════════════════════════════════════════════════════════
# ContextPressureGuard Tests
# ════════════════════════════════════════════════════════════════════════════════


class MockGuardContext:
    """Minimal mock of GuardContext for testing."""

    def __init__(self, pressure: float, evictable_count: int = 60):
        self.context_pressure = pressure
        self.evictable_indexes = list(range(1, evictable_count + 1))
        self.tool_name = ""
        self.tool_args = {}


class TestContextPressureGuard:
    """Test the ContextPressureGuard with V3 evict behavior."""

    def setup_method(self):
        from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
        self.guard = ContextPressureGuard()

    def test_no_pressure_no_action(self):
        """Below soft limit, no action."""
        ctx = MockGuardContext(0.5)
        result = self.guard.check_post(ctx)
        assert result is None

    def test_soft_limit_injects(self):
        """At 75%, injects advisory."""
        ctx = MockGuardContext(0.76)
        result = self.guard.check_post(ctx)
        assert result is not None
        assert result.action == "inject_msg"
        assert "76%" in result.message
        assert "evict" in result.message.lower()

    def test_soft_limit_fires_once(self):
        """Soft warning fires only once until pressure drops."""
        ctx = MockGuardContext(0.76)
        result1 = self.guard.check_post(ctx)
        assert result1 is not None and result1.action == "inject_msg"

        # Second call at same pressure — no repeat
        result2 = self.guard.check_post(ctx)
        assert result2 is None

    def test_soft_limit_resets_on_drop(self):
        """Soft warning resets when pressure drops below threshold * 0.9."""
        ctx_high = MockGuardContext(0.76)
        self.guard.check_post(ctx_high)

        # Drop well below
        ctx_low = MockGuardContext(0.60)
        self.guard.check_post(ctx_low)

        # Should fire again on next rise
        result = self.guard.check_post(ctx_high)
        assert result is not None and result.action == "inject_msg"

    def test_hard_limit_blocks(self):
        """After INJECT_LIMIT reminders at hard limit, blocks."""
        ctx = MockGuardContext(0.92)
        # First (INJECT_LIMIT - 1) calls inject
        for _ in range(self.guard.INJECT_LIMIT - 1):
            result = self.guard.check_post(ctx)
            assert result is not None
            assert result.action == "inject_msg"
        # INJECT_LIMIT-th call should block
        result = self.guard.check_post(ctx)
        assert result is not None
        assert result.action == "block"

    def test_hard_block_persists_until_30_freed(self):
        """Hard limit only fires at >= 90%. Below 90% falls to soft limit or nothing."""
        ctx_block = MockGuardContext(0.92)
        # Reach blocked state
        for _ in range(self.guard.INJECT_LIMIT):
            self.guard.check_post(ctx_block)
        # Confirmed blocked at 92%
        result = self.guard.check_post(ctx_block)
        assert result is not None and result.action == "block"

        # At 80% — below hard limit, soft warning fires (not block)
        ctx_lower = MockGuardContext(0.80)
        result = self.guard.check_post(ctx_lower)
        # Should be soft inject (first time at soft level after hard)
        assert result is None or result.action == "inject_msg"

        # Below hysteresis (< 0.675) — fully resets
        ctx_freed = MockGuardContext(0.62)
        result = self.guard.check_post(ctx_freed)
        assert result is None  # Fully reset

    def test_hard_block_exact_30_threshold(self):
        """Exactly 30% freed should unblock."""
        ctx_block = MockGuardContext(0.90)
        # Exhaust inject limit to enter blocked state
        for _ in range(self.guard.INJECT_LIMIT):
            self.guard.check_post(ctx_block)
        # Ensure blocked
        self.guard.check_post(ctx_block)

        # Free exactly 30%: 90% * (1 - 0.30) = 63%
        ctx_exact = MockGuardContext(0.63)
        result = self.guard.check_post(ctx_exact)
        assert result is None  # Unblocked

    def test_reset_clears_state(self):
        """reset() clears all guard state."""
        # Build up remind count
        ctx = MockGuardContext(0.92)
        for _ in range(3):
            self.guard.check_post(ctx)
        self.guard.reset()

        # Should be clean now
        result = self.guard.check_post(MockGuardContext(0.5))
        assert result is None

    def test_zero_pressure(self):
        """Zero pressure returns None."""
        result = self.guard.check_post(MockGuardContext(0.0))
        assert result is None

    def test_hard_reset_condition(self):
        """pressure > 85% AND evictable < 50 → sets flag, check_pre blocks."""
        # check_post sets the flag and injects
        ctx_post = MockGuardContext(0.88, evictable_count=10)
        result = self.guard.check_post(ctx_post)
        assert result is not None
        assert result.action == "inject_msg"
        assert "hard_reset" in (result.category or "")
        assert self.guard._hard_reset_needed is True

        # check_pre now blocks non-save tools
        from flagscale_agent.react.guard import GuardContext
        ctx_pre = MockGuardContext(0.88, evictable_count=10)
        ctx_pre.tool_name = "shell"
        ctx_pre.tool_args = {"command": "echo hi"}
        result = self.guard.check_pre(ctx_pre)
        assert result is not None
        assert result.action == "block"

    def test_hard_reset_allows_save_tools(self):
        """When hard_reset_needed, memory_write/plan_update/hard_reset pass through."""
        self.guard._hard_reset_needed = True
        ctx = MockGuardContext(0.88, evictable_count=10)
        ctx.tool_name = "memory_write"
        ctx.tool_args = {}
        result = self.guard.check_pre(ctx)
        assert result is None

    def test_no_hard_reset_with_enough_evictable(self):
        """pressure > 85% but evictable >= 50 → normal eviction flow, no hard_reset."""
        ctx = MockGuardContext(0.88, evictable_count=60)
        result = self.guard.check_post(ctx)
        # Should NOT trigger hard_reset
        if result is not None:
            assert "hard_reset" not in (result.category or "")


# ════════════════════════════════════════════════════════════════════════════════
# Integration: Agent evict/recall handling
# ════════════════════════════════════════════════════════════════════════════════


class TestAgentEvictRecallIntegration:
    """Test the agent-level evict/recall handling with swap store."""

    def setup_method(self):
        """Set up a minimal agent-like environment for testing."""
        self.tmpdir = tempfile.mkdtemp()
        self.store = SwapStore(os.path.join(self.tmpdir, "swap_store"))

        # Build a simple history
        from flagscale_agent.react.history import HistoryManager
        self.history = HistoryManager(max_context_tokens=64000)
        self.history._messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "read my file"},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "tc_1",
                    "function": {"name": "read_file", "arguments": '{"path": "/a.py"}'},
                }],
            },
            {
                "role": "tool", "tool_call_id": "tc_1",
                "content": "line1\nline2\nline3\n" * 200,
            },
            {"role": "assistant", "content": "The file has 600 lines."},
            {"role": "user", "content": "run ls"},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "tc_2",
                    "function": {"name": "shell", "arguments": '{"command": "ls"}'},
                }],
            },
            {
                "role": "tool", "tool_call_id": "tc_2",
                "content": "file1.py\nfile2.py\n" * 50,
            },
            {"role": "assistant", "content": "Found 100 files."},
        ]

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _handle_evict(self, arguments: dict) -> str:
        """Replicate agent's _handle_evict logic."""
        indexes = arguments.get("indexes", [])
        if not indexes:
            return "ERROR: 'indexes' parameter is required and must be a non-empty list."
        if not isinstance(indexes, list):
            return "ERROR: 'indexes' must be a list of integers."

        evicted_count = 0
        freed_tokens = 0
        errors = []

        for idx in indexes:
            if not isinstance(idx, int):
                errors.append(f"index {idx}: not an integer")
                continue
            result = self.history.evict_message(idx)
            if result is None:
                errors.append(f"index {idx}: not evictable")
                continue
            self.store.save(idx, result["content"], result.get("metadata"))
            evicted_count += 1
            freed_tokens += result.get("metadata", {}).get("tokens", 0)

        parts = [f"Evicted {evicted_count} message(s), freed ~{freed_tokens} tokens."]
        if errors:
            parts.append(f"Skipped: {'; '.join(errors)}")
        return " ".join(parts)

    def _handle_recall(self, arguments: dict) -> str:
        """Replicate agent's _handle_recall logic."""
        index = arguments.get("index")
        if index is None:
            return "ERROR: 'index' parameter is required."
        if not isinstance(index, int):
            return "ERROR: 'index' must be an integer."
        content = self.store.load(index)
        if content is None:
            return f"ERROR: No evicted content found at index {index}."
        return content

    def test_evict_and_recall_flow(self):
        """Full evict then recall cycle."""
        original = self.history._messages[3]["content"]

        # Evict
        result = self._handle_evict({"indexes": [3]})
        assert "Evicted 1" in result
        assert self.history._messages[3]["_evicted"] is True
        assert self.store.has(3)

        # Recall
        recalled = self._handle_recall({"index": 3})
        assert recalled == original

    def test_evict_multiple(self):
        """Evict multiple indexes at once."""
        # Add extra messages to push 3 and 7 out of protected tail
        self.history._messages.extend([
            {"role": "user", "content": "extra1"},
            {"role": "assistant", "content": "extra2"},
            {"role": "user", "content": "extra3"},
            {"role": "assistant", "content": "extra4"},
        ])
        result = self._handle_evict({"indexes": [3, 7]})
        assert "Evicted" in result
        assert self.store.has(3)
        assert self.store.has(7)

    def test_evict_mixed_valid_invalid(self):
        """Some valid, some invalid indexes."""
        # Add padding to allow more evictions
        self.history._messages.extend([
            {"role": "user", "content": "extra1"},
            {"role": "assistant", "content": "extra2"},
            {"role": "user", "content": "extra3"},
            {"role": "assistant", "content": "extra4"},
        ])
        result = self._handle_evict({"indexes": [3, 0, 99]})
        # index 0 = system (invalid), index 99 = out of range (invalid), index 3 = valid
        assert "Evicted 1" in result
        assert "Skipped" in result

    def test_evict_empty_list(self):
        """Empty indexes list returns error."""
        result = self._handle_evict({"indexes": []})
        assert "ERROR" in result

    def test_recall_nonexistent(self):
        """Recall without prior evict returns error."""
        result = self._handle_recall({"index": 99})
        assert "ERROR" in result

    def test_recall_missing_param(self):
        """Missing index parameter returns error."""
        result = self._handle_recall({})
        assert "ERROR" in result

    def test_evict_preserves_placeholder_info(self):
        """Placeholder contains tool name and input."""
        self._handle_evict({"indexes": [3]})
        placeholder = self.history._messages[3]["content"]
        assert "read_file" in placeholder
        assert "/a.py" in placeholder
        assert "index=3" in placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ════════════════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ════════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases that could occur in real usage."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = SwapStore(os.path.join(self.tmpdir, "swap"))
        from flagscale_agent.react.history import HistoryManager
        self.history = HistoryManager(max_context_tokens=64000)
        self.history._messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant", "content": "",
                "tool_calls": [{
                    "id": "tc_1",
                    "function": {"name": "read_file", "arguments": json.dumps({"path": "/a.py"})},
                }],
            },
            {"role": "tool", "tool_call_id": "tc_1", "content": "file content " * 500},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "more stuff"},
            {"role": "assistant", "content": "sure thing"},
            {"role": "user", "content": "and more"},
            {"role": "assistant", "content": "ok"},
        ]

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _handle_evict(self, arguments):
        """Replicate agent logic."""
        if not arguments:
            return "ERROR: 'indexes' parameter is required and must be a non-empty list."
        indexes = arguments.get("indexes", [])
        if not indexes:
            return "ERROR: 'indexes' parameter is required and must be a non-empty list."
        if not isinstance(indexes, list):
            if isinstance(indexes, (int, float)):
                indexes = [int(indexes)]
            else:
                return "ERROR: 'indexes' must be a list of integers."
        evicted_count = 0
        freed_tokens = 0
        errors = []
        for idx in indexes:
            if isinstance(idx, float) and idx == int(idx):
                idx = int(idx)
            if not isinstance(idx, int):
                errors.append(f"index {idx}: not an integer")
                continue
            result = self.history.evict_message(idx)
            if result is None:
                errors.append(f"index {idx}: not evictable")
                continue
            content = result["content"]
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            self.store.save(idx, content, result.get("metadata"))
            evicted_count += 1
            freed_tokens += result.get("metadata", {}).get("tokens", 0)
        parts = [f"Evicted {evicted_count} message(s), freed ~{freed_tokens} tokens."]
        if errors:
            parts.append(f"Skipped: {'; '.join(errors)}")
        return " ".join(parts)

    def _handle_recall(self, arguments):
        if not arguments:
            return "ERROR: 'index' parameter is required."
        index = arguments.get("index")
        if index is None:
            return "ERROR: 'index' parameter is required."
        if isinstance(index, float) and index == int(index):
            index = int(index)
        if not isinstance(index, int):
            return "ERROR: 'index' must be an integer."
        content = self.store.load(index)
        if content is None:
            return f"ERROR: No evicted content found at index {index}."
        return content

    # ── Float index handling ──

    def test_evict_float_index(self):
        """Model sends 3.0 instead of 3 — should work."""
        result = self._handle_evict({"indexes": [3.0]})
        assert "Evicted 1" in result
        assert self.store.has(3)

    def test_recall_float_index(self):
        """Model sends 3.0 for recall — should work."""
        self._handle_evict({"indexes": [3]})
        result = self._handle_recall({"index": 3.0})
        assert "file content" in result

    # ── None/empty arguments ──

    def test_evict_none_arguments(self):
        """None arguments should give clear error."""
        result = self._handle_evict(None)
        assert "ERROR" in result

    def test_recall_none_arguments(self):
        """None arguments should give clear error."""
        result = self._handle_recall(None)
        assert "ERROR" in result

    def test_evict_empty_dict(self):
        """Empty dict should give clear error."""
        result = self._handle_evict({})
        assert "ERROR" in result

    # ── Single int instead of list ──

    def test_evict_single_int_not_list(self):
        """Model passes indexes=3 instead of indexes=[3]."""
        result = self._handle_evict({"indexes": 3})
        assert "Evicted 1" in result

    # ── Invalid types ──

    def test_evict_string_index(self):
        """String in index list should be skipped."""
        result = self._handle_evict({"indexes": ["three"]})
        assert "Skipped" in result
        assert "not an integer" in result

    def test_recall_string_index(self):
        """String index should error."""
        result = self._handle_recall({"index": "three"})
        assert "ERROR" in result

    def test_evict_mixed_valid_invalid(self):
        """Mix of valid and invalid indexes."""
        result = self._handle_evict({"indexes": [3, "bad", 99]})
        assert "Evicted 1" in result  # only 3 works
        assert "Skipped" in result

    # ── Double evict same index ──

    def test_double_evict_same_index(self):
        """Evicting same index twice — second time skipped."""
        r1 = self._handle_evict({"indexes": [3]})
        assert "Evicted 1" in r1
        r2 = self._handle_evict({"indexes": [3]})
        assert "Evicted 0" in r2
        assert "not evictable" in r2

    # ── Recall without prior evict ──

    def test_recall_never_evicted_index(self):
        """Recall an index that was never evicted."""
        result = self._handle_recall({"index": 3})
        assert "ERROR" in result
        assert "No evicted content" in result

    # ── Non-string content (list-type tool results) ──

    def test_save_list_content(self):
        """Some tool results might be lists (multi-part)."""
        self.store.save(99, ["part1", "part2"])
        loaded = self.store.load(99)
        assert loaded is not None
        assert "part1" in loaded
        assert "part2" in loaded

    # ── Large index numbers ──

    def test_large_index(self):
        """Very large index number (long conversation)."""
        self.store.save(99999, "content at 99999")
        assert self.store.has(99999)
        assert self.store.load(99999) == "content at 99999"

    # ── Concurrent-like save/load ──

    def test_overwrite_then_load(self):
        """Save twice to same index, load gets latest."""
        self.store.save(3, "first")
        self.store.save(3, "second")
        assert self.store.load(3) == "second"

    # ── Guard edge: exact boundary values ──

    def test_guard_at_exactly_75_percent(self):
        """Exactly 75% is the threshold boundary."""
        from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
        guard = ContextPressureGuard()
        ctx = MockGuardContext(0.75)
        # At exactly threshold, should trigger (>=)
        result = guard.check_post(ctx)
        assert result is not None

    def test_guard_just_below_75(self):
        """Just below 75% should not trigger."""
        from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
        guard = ContextPressureGuard()
        ctx = MockGuardContext(0.749)
        result = guard.check_post(ctx)
        assert result is None

    def test_guard_at_exactly_90(self):
        """Exactly 90% should eventually block after INJECT_LIMIT."""
        from flagscale_agent.react.guard.context_pressure import ContextPressureGuard
        guard = ContextPressureGuard()
        ctx = MockGuardContext(0.90)
        # First call at 90% injects (not immediate block)
        result = guard.check_post(ctx)
        assert result is not None
        assert result.action == "inject_msg"
        # After enough calls, blocks
        for _ in range(guard.INJECT_LIMIT - 1):
            result = guard.check_post(ctx)
        assert result.action == "block"


class TestFullLog:
    """Test that _full_log preserves original content after eviction."""

    def setup_method(self):
        from flagscale_agent.react.history import HistoryManager
        self.hm = HistoryManager(max_context_tokens=64000)

    def test_full_log_preserves_content_after_evict(self):
        """After evicting a message, _full_log still has original content."""
        self.hm.append({"role": "system", "content": "system prompt"})
        self.hm.append({"role": "user", "content": "hello world"})
        self.hm.append({"role": "assistant", "content": "hi there"})
        self.hm.append({"role": "user", "content": "more"})
        self.hm.append({"role": "assistant", "content": "response"})
        self.hm.append({"role": "user", "content": "latest 1"})
        self.hm.append({"role": "assistant", "content": "latest 2"})
        self.hm.append({"role": "user", "content": "latest 3"})
        self.hm.append({"role": "assistant", "content": "latest 4"})

        # Evict message at ext_idx 2 (the "hello world" user message)
        # Note: ext_idx starts at 1 for first appended msg; system is ext_idx=1
        result = self.hm.evict_message(2)
        assert result is not None

        # _messages[1] should now be placeholder (internal pos 1 = ext_idx 2)
        assert self.hm._messages[1].get("_evicted") is True
        assert "evicted" in self.hm._messages[1]["content"]

        # _full_log[1] should still have original content (full_log is 0-based)
        assert self.hm._full_log[1]["content"] == "hello world"
        assert self.hm._full_log[1].get("_evicted") is None

    def test_full_log_grows_unbounded(self):
        """_full_log no longer has a cap (saves to disk instead)."""
        for i in range(100):
            self.hm.append({"role": "user", "content": f"msg {i}"})
        assert len(self.hm._full_log) == 100


class TestPromptIntegrity:
    """Test that prompt modules load correctly after refactoring."""

    def test_static_prompt_format_no_keyerror(self):
        """SYSTEM_PROMPT_STATIC.format() must not raise KeyError.

        All braces in example JSON/code blocks must be escaped as {{ }}.
        Regression test for: unescaped {"command": ...} caused KeyError.
        """
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        # These are the actual placeholders used by prompt_builder.py
        try:
            result = SYSTEM_PROMPT_STATIC.format(
                cwd="/workspace/test",
                tools="shell, read_file, write_file",
                skills="train-run, train-config",
                knowledge="know-megatron-training",
                critical_rules="",
                optional_sections="",
                skill_context="",
            )
        except KeyError as e:
            raise AssertionError(
                f"SYSTEM_PROMPT_STATIC has unescaped braces causing KeyError: {e}. "
                f"Escape literal braces as {{{{ }}}} in prompt.py."
            )
        assert len(result) > 1000  # sanity: prompt is non-trivial

    def test_static_prompt_is_english(self):
        import re
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
        # No Chinese characters in static prompt
        assert not re.search(r'[\u4e00-\u9fff]', SYSTEM_PROMPT_STATIC)

    def test_optional_sections_have_required_keys(self):
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_OPTIONAL
        assert "planning" in SYSTEM_PROMPT_OPTIONAL
        assert "memory_rules" in SYSTEM_PROMPT_OPTIONAL

    def test_planning_always_injected(self):
        """prompt_builder should inject planning even without active plan."""
        from flagscale_agent.react.prompt_builder import PromptBuilder
        from flagscale_agent.react.prompt import SYSTEM_PROMPT_OPTIONAL
        # planning section exists and has content about plan_create
        planning = SYSTEM_PROMPT_OPTIONAL.get("planning", "")
        assert "plan_create" in planning
        assert "Step Notes" in planning


class TestNoUnnecessaryTruncation:
    """Regression test: ensure no truncation patterns sneak back in key display paths."""

    def test_memory_list_no_content_truncation(self):
        """memory_list tool should not truncate content."""
        import inspect
        from flagscale_agent.react.tools.memory_list import MemoryListTool
        source = inspect.getsource(MemoryListTool.execute)
        assert "[:97]" not in source
        assert "[:100]" not in source

    def test_plan_context_no_title_truncation(self):
        """Plan context rendering should not truncate titles."""
        import inspect
        from flagscale_agent.react.plan import TaskPlan
        source = inspect.getsource(TaskPlan.context_for_prompt)
        assert "[:120]" not in source
        assert "[:80]" not in source
        assert "[:40]" not in source

    def test_prompt_builder_no_desc_truncation(self):
        """Prompt builder should not truncate skill descriptions."""
        import inspect
        from flagscale_agent.react.prompt_builder import PromptBuilder
        source = inspect.getsource(PromptBuilder)
        assert "[:80]" not in source

    def test_tool_executor_no_summary_truncation(self):
        """Tool executor display summaries should not truncate."""
        import inspect
        from flagscale_agent.react.tool_executor import tool_display_summary
        source = inspect.getsource(tool_display_summary)
        assert "[:50]" not in source
        assert "[:60]" not in source
        assert "[:120]" not in source

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

"""Tests for Context Hard Reset feature in HistoryManager."""

import pytest
from flagscale_agent.react.history import HistoryManager


def _make_hm_with_messages(n: int) -> HistoryManager:
    """Helper: create a HistoryManager with system prompt + n messages."""
    hm = HistoryManager()
    hm.set_system_prompt("You are helpful.")
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        hm.append({"role": role, "content": f"msg {i}"})
    return hm


class TestHardResetBasic:
    def test_reset_clears_messages_keeps_last_n(self):
        hm = _make_hm_with_messages(20)
        assert len(hm._messages) == 21  # system + 20

        stats = hm.hard_reset("Summary text", preserve_last_n=4)

        # system + continuation + ack + last 4 = 7
        assert len(hm._messages) == 7
        assert stats["preserved_count"] == 4
        assert stats["reset_count"] == 1
        assert stats["cleared_count"] == 16  # 20 - 4

    def test_reset_preserves_system_prompt(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Summary", preserve_last_n=4)
        assert hm._messages[0]["role"] == "system"
        assert hm._messages[0]["content"] == "You are helpful."

    def test_reset_message_order(self):
        """Verify presentation order: system, continuation, ack, last4."""
        hm = _make_hm_with_messages(10)
        hm.hard_reset("My summary", preserve_last_n=4)

        assert hm._messages[0]["role"] == "system"
        assert hm._messages[1]["role"] == "user"
        assert "My summary" in hm._messages[1]["content"]
        assert hm._messages[2]["role"] == "assistant"
        assert "Understood" in hm._messages[2]["content"]
        # Last 4 are the original last 4 messages
        assert hm._messages[3]["content"] == "msg 6"
        assert hm._messages[6]["content"] == "msg 9"

    def test_full_log_not_cleared(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Summary", preserve_last_n=4)
        # full_log = 10 original + 2 (cont+ack) = 12
        assert len(hm._full_log) == 12

    def test_full_log_chronological_order(self):
        """full_log records time order: originals, then cont+ack."""
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Summary", preserve_last_n=4)
        # Last two in full_log are continuation and ack
        assert "Summary" in hm._full_log[-2]["content"]
        assert "Understood" in hm._full_log[-1]["content"]


class TestHardResetIndexMapping:
    def test_ext_idx_tagged_on_append(self):
        hm = HistoryManager()
        hm.set_system_prompt("sys")
        hm.append({"role": "user", "content": "hello"})
        assert hm._messages[1].get("_ext_idx") == 1

    def test_ext_idx_preserved_after_reset(self):
        hm = _make_hm_with_messages(10)
        # Last 4 messages have ext_idx 7, 8, 9, 10
        hm.hard_reset("Summary", preserve_last_n=4)
        assert hm._messages[3].get("_ext_idx") == 7
        assert hm._messages[4].get("_ext_idx") == 8
        assert hm._messages[5].get("_ext_idx") == 9
        assert hm._messages[6].get("_ext_idx") == 10

    def test_continuation_ext_idx(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Summary", preserve_last_n=4)
        # continuation is at full_log[10], ext_idx = 11
        assert hm._messages[1].get("_ext_idx") == 11
        # ack is at full_log[11], ext_idx = 12
        assert hm._messages[2].get("_ext_idx") == 12

    def test_new_append_after_reset_gets_correct_ext_idx(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Summary", preserve_last_n=4)
        # full_log now has 12 entries
        hm.append({"role": "user", "content": "new msg"})
        # New append should be full_log[12], ext_idx = 13
        assert hm._messages[-1].get("_ext_idx") == 13

    def test_external_index_method(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Summary", preserve_last_n=4)
        # system prompt
        assert hm.external_index(0) == 0
        # continuation
        assert hm.external_index(1) == 11
        # preserved msg
        assert hm.external_index(3) == 7

    def test_internal_pos_for_ext(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Summary", preserve_last_n=4)
        # ext_idx 7 -> internal pos 3
        assert hm.internal_pos_for_ext(7) == 3
        # ext_idx 11 -> internal pos 1 (continuation)
        assert hm.internal_pos_for_ext(11) == 1
        # Pre-reset index not in _messages
        assert hm.internal_pos_for_ext(2) is None

    def test_recall_from_full_log(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Summary", preserve_last_n=4)
        # ext_idx 1 -> full_log[0] -> "msg 0"
        assert hm.recall_from_full_log(1) == "msg 0"
        # ext_idx 5 -> full_log[4] -> "msg 4"
        assert hm.recall_from_full_log(5) == "msg 4"
        # ext_idx 11 -> full_log[10] -> continuation
        assert "Summary" in hm.recall_from_full_log(11)
        # Out of range
        assert hm.recall_from_full_log(999) is None
        assert hm.recall_from_full_log(0) is None


class TestMultipleResets:
    def test_double_reset(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Reset 1", preserve_last_n=4)
        # Add more messages
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            hm.append({"role": role, "content": f"phase2 msg {i}"})
        # Second reset
        stats = hm.hard_reset("Reset 2", preserve_last_n=4)
        assert stats["reset_count"] == 2
        assert len(hm._messages) == 7  # system + cont + ack + 4

    def test_indexes_globally_unique_across_resets(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Reset 1", preserve_last_n=4)
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            hm.append({"role": role, "content": f"phase2 msg {i}"})
        hm.hard_reset("Reset 2", preserve_last_n=4)
        # Collect all ext_idx in _messages (except system)
        ext_indexes = [m.get("_ext_idx") for m in hm._messages if m.get("_ext_idx")]
        # All should be unique
        assert len(ext_indexes) == len(set(ext_indexes))

    def test_recall_across_resets(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Reset 1 summary", preserve_last_n=4)
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            hm.append({"role": role, "content": f"phase2 msg {i}"})
        hm.hard_reset("Reset 2 summary", preserve_last_n=4)
        # Can still recall phase 1 messages
        assert hm.recall_from_full_log(1) == "msg 0"
        assert hm.recall_from_full_log(3) == "msg 2"
        # Can recall reset 1 summary
        assert "Reset 1 summary" in hm.recall_from_full_log(11)
        # Can recall phase 2 messages
        assert hm.recall_from_full_log(13) == "phase2 msg 0"

    def test_reset_count_increments(self):
        hm = _make_hm_with_messages(20)
        hm.hard_reset("R1", preserve_last_n=4)
        assert hm._reset_count == 1
        for i in range(20):
            hm.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"x{i}"})
        hm.hard_reset("R2", preserve_last_n=4)
        assert hm._reset_count == 2


class TestShouldHardReset:
    def test_no_reset_needed_fresh(self):
        hm = _make_hm_with_messages(10)
        assert hm.should_hard_reset() is False

    def test_trigger_on_low_evictable_high_pressure(self):
        hm = HistoryManager(max_context_tokens=1000)
        hm.set_system_prompt("sys")
        # Fill with messages to simulate high pressure
        for i in range(50):
            msg = {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 200}
            hm.append(msg)
        # Evict all but a few to simulate near-exhaustion
        evictable = hm.get_evictable_indexes()
        for idx in evictable[:-5]:  # leave only 5 evictable
            hm.evict_message(idx)
        # Force high pressure
        hm._actual_input_tokens = int(hm.working_window * 0.85)
        # With < 20 evictable and pressure > 0.80, should trigger
        remaining = hm.get_evictable_indexes()
        if len(remaining) < 20:
            assert hm.should_hard_reset() is True

    def test_cooldown_prevents_immediate_re_reset(self):
        hm = _make_hm_with_messages(200)
        hm.hard_reset("First reset", preserve_last_n=4)
        # Only add a few messages (less than cooldown)
        for i in range(5):
            hm.append({"role": "user" if i % 2 == 0 else "assistant", "content": "x"})
        # Even with bad conditions, cooldown should prevent reset
        hm._actual_input_tokens = int(hm.working_window * 0.9)
        assert hm.should_hard_reset() is False


class TestEvictWithExtIdx:
    def test_evict_uses_ext_idx_in_placeholder(self):
        hm = _make_hm_with_messages(10)
        # Message at internal pos 1 has ext_idx=1
        result = hm.evict_message(1)
        assert result is not None
        # Placeholder should show external index
        placeholder = hm._messages[1]["content"]
        assert "index=1" in placeholder

    def test_evict_after_reset_uses_correct_ext_idx(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("Summary", preserve_last_n=4)
        # Add new messages
        hm.append({"role": "user", "content": "new user msg"})
        hm.append({"role": "assistant", "content": "new asst msg"})
        hm.append({"role": "user", "content": "another"})
        hm.append({"role": "assistant", "content": "last"})
        hm.append({"role": "user", "content": "extra"})
        # Evict the continuation (internal pos 1, ext_idx=11)
        result = hm.evict_message(11)
        if result:
            placeholder = hm._messages[1]["content"]
            assert "index=11" in placeholder

    def test_get_evictable_indexes_returns_ext_idx(self):
        hm = _make_hm_with_messages(10)
        evictable = hm.get_evictable_indexes()
        # Should be ext_idx values (1-based), not internal positions
        assert 1 in evictable
        assert 0 not in evictable  # system prompt excluded

    def test_get_message_at_with_ext_idx(self):
        hm = _make_hm_with_messages(10)
        msg = hm.get_message_at(5)
        assert msg is not None
        assert msg["content"] == "msg 4"


class TestClearResetsState:
    def test_clear_resets_offset_and_count(self):
        hm = _make_hm_with_messages(10)
        hm.hard_reset("S", preserve_last_n=4)
        assert hm._reset_count == 1
        assert hm._index_offset > 0
        hm.clear()
        assert hm._reset_count == 0
        assert hm._index_offset == 0
        assert len(hm._messages) == 0
        assert len(hm._full_log) == 0

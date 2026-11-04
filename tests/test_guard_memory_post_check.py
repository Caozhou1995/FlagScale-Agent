# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may take a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for MemoryPostCheckGuard — always-on post-write / post-read nudges.

The guard has NO throttle: every successful memory write/read fires an
advisory inject; only failure/empty/non-memory paths stay silent.
"""

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.memory_post_check import MemoryPostCheckGuard


def _ctx(tool_name, tool_args=None, tool_result=None):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_result=tool_result,
    )


class TestWriteReconcile:
    def test_no_verdict_on_failed_write(self):
        g = MemoryPostCheckGuard()
        v = g.check_post(_ctx("memory_write", {"key": "fact/x/y"},
                              "ERROR: Invalid key 'fact/x/y'."))
        assert v is None

    def test_write_nudges_immediately_no_cooldown(self):
        g = MemoryPostCheckGuard()
        v = g.check_post(_ctx("memory_write", {"key": "fact/x/y"},
                              "Memorized [fact] 'fact/x/y' (10 chars)."))
        assert v is not None
        assert v.action == "inject"
        assert v.category == "memory_post_check_write"
        assert "fact/x/y" in v.message
        assert "supersedes" in v.message

    def test_same_key_nudges_every_time(self):
        g = MemoryPostCheckGuard()
        results = [
            g.check_post(_ctx("memory_write", {"key": "fact/x/y"},
                              "Memorized [fact] 'fact/x/y' (10 chars)."))
            for _ in range(3)
        ]
        assert all(r is not None for r in results)
        assert all("fact/x/y" in r.message for r in results)

    def test_distinct_keys_each_nudge(self):
        g = MemoryPostCheckGuard()
        a = g.check_post(_ctx("memory_write", {"key": "fact/a/b"},
                              "Memorized [fact] 'fact/a/b' (1 chars)."))
        b = g.check_post(_ctx("memory_write", {"key": "fact/c/d"},
                              "Memorized [fact] 'fact/c/d' (1 chars)."))
        assert a is not None and b is not None
        assert "fact/c/d" in b.message

    def test_write_reconcile_lists_four_points(self):
        g = MemoryPostCheckGuard()
        v = g.check_post(_ctx("memory_write", {"key": "fact/a/b"},
                              "Memorized [fact] 'fact/a/b' (1 chars)."))
        for token in ("CONTRADICTION", "DURABILITY", "GENERALIZATION", "KEY"):
            assert token in v.message
class TestReadReconcile:
    def test_no_verdict_on_empty_read(self):
        g = MemoryPostCheckGuard()
        v = g.check_post(_ctx("memory_read", {"key": "fact/x/y"},
                              "No memory found for 'fact/x/y'."))
        assert v is None

    def test_read_nudge_on_hit_no_cooldown(self):
        g = MemoryPostCheckGuard()
        v = g.check_post(_ctx("memory_read", {"key": "fact/x/y"},
                              "[fact] fact/x/y\nContent: foo\nCreated: now"))
        assert v is not None
        assert v.action == "inject"
        assert v.category == "memory_post_check_read"
        assert "STALENESS" in v.message

    def test_prefix_read_counts_as_hit(self):
        g = MemoryPostCheckGuard()
        v = g.check_post(_ctx("memory_read", {"key": "fact/x/"},
                              "Found 2 entries:\n[fact] fact/x/a: a\n[fact] fact/x/b: b"))
        assert v is not None

    def test_read_nudges_every_time_no_cap(self):
        g = MemoryPostCheckGuard()
        nudges = 0
        for i in range(20):
            v = g.check_post(_ctx("memory_read", {"key": f"fact/x/{i}"},
                                  "[fact] k\nContent: v\nCreated: now"))
            assert v is not None  # always-on: no MAX_READ_NUDGES cap
            nudges += 1
        assert nudges == 20


class TestGuardShape:
    def test_pre_is_always_none(self):
        g = MemoryPostCheckGuard()
        assert g.check_pre(_ctx("memory_write", {"key": "fact/x/y"})) is None
        assert g.check_pre(_ctx("memory_read", {"key": "fact/x/y"})) is None

    def test_non_memory_tools_ignored(self):
        g = MemoryPostCheckGuard()
        assert g.check_post(_ctx("shell", tool_result="Memorized [")) is None
        assert g.check_post(_ctx("evict", tool_result="freed")) is None

    def test_no_throttle_state_on_guard(self):
        g = MemoryPostCheckGuard()
        for attr in ("_since_last", "_nudged_write_keys", "_read_nudges",
                     "COOLDOWN", "MAX_READ_NUDGES"):
            assert not hasattr(g, attr)


class TestWriteDoneCriterion:
    """The write inject must carry its own termination rule, not just a list.

    Observed in the last full evaluation: 22 of 76 trials kept writing memory
    after the plan completed, the worst repeating one key 17 times, because
    the reminder said the write was "not done" but never said what DONE looks
    like. The inject now answers that in text; it still emits on every
    successful write (no state, no per-key dedup), so the two properties are
    asserted together.
    """

    def test_write_inject_states_the_done_criterion(self):
        g = MemoryPostCheckGuard()
        v = g.check_post(_ctx("memory_write", {"key": "fact/a/b"},
                              "Memorized [fact] 'fact/a/b' (1 chars)."))
        assert "DONE CRITERION" in v.message
        assert "ONCE" in v.message

    def test_write_inject_forbids_rewriting_the_same_key(self):
        g = MemoryPostCheckGuard()
        v = g.check_post(_ctx("memory_write", {"key": "fact/a/b"},
                              "Memorized [fact] 'fact/a/b' (1 chars)."))
        low = v.message.lower()
        assert "do not re-write the same key" in low
        assert "second write of the same key" in low

    def test_done_criterion_does_not_introduce_dedup_state(self):
        # Text-only change: the always-on contract must survive it.
        g = MemoryPostCheckGuard()
        for _ in range(3):
            v = g.check_post(_ctx("memory_write", {"key": "fact/a/b"},
                                  "Memorized [fact] 'fact/a/b' (1 chars)."))
            assert v is not None
        for attr in ("_seen_write_keys", "_nudged_write_keys", "_write_counts"):
            assert not hasattr(g, attr)


class TestReadDoneCriterion:
    def test_read_inject_states_the_done_criterion(self):
        g = MemoryPostCheckGuard()
        v = g.check_post(_ctx("memory_read", {"key": "fact/x/y"},
                              "[fact] fact/x/y\nContent: foo\nCreated: now"))
        assert "DONE CRITERION" in v.message
        assert "re-read the same key" in v.message

    def test_read_still_nudges_every_time_after_the_clause(self):
        # The clause must not turn into a throttle.
        g = MemoryPostCheckGuard()
        for i in range(10):
            v = g.check_post(_ctx("memory_read", {"key": f"fact/x/{i}"},
                                  "[fact] k\nContent: v\nCreated: now"))
            assert v is not None

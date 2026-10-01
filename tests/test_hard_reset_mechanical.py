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

"""Tests for the hard-reset MECHANICAL STATE injection (prop_770ca778).

A hard reset replaces the conversation with a MODEL-generated summary, which is
unverified and can silently drop the task's identity (the pov-ray regression
that motivated test_hard_reset_constraints.py). These tests cover the
harness-side fix: the continuation message re-injects load-bearing state
VERBATIM — (1) the original task text, (2) the active plan, (3) the memory keys
touched this session — ahead of the model summary, and degrades safely if any
piece fails.

These are REAL (non-mocked) tests: a real WorkerAgent is constructed over real
HistoryManager / Memory / TaskPlan objects on tmp dirs, so the same code path
the live agent runs is exercised end to end.
"""

import pytest

from flagscale_agent.react.agent import WorkerAgent
from flagscale_agent.react.history import HistoryManager
from flagscale_agent.react.memory import Memory
from flagscale_agent.react.plan import TaskPlan


def _make_agent(tmp_path):
    """Construct an agent exercising only the continuation path.

    `_build_continuation_message` reads exactly three REAL collaborators:
    `self.history`, `self.memory`, `self.task_plan`. Building a full
    WorkerAgent would drag in provider/kernel wiring unrelated to this code,
    so we allocate via __new__ and attach real (not mocked) collaborators on
    isolated tmp dirs. The state objects ARE the production classes.
    """
    sess = tmp_path / "sess"
    sess.mkdir(parents=True, exist_ok=True)
    agent = WorkerAgent.__new__(WorkerAgent)
    agent._session_id = "test-session-123"
    agent._session_dir = str(sess)
    agent.history = HistoryManager()
    agent.memory = Memory(str(tmp_path / "mem"))
    agent.task_plan = TaskPlan(str(sess / "plans"))
    return agent


def _seed_task(agent, task_text):
    agent.history.set_system_prompt("You are helpful.")
    agent.history.append({"role": "user", "content": task_text})
    agent.history.append({"role": "assistant", "content": "ack"})


class TestMechanicalBlockRealAgent:
    def test_continuation_contains_verbatim_task(self, tmp_path):
        agent = _make_agent(tmp_path)
        task = "Build pov-ray v2.2 ONLY - do NOT build v3.7."
        _seed_task(agent, task)
        out = agent._build_continuation_message("MODEL SUMMARY")
        assert "MECHANICAL STATE" in out
        assert task in out, "original task text must be re-injected verbatim"
        assert "MODEL SUMMARY" in out, "model summary must still be present"

    def test_mechanical_block_precedes_model_summary(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_task(agent, "do the thing")
        out = agent._build_continuation_message("MODEL SUMMARY")
        assert out.index("MECHANICAL STATE") < out.index("MODEL SUMMARY")

    def test_continuation_contains_active_plan(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_task(agent, "do the thing")
        agent.task_plan.create(
            "My Plan", ["Step Alpha", "Step Beta"],
            session_id=agent._session_id,
        )
        out = agent._build_continuation_message("SUMMARY")
        assert "Step Alpha" in out, "active plan must be re-injected verbatim"

    def test_continuation_contains_session_memory_keys(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_task(agent, "do the thing")
        agent.memory.put(
            "fact/testdomain/marker", "fact", "content here",
            session_id=agent._session_id,
        )
        out = agent._build_continuation_message("SUMMARY")
        assert "fact/testdomain/marker" in out

    def test_memory_keys_exclude_other_sessions(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_task(agent, "do the thing")
        agent.memory.put("fact/testdomain/mine", "fact", "x",
                         session_id=agent._session_id)
        agent.memory.put("fact/testdomain/theirs", "fact", "y",
                         session_id="some-other-session")
        out = agent._build_continuation_message("SUMMARY")
        assert "fact/testdomain/mine" in out
        assert "fact/testdomain/theirs" not in out

    def test_hard_reset_injects_mechanical_block_into_history(self, tmp_path):
        """The real injection path: the continuation, built by the agent, is
        what HistoryManager stores after a reset."""
        agent = _make_agent(tmp_path)
        task = "the original task statement"
        _seed_task(agent, task)
        cont = agent._build_continuation_message("MODEL SUMMARY")
        agent.history.hard_reset(cont, preserve_last_n=0)
        first_user = next(
            m for m in agent.history._messages if m["role"] == "user")
        assert "MECHANICAL STATE" in first_user["content"]
        assert task in first_user["content"]

    def test_header_reset_number_matches_persisted_count(self, tmp_path):
        """F3 regression: _build_continuation_message runs BEFORE hard_reset()
        increments _reset_count, so the header must label the reset being built
        (#1 on the first reset), agreeing with the persisted reset_count."""
        agent = _make_agent(tmp_path)
        _seed_task(agent, "task")
        cont = agent._build_continuation_message("SUMMARY")
        assert "[Context Hard Reset #1 - conversation auto-compacted]" in cont
        # and after the reset the persisted count is also 1
        agent.history.hard_reset(cont, preserve_last_n=0)
        assert agent.history._reset_count == 1


class TestMechanicalBlockDegradation:
    def test_no_plan_no_memory_still_well_formed(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_task(agent, "task text")
        out = agent._build_continuation_message("SUMMARY")
        assert "MECHANICAL STATE" in out
        assert "(no active plan)" in out
        assert "(no memory keys written this session)" in out
        assert "SUMMARY" in out

    def test_empty_history_yields_unavailable_placeholder(self, tmp_path):
        agent = _make_agent(tmp_path)
        # No user turn at all -> placeholder, not an exception/crash.
        block = agent._build_mechanical_state_block()
        assert "MECHANICAL STATE" in block
        assert "unavailable" in block

    def test_helper_failure_degrades_to_unavailable(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_task(agent, "task text")
        # Break the memory object: helper must return a placeholder, not raise.
        agent.memory = object()  # no list_entries()
        block = agent._build_mechanical_state_block()
        assert block != "", "block must survive a single helper failure"
        assert "(unavailable: AttributeError)" in block

    def test_total_block_failure_returns_empty(self, tmp_path, monkeypatch):
        agent = _make_agent(tmp_path)
        _seed_task(agent, "task text")
        # Each helper degrades internally, so to exercise the OUTER guard we
        # force a helper itself to raise — the assembler must swallow it and
        # return "" (the reset then falls back to the model-summary-only format).
        def _boom():
            raise RuntimeError("boom")
        monkeypatch.setattr(agent, "_mech_original_task", _boom)
        assert agent._build_mechanical_state_block() == ""

    def test_all_collaborators_broken_still_well_formed(self, tmp_path):
        """Every helper catches its own exception -> a well-formed block with
        placeholders, never a crash."""
        agent = _make_agent(tmp_path)
        _seed_task(agent, "task text")
        agent.history = object()
        agent.memory = object()
        agent.task_plan = object()
        block = agent._build_mechanical_state_block()
        assert block != ""
        assert block.count("unavailable") >= 3


class TestMechanicalBlockCaps:
    def test_oversized_task_truncated(self, tmp_path):
        agent = _make_agent(tmp_path)
        huge = "X" * (agent._MECH_TASK_CAP + 5000)
        _seed_task(agent, huge)
        out = agent._mech_original_task()
        assert "[truncated]" in out
        assert len(out) < agent._MECH_TASK_CAP + 200

    def test_continuation_skips_prior_reset_continuation(self, tmp_path):
        """The task text is the first NON-continuation user turn — a REAL harness
        continuation (exact header shape) injected earlier must not be mistaken
        for the original task."""
        agent = _make_agent(tmp_path)
        agent.history.set_system_prompt("sys")
        # Use the EXACT header _build_continuation_message emits. A loose prefix
        # ("#1 - x") is not the harness shape and must NOT be relied on.
        agent.history.append({"role": "user", "content":
            "[Context Hard Reset #1 - conversation auto-compacted]\n"
            "Previous conversation: 2264 messages total\nstale summary body"})
        real_task = "the real task statement"
        agent.history.append({"role": "user", "content": real_task})
        out = agent._mech_original_task()
        assert out == real_task

    def test_task_starting_with_marker_prefix_is_kept(self, tmp_path):
        """F1 regression: a REAL task that merely starts with '[Context Hard Reset'
        (e.g. it quotes an earlier reset paste) must NOT be skipped as a
        continuation."""
        agent = _make_agent(tmp_path)
        agent.history.set_system_prompt("sys")
        task = ("[Context Hard Reset #1 - quoted operator paste]\n"
                "build pov-ray v2.2 only")
        agent.history.append({"role": "user", "content": task})
        assert agent._mech_original_task() == task

    def test_single_marker_task_not_lost(self, tmp_path):
        """F1 regression: if the ONLY user turn starts with the marker prefix but
        is not the harness header, it must still be returned (not lost)."""
        agent = _make_agent(tmp_path)
        agent.history.set_system_prompt("sys")
        task = "[Context Hard Reset #0 - build pov-ray v2.2"
        agent.history.append({"role": "user", "content": task})
        assert agent._mech_original_task() == task

    def test_none_content_user_turn_skipped(self, tmp_path):
        """F2 regression: a user turn with content=None must be skipped, not
        leaked as the literal string 'None'."""
        agent = _make_agent(tmp_path)
        agent.history.set_system_prompt("sys")
        agent.history.append({"role": "user", "content": None})
        real_task = "the real task"
        agent.history.append({"role": "user", "content": real_task})
        assert agent._mech_original_task() == real_task


class TestMechanicalLastUserTurn:
    """(1b) LAST 5 USER TURNS: the recent REAL user inputs are the current
    directive chain and must survive a reset verbatim — not by tail-4 luck."""

    def test_last_turn_in_continuation_between_1_and_2(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_task(agent, "turn-1 task")
        agent.history.append({"role": "user", "content": "turn-2 directive"})
        out = agent._build_continuation_message("SUMMARY")
        assert "### (1b) LAST 5 USER TURNS" in out
        assert "turn-2 directive" in out, "latest real user turn verbatim"
        # Ordering: (1) < (1b) < (2)
        i1 = out.index("### (1) ORIGINAL TASK")
        i1b = out.index("### (1b) LAST 5 USER TURNS")
        i2 = out.index("### (2) ACTIVE PLAN")
        assert i1 < i1b < i2

    def test_session_input_history_is_primary_source(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_task(agent, "turn-1 task")
        agent.history.append({"role": "user", "content": "tool flood below"})
        agent.history.append({"role": "user", "content": [
            {"type": "tool_result", "content": "42"}]})
        agent.history.append({"role": "assistant", "content": "done"})
        # Primary source: only genuine user inputs enter this list.
        agent._session_input_history = ["first directive", "latest directive"]
        # Multi-entry -> [-N] timeline, oldest->newest, [-1] = CURRENT.
        out = agent._mech_last_user_turn()
        assert out == ("[-2] first directive\n"
                       "[-1] (CURRENT directive) latest directive")

    def test_fallback_scans_full_log_when_history_list_empty(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._session_input_history = []
        agent.history.set_system_prompt("sys")
        agent.history.append({"role": "user", "content": "earlier directive"})
        agent.history.append({"role": "user", "content": [
            {"type": "tool_result", "content": "tool output"}]})
        agent.history.append({"role": "assistant", "content": "ok"})
        agent.history.append({"role": "user", "content": "the real last turn"})
        # Fallback collects newest-first then reverses: same timeline order.
        out = agent._mech_last_user_turn()
        assert out == ("[-2] earlier directive\n"
                       "[-1] (CURRENT directive) the real last turn")

    def test_fallback_skips_harness_continuation_header(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._session_input_history = []
        agent.history.set_system_prompt("sys")
        agent.history.append({"role": "user", "content":
            "[Context Hard Reset #1 - conversation auto-compacted]\n"
            "Previous conversation: 100 messages total\nstale summary"})
        agent.history.append({"role": "user", "content": "the real last turn"})
        assert agent._mech_last_user_turn() == "the real last turn"

    def test_empty_history_degrades_to_placeholder(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._session_input_history = []
        # No user turns at all -> placeholder, never a crash.
        assert "unavailable" in agent._mech_last_user_turn()

    def test_history_list_missing_degrades(self, tmp_path):
        agent = _make_agent(tmp_path)
        # _make_agent never sets _session_input_history: getattr default path
        # must fall through to the full-log scan (empty here -> placeholder).
        assert not hasattr(agent, "_session_input_history")
        block = agent._build_mechanical_state_block()
        assert "(1b) LAST 5 USER TURNS" in block
        assert "unavailable" in block

    def test_oversized_last_turn_truncated(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._session_input_history = ["Y" * (agent._MECH_LAST_TURN_CAP + 500)]
        out = agent._mech_last_user_turn()
        assert "[truncated]" in out
        assert len(out) < agent._MECH_LAST_TURN_CAP + 200

    def test_helper_failure_degrades(self, tmp_path):
        agent = _make_agent(tmp_path)
        _seed_task(agent, "task")
        # Break the history collaborator: _mech_last_user_turn's internal
        # try/except must degrade to a placeholder, not propagate.
        agent.history = object()  # no _full_log
        block = agent._build_mechanical_state_block()
        assert block != ""
        assert "(unavailable: AttributeError)" in block

    def test_reset_after_tool_flood_keeps_last_directive(self, tmp_path):
        """THE load-bearing scenario: a >4-message tool-call flood after the
        last real user directive; the reset continuation still shows it."""
        agent = _make_agent(tmp_path)
        _seed_task(agent, "turn-1 task")
        for i in range(10):
            agent.history.append({"role": "user", "content": [
                {"type": "tool_result", "content": f"out {i}"}]})
            agent.history.append({"role": "assistant",
                                  "content": f"thinking {i}"})
        last_directive = "commit the fix now, then stop"
        agent.history.append({"role": "user", "content": [
            {"type": "tool_result", "content": "final output"}]})
        agent.history.append({"role": "user", "content": last_directive})
        for i in range(6):
            agent.history.append({"role": "user", "content": [
                {"type": "tool_result", "content": f"post {i}"}]})
            agent.history.append({"role": "assistant",
                                  "content": f"work {i}"})
        cont = agent._build_continuation_message("SUMMARY")
        assert last_directive in cont
        agent.history.hard_reset(cont, preserve_last_n=4)
        cont2 = agent._build_continuation_message("SUMMARY2")
        assert last_directive in cont2

    def test_primary_source_skips_guard_injection(self, tmp_path):
        """Guard block injections enter _full_log as bare unmarked user
        messages but NEVER enter _session_input_history — so on the
        interactive path the primary source shields the last directive."""
        agent = _make_agent(tmp_path)
        _seed_task(agent, "turn-1 task")
        directive = "run the precision check next"
        agent.history.append({"role": "user", "content": directive})
        agent._session_input_history = ["turn-1 task", directive]
        # Simulate a guard block injection AFTER the directive (raw, unmarked).
        agent.history.append({"role": "user",
                              "content": "[blocked by guard] OVERRIDE REQUIRED"})
        out = agent._mech_last_user_turn()
        assert "[-1] (CURRENT directive) run the precision check next" in out
        assert "[blocked by guard]" not in out

    def test_fallback_known_limit_guard_injection_returned(self, tmp_path):
        """DOCUMENTED LIMIT (not a bug to fix silently): with the primary
        source empty (single-shot/worker paths), a guard block injection is
        the latest unmarked user message and the fallback returns it. This
        test PINS the behavior so a future marker mechanism flips it
        deliberately."""
        agent = _make_agent(tmp_path)
        agent._session_input_history = []
        agent.history.set_system_prompt("sys")
        agent.history.append({"role": "user", "content": "real directive"})
        agent.history.append({"role": "user",
                              "content": "[blocked by guard] OVERRIDE REQUIRED"})
        out = agent._mech_last_user_turn()
        assert out.endswith("[blocked by guard] OVERRIDE REQUIRED")
        assert "[-1] (CURRENT directive) [blocked by guard]" in out

    def test_depth_cap_five_entries(self, tmp_path):
        """Only the last _MECH_LAST_TURN_DEPTH=5 turns are kept, numbered
        [-5]..[-1], regardless of how many came before."""
        agent = _make_agent(tmp_path)
        agent._session_input_history = [f"directive-{i}" for i in range(8)]
        out = agent._mech_last_user_turn()
        for i in range(3, 8):
            assert f"directive-{i}" in out
        for i in range(0, 3):
            assert f"directive-{i}" not in out
        assert "[-5] directive-3" in out
        assert "[-2] directive-6" in out
        assert "[-1] (CURRENT directive) directive-7" in out
        lines = out.split("\n")
        assert len(lines) == 5

    def test_fewer_than_depth_still_numbered_from_minus_n(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._session_input_history = ["a", "b", "c"]
        out = agent._mech_last_user_turn()
        assert out == ("[-3] a\n"
                       "[-2] b\n"
                       "[-1] (CURRENT directive) c")

    def test_single_entry_stays_plain_text(self, tmp_path):
        """Backward-compat: a single last turn renders WITHOUT the [-1]
        marker (plain text), preserving the original single-directive form."""
        agent = _make_agent(tmp_path)
        agent._session_input_history = ["only directive"]
        assert agent._mech_last_user_turn() == "only directive"

    def test_tool_flood_between_directives_no_gaps_in_chain(self, tmp_path):
        """Tool-result floods between user turns do not consume depth slots:
        the chain is the last 5 REAL turns, not the last 5 messages."""
        agent = _make_agent(tmp_path)
        agent.history.set_system_prompt("sys")
        for i in range(7):
            agent.history.append({"role": "user", "content": f"d-{i}"})
            agent.history.append({"role": "user", "content": [
                {"type": "tool_result", "content": f"noise-{i}"}]})
            agent.history.append({"role": "assistant", "content": "w"})
        agent._session_input_history = []
        out = agent._mech_last_user_turn()
        assert "[-5] d-2" in out
        assert "[-1] (CURRENT directive) d-6" in out
        assert "noise-" not in out

    def test_chain_in_continuation_after_reset(self, tmp_path):
        """The directive chain survives INTO the post-reset continuation."""
        agent = _make_agent(tmp_path)
        _seed_task(agent, "turn-1 task")
        for d in ("run tests", "check the log", "commit the fix"):
            agent.history.append({"role": "user", "content": d})
            hist = getattr(agent, "_session_input_history", None)
            if hist is None:
                hist = []
                agent._session_input_history = hist
            hist.append(d)
        for i in range(6):
            agent.history.append({"role": "user", "content": [
                {"type": "tool_result", "content": f"out {i}"}]})
            agent.history.append({"role": "assistant", "content": "w"})
        cont = agent._build_continuation_message("SUMMARY")
        assert "[-3] run tests" in cont
        assert "[-2] check the log" in cont
        assert "[-1] (CURRENT directive) commit the fix" in cont

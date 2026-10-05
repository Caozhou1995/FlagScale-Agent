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

"""Session-state resilience tests.

Two contracts:
1. recall_search parent-log fallback (proposal b82d3f17): worker/reviewer
   sessions live under <parent_session>/subagents/<task_id>/ and their own
   directory may have no conversation_full.json — recall_search must then
   search the parent session's log instead of erroring out.
2. Plan lifecycle persistence (proposal 949b768b): a fresh TaskPlan bound to
   an existing session's plans dir must keep operating on the persisted
   active plan across instance recreation / session restore, including when
   the active.yaml pointer is missing or corrupt (the on-disk plan scan is
   the source of truth). Backward compat: a plans dir with no plan at all
   still reports "No active plan".
"""

import json
import os
import shutil
import tempfile

import pytest
import yaml

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.tools.recall_search import RecallSearchTool


def _write_log(session_dir, messages):
    """Write a minimal conversation_full.json with the given messages."""
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, "conversation_full.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"messages": messages, "index_offset": 0,
                   "reset_count": 0, "turn_count": len(messages)}, f)
    return path


PARENT_MESSAGES = [
    {"role": "user", "content": "kick off the training job"},
    {"role": "assistant", "content": "launching FlagScale now"},
    {"role": "user", "content": [
        {"type": "tool_result", "content": "NCCL timeout on rank 3"},
    ]},
    {"role": "assistant", "content": "the NCCL timeout came from a bad NIC"},
    {"role": "user", "content": "parent-only anchor phrase flockxyz"},
]


@pytest.fixture
def worker_session():
    """A parent session with its own log plus a subagent dir with NO log."""
    with tempfile.TemporaryDirectory() as root:
        parent_dir = os.path.join(root, "sess_parent")
        child_dir = os.path.join(parent_dir, "subagents", "task_abc")
        os.makedirs(child_dir)
        _write_log(parent_dir, PARENT_MESSAGES)
        yield parent_dir, child_dir


class TestRecallSearchParentFallback:
    def test_worker_without_own_log_searches_parent(self, worker_session):
        parent_dir, child_dir = worker_session
        tool = RecallSearchTool(child_dir)
        out = tool.execute(query="flockxyz")
        assert "ERROR" not in out
        assert "1 hit(s)" in out
        assert "flockxyz" in out

    def test_worker_fallback_hits_carry_parent_indexes(self, worker_session):
        parent_dir, child_dir = worker_session
        tool = RecallSearchTool(child_dir)
        out = tool.execute(query="NCCL timeout")
        # ext index = message position + 1 (same rule as the primary log)
        assert "[index=3" in out
        assert "2 hit(s)" in out

    def test_worker_fallback_marks_parent_log_in_output(self, worker_session):
        parent_dir, child_dir = worker_session
        tool = RecallSearchTool(child_dir)
        out = tool.execute(query="flockxyz")
        # exact fallback marker, not an accidental "parent" from a dir name
        assert "parent session log (fallback)" in out

    def test_primary_log_run_has_no_fallback_marker(self, worker_session):
        parent_dir, child_dir = worker_session
        _write_log(child_dir, [{"role": "user", "content": "local note"}])
        tool = RecallSearchTool(child_dir)
        out = tool.execute(query="local note")
        assert "fallback" not in out

    def test_worker_fallback_role_filter(self, worker_session):
        parent_dir, child_dir = worker_session
        tool = RecallSearchTool(child_dir)
        out = tool.execute(query="NCCL", role="assistant")
        assert "1 hit(s)" in out

    def test_no_log_anywhere_errors_with_both_paths(self, worker_session):
        parent_dir, child_dir = worker_session
        os.remove(os.path.join(parent_dir, "conversation_full.json"))
        tool = RecallSearchTool(child_dir)
        out = tool.execute(query="flockxyz")
        assert "ERROR" in out
        # both attempted paths named for debuggability
        assert os.path.join(child_dir, "conversation_full.json") in out
        assert os.path.join(parent_dir, "conversation_full.json") in out

    def test_own_log_takes_precedence_over_parent(self, worker_session):
        parent_dir, child_dir = worker_session
        _write_log(child_dir, [{"role": "user", "content": "child-local flockxyz"}])
        tool = RecallSearchTool(child_dir)
        out = tool.execute(query="flockxyz")
        assert "child-local" in out
        # the parent's message with the same keyword must NOT be scanned
        assert "scanned 1 messages" in out

    def test_sibling_subagent_log_is_not_searched(self, worker_session):
        parent_dir, child_dir = worker_session
        sibling = os.path.join(parent_dir, "subagents", "task_other")
        _write_log(sibling, [{"role": "user", "content": "sibling flockxyz"}])
        tool = RecallSearchTool(child_dir)
        out = tool.execute(query="flockxyz")
        assert "sibling" not in out

    def test_worker_with_no_log_at_all_still_errors(self, tmp_path):
        # Standalone dir with neither own nor parent log — backward compat.
        tool = RecallSearchTool(str(tmp_path))
        out = tool.execute(query="anything")
        assert "ERROR" in out


class TestPlanActiveSurvival:
    """Pin the anchor's prescribed restore sequence + hostile pointer states.

    Anchor sequence: create plan -> construct a fresh PlanManager(TaskPlan)
    against the same session dir -> plan_update must operate on the plan.
    """

    def _fresh_manager(self, plan_dir):
        return TaskPlan(plan_dir)

    def test_update_after_manager_recreation(self, tmp_path):
        d = str(tmp_path)
        tp1 = TaskPlan(d)
        tp1.create("Restore me", ["A", "B"])
        tp2 = self._fresh_manager(d)
        plan = tp2.update_step(1, "done", "survived restore")
        assert plan["steps"][0]["status"] == "done"
        assert tp2.get_active()["title"] == "Restore me"

    def test_update_with_active_pointer_deleted(self, tmp_path):
        d = str(tmp_path)
        tp1 = TaskPlan(d)
        created = tp1.create("Pointer lost", ["A"])
        os.remove(os.path.join(d, "active.yaml"))
        tp2 = self._fresh_manager(d)
        plan = tp2.update_step(1, "done")
        assert plan["id"] == created["id"]
        # pointer file is re-established by the next save
        assert os.path.isfile(os.path.join(d, "active.yaml"))

    def test_update_with_active_pointer_corrupt(self, tmp_path):
        d = str(tmp_path)
        tp1 = TaskPlan(d)
        created = tp1.create("Pointer corrupt", ["A"])
        with open(os.path.join(d, "active.yaml"), "w") as f:
            f.write("{not yaml!!!")
        tp2 = self._fresh_manager(d)
        plan = tp2.update_step(1, "done")
        assert plan["id"] == created["id"]

    def test_update_with_dangling_active_pointer(self, tmp_path):
        d = str(tmp_path)
        tp1 = TaskPlan(d)
        created = tp1.create("Dangling ref", ["A"])
        with open(os.path.join(d, "active.yaml"), "w") as f:
            yaml.dump({"active_id": "plan_gone000000"}, f)
        tp2 = self._fresh_manager(d)
        plan = tp2.update_step(1, "done")
        assert plan["id"] == created["id"]

    def test_step_state_survives_restore(self, tmp_path):
        d = str(tmp_path)
        tp1 = TaskPlan(d)
        tp1.create("Progress kept", ["A", "B", "C"])
        tp1.update_step(1, "done")
        tp2 = self._fresh_manager(d)
        plan = tp2.get_active()
        statuses = {s["id"]: s["status"] for s in plan["steps"]}
        assert statuses == {1: "done", 2: "doing", 3: "pending"}

    def test_no_plan_at_all_still_reports_none(self, tmp_path):
        d = str(tmp_path)
        tp = TaskPlan(d)
        with pytest.raises(ValueError, match="No active plan"):
            tp.update_step(1, "done")
        assert tp.get_active() is None

    def test_no_plan_at_all_summary(self, tmp_path):
        tp = TaskPlan(str(tmp_path))
        assert tp.summary() == "No active plan."

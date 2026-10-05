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

"""Regression tests for a cluster of guard fixes.

Covers five defects discovered by source analysis + run evidence:
  1. KnowledgeSkillGuard deadlock: a write blocked with "[PostEditBlocked]"
     mandates a cold re-read, but the re-read itself advanced the 40-call
     counter and got blocked by the very same guard.
  2. VerificationGuard wrap-up: re-reported the SAME open proposals verbatim
     at every completion — long lists drown the new signal.
  3. Background job health monitor false-positive kill of a deliberate
     `sleep N; <probe>` command (idle monitor counts a declared sleep as a
     silent stall and kills the job).
  4. StartupGuard/ResearchPhase did not accept memory_read/recall_search as
     a research pass (internal knowledge retrieval is an info-gain act).
  5. Reviewer/diverger demand suppression keyed on a PROSE substring of the
     rendered contract instead of the structured constraints.reviewer flag.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.knowledge_skill import KnowledgeSkillGuard
from flagscale_agent.react.guard.plan import _self_is_diverger_worker
from flagscale_agent.react.guard.startup import (
    ResearchPhase,
    _RESEARCH_TOOLS,
    install_background_monitor_exemption,
    sleep_probe_quiet_budget,
)
from flagscale_agent.react.guard.verification import (
    VerificationGuard,
    _self_is_reviewer_worker,
)
from flagscale_agent.react.multi_agent.wiring import (
    CONTRACT_PATH_ENV, TASK_ID_ENV,
)
from flagscale_agent.react.tools.shell import ShellJobsTool


_POST_EDIT_BLOCKED = (
    "[BLOCKED BY GUARD] This tool call was prevented.\n\n"
    "[PostEditBlocked] This edit did NOT land — cold-verify the far end."
)


# --------------------------------------------------------------------------
# Fix 1 — knowledge gate must not deadlock on a guard-mandated re-read
# --------------------------------------------------------------------------

class TestKnowledgeRereadExemption:
    def test_mandated_reread_is_exempt_and_does_not_count(self):
        g = KnowledgeSkillGuard()
        g._calls_since_knowledge = 39  # one call from the 40-call block

        # A write was blocked with the PostEditBlocked mandate.
        g.check_post(GuardContext(
            tool_name="edit_file", tool_args={"path": "/x/a.py"},
            tool_result=_POST_EDIT_BLOCKED,
        ))

        # The mandated cold re-read must be let through (no block, no count).
        v = g.check_pre(GuardContext(
            tool_name="read_file", tool_args={"path": "/x/a.py"}))
        assert v is None, "mandated re-read must be exempt from the block"
        g.check_post(GuardContext(
            tool_name="read_file", tool_args={"path": "/x/a.py"},
            tool_result="file body"))
        assert g._calls_since_knowledge == 39, "exempt read must not count"

        # The re-issued edit is exempt too, and clears the mandate.
        v2 = g.check_pre(GuardContext(
            tool_name="edit_file", tool_args={"path": "/x/a.py"}))
        assert v2 is None, "re-issued edit must be exempt"
        assert not getattr(g, "_mandated_reverify", False), "mandate consumed"

        # A DIFFERENT path is NOT exempt — normal counting resumes.
        v3 = g.check_pre(GuardContext(
            tool_name="read_file", tool_args={"path": "/x/other.py"}))
        assert v3 is not None and v3.action == "block", (
            "a non-mandated call at the threshold must still block")

    def test_blanket_prose_without_mandate_does_not_exempt(self):
        g = KnowledgeSkillGuard()
        g._calls_since_knowledge = 39
        v = g.check_pre(GuardContext(
            tool_name="read_file", tool_args={"path": "/x/a.py"}))
        assert v is not None and v.action == "block"

    def test_stale_mandate_cleared_by_other_tool(self):
        g = KnowledgeSkillGuard()
        g._calls_since_knowledge = 10
        g.check_post(GuardContext(
            tool_name="edit_file", tool_args={"path": "/x/a.py"},
            tool_result=_POST_EDIT_BLOCKED))
        assert getattr(g, "_mandated_reverify", False)
        # An unrelated real call makes the mandate stale.
        g.check_post(GuardContext(
            tool_name="shell", tool_args={"command": "ls"}, tool_result="ok"))
        assert not getattr(g, "_mandated_reverify", False), "stale mandate cleared"


# --------------------------------------------------------------------------
# Fix 2 — wrap-up reports only the DELTA of open proposals
# --------------------------------------------------------------------------

class _FakeRegistry:
    def __init__(self, lines):
        self._lines = lines

    def render_open(self):
        return self._lines


_HEADER = "Open improvement proposals still awaiting review:"
_P1 = "  • prop_aaaa1111 (proposed): [agent-code] fix A"
_P2 = "  • prop_bbbb2222 (proposed): [skill] do B"
_P3 = "  • prop_cccc3333 (proposed): [knowledge] doc C"


class TestWrapupDelta:
    def test_first_wrapup_lists_all_then_only_delta(self):
        reg = _FakeRegistry("\n".join([_HEADER, _P1, _P2]))
        g = VerificationGuard(proposals=reg)

        first = g._text_complete_hygiene_message()
        assert "prop_aaaa1111" in first and "prop_bbbb2222" in first

        # Same registry, second completion: nothing new → no verbatim re-list.
        second = g._text_complete_hygiene_message()
        assert "prop_aaaa1111" not in second, "already-reported must not repeat"
        assert "prop_bbbb2222" not in second
        assert "no NEW or CHANGED" in second

        # A genuinely new proposal → only IT is listed.
        reg._lines = "\n".join([_HEADER, _P1, _P2, _P3])
        third = g._text_complete_hygiene_message()
        assert "prop_cccc3333" in third
        assert "prop_aaaa1111" not in third
        assert "prop_bbbb2222" not in third

    def test_status_change_relisted(self):
        reg = _FakeRegistry("\n".join([_HEADER, _P1]))
        g = VerificationGuard(proposals=reg)
        g._text_complete_hygiene_message()
        # Same id, changed status line → treated as delta (re-listed).
        reg._lines = "\n".join(
            [_HEADER, _P1.replace("(proposed)", "(approved)")])
        msg = g._text_complete_hygiene_message()
        assert "prop_aaaa1111" in msg
        assert "(approved)" in msg

class TestResearchGate:
    def test_memory_tools_accepted_as_research(self):
        assert "memory_read" in _RESEARCH_TOOLS
        assert "recall_search" in _RESEARCH_TOOLS
        # Fresh phase: memory_read is let through and satisfies the gate.
        ph = ResearchPhase()
        assert ph.check(GuardContext(
            tool_name="memory_read", tool_args={"key": "x"})) is None
        ph.observe_post(GuardContext(
            tool_name="memory_read", tool_args={"key": "x"}, tool_result="v"))
        assert ph.is_satisfied(), "memory_read must satisfy the research phase"
        # After satisfaction, a real work call is allowed.
        assert ph.check(GuardContext(
            tool_name="write_file", tool_args={"path": "/x"})) is None

    def test_recall_search_accepted(self):
        ph = ResearchPhase()
        assert ph.check(GuardContext(
            tool_name="recall_search", tool_args={"query": "x"})) is None
        ph.observe_post(GuardContext(
            tool_name="recall_search", tool_args={"query": "x"},
            tool_result="hits"))
        assert ph.is_satisfied()

    def test_block_message_names_memory_tools(self):
        ph = ResearchPhase()
        v = ph.check(GuardContext(
            tool_name="write_file", tool_args={"path": "/x"}))
        assert v is not None and v.action == "block"
        assert "memory_read" in v.message or "recall_search" in v.message


# --------------------------------------------------------------------------
# Fix 3 — sleep-probe jobs exempt from false-positive idle kill
# --------------------------------------------------------------------------

_GRACE = 120


class _FakeKillEvaluator:
    def evaluate(self, *a, **k):
        return {
            "status": "ok", "should_kill": True, "kill_reason": "silent stall",
            "activity": "CPU 0%", "output_changed": False, "stall_count": 9,
            "recent_text": "",
        }


class TestSleepProbe:
    def test_predicate(self):
        assert sleep_probe_quiet_budget(
            "sleep 75; echo probe-ready") == 75 + _GRACE
        assert sleep_probe_quiet_budget(
            "sleep 30 && sleep 45 && curl -sI http://x") == 75 + _GRACE
        # A bare `sleep N` is a NORMAL command — a hung sleep must still be
        # monitorable/killable. Only a declared sleep FOLLOWED by a probe is
        # exempt.
        assert sleep_probe_quiet_budget("sleep 75") is None
        assert sleep_probe_quiet_budget("sleep 30") is None
        # Not bare-sleep-first — must NOT be exempt (could mask a real stall).
        assert sleep_probe_quiet_budget("cd /x && sleep 75; probe") is None
        assert sleep_probe_quiet_budget("env -u FOO sleep 75; x") is None
        assert sleep_probe_quiet_budget("time sleep 60") is None
        assert sleep_probe_quiet_budget("tail -f /dev/null") is None

    def test_health_tick_exempts_declared_sleep(self):
        install_background_monitor_exemption()
        tool = ShellJobsTool()
        job = SimpleNamespace(
            command="sleep 75; echo probe-ready", evaluator=None,
            health_sampler=None, health_note="", stdout_chunks=[],
            stderr_chunks=[], start=time.time())
        tick = tool._health_tick(job)
        assert tick["kill"] is False

    def test_non_sleep_still_killed(self):
        install_background_monitor_exemption()
        tool = ShellJobsTool()
        job = SimpleNamespace(
            command="tail -f /dev/null", evaluator=_FakeKillEvaluator(),
            health_sampler=object(), health_note="", stdout_chunks=[],
            stderr_chunks=[], start=time.time())
        tick = tool._health_tick(job)
        assert tick["kill"] is True

    def test_sleep_probe_past_budget_still_killed(self):
        install_background_monitor_exemption()
        tool = ShellJobsTool()
        job = SimpleNamespace(
            command="sleep 10; echo x", evaluator=_FakeKillEvaluator(),
            health_sampler=object(), health_note="", stdout_chunks=[],
            stderr_chunks=[], start=time.time() - 999)
        tick = tool._health_tick(job)
        assert tick["kill"] is True, "past the quiet budget, normal kill resumes"


# --------------------------------------------------------------------------
# Fix 5 — suppression keyed on structured constraints.reviewer, not prose
# --------------------------------------------------------------------------

def _make_contract(tmp_path, monkeypatch, reviewer, prompt_body):
    task_dir = tmp_path / "task_x"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "contract.prompt").write_text(prompt_body, encoding="utf-8")
    (task_dir / "contract.json").write_text(
        json.dumps({"constraints": {"reviewer": reviewer, "writable": []}}),
        encoding="utf-8")
    monkeypatch.setenv(TASK_ID_ENV, "task_x")
    monkeypatch.setenv(CONTRACT_PATH_ENV, str(task_dir / "contract.prompt"))


class TestSuppressionFlag:
    def test_reviewer_keyed_on_flag_not_prose(self, tmp_path, monkeypatch):
        # reviewer=True but NO "## Reviewer discipline" prose → still suppressed.
        _make_contract(tmp_path, monkeypatch, True, "just a plain contract")
        assert _self_is_reviewer_worker() is True

    def test_reviewer_prose_without_flag_does_not_suppress(
            self, tmp_path, monkeypatch):
        # The banned failure mode: prose present but flag False → NOT suppressed.
        _make_contract(tmp_path, monkeypatch, False,
                       "## Reviewer discipline\nread-only review body")
        assert _self_is_reviewer_worker() is False

    def test_diverger_keyed_on_flag_not_prose(self, tmp_path, monkeypatch):
        _make_contract(tmp_path, monkeypatch, True, "plain diverger contract")
        assert _self_is_diverger_worker() is True
        # Prose present, flag absent/False → NOT suppressed.
        _make_contract(tmp_path, monkeypatch, False,
                       "Propose 2-3 genuinely different framings of the task")
        assert _self_is_diverger_worker() is False

    def test_no_env_is_false(self, monkeypatch):
        monkeypatch.delenv(TASK_ID_ENV, raising=False)
        monkeypatch.delenv(CONTRACT_PATH_ENV, raising=False)
        assert _self_is_reviewer_worker() is False
        assert _self_is_diverger_worker() is False

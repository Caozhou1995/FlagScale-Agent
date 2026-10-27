"""Reviewer N1-N4 fixes (second round) — regression tests.

N1: forbidden lines must survive the FULL per-turn chain
    (plan -> _build_plan_context -> _build_dashboard), not be dropped
    by the dashboard's title/step parser.
N2: add_forbidden on a duplicate action with new why/instead REFRESHES
    the stored entry (the cap-full advice must be implementable).
N3: over-cap create drops are disclosed as a third state, distinct
    from malformed/duplicate, with a do-not-resend instruction.
N4: _normalize_forbidden must not mutate the caller's list, and the
    forbidden_meta must be exact for JSON-blob input.
"""
import json
import tempfile

import pytest

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.prompt_builder import PromptBuilder


def _plan(tmp_path, forbidden):
    tp = TaskPlan(plan_dir=str(tmp_path))
    tp.create(title="T", steps=["s1"], forbidden=forbidden)
    return tp


def _producer_context(tp):
    """Mirror agent.WorkerAgent._build_plan_context's relevant shape."""
    active = tp.get_active()
    lines = [f'<active-plan title="{active.get("title", "")}">']
    lines.append("<current-forbidden>")
    lines.extend(tp._forbidden_lines(active))
    lines.append("</current-forbidden>")
    lines.append("</active-plan>")
    return "\n".join(lines)


class TestN1DashboardCarriesForbidden:
    def test_dashboard_shows_list(self, tmp_path):
        tp = _plan(tmp_path, [{"action": "do not spin on probes",
                               "why": "burns budget", "instead": "secure it"}])
        d = PromptBuilder(None)._build_dashboard(_producer_context(tp))
        assert "不做清单" in d
        assert "do not spin on probes" in d

    def test_dashboard_empty_plan_clean(self):
        d = PromptBuilder(None)._build_dashboard('<active-plan title="T2"></active-plan>')
        assert "不做清单" not in d

    def test_dashboard_tags_removed_from_step_scan(self, tmp_path):
        """The forbidden text may contain '[x] Step' shapes; extraction must
        remove the block BEFORE step counting so it cannot inflate totals."""
        tp = _plan(tmp_path, [{"action": "never trust [🔄] Step 9 patterns",
                               "why": "", "instead": ""}])
        ctx = _producer_context(tp)
        d = PromptBuilder(None)._build_dashboard(ctx)
        assert "Step:" not in d  # no step markers exist OUTSIDE the block

    def test_producer_wraps_in_tags(self, tmp_path):
        tp = _plan(tmp_path, [{"action": "no X"}])
        ctx = _producer_context(tp)
        assert "<current-forbidden>" in ctx and "</current-forbidden>" in ctx


class TestN2RefreshOnDuplicate:
    def test_resend_refreshes_why_instead(self, tmp_path):
        tp = _plan(tmp_path, [{"action": "act0", "why": "w0", "instead": ""}])
        r = tp.add_forbidden([{"action": "act0", "why": "NEW", "instead": "NEW2"}])
        assert r["added"] == 0 and r["skipped_duplicates"] == 1
        assert r["refreshed"] == 1
        stored = [e for e in tp.get_active()["forbidden"] if e["action"] == "act0"][0]
        assert stored["why"] == "NEW" and stored["instead"] == "NEW2"

    def test_resend_without_new_text_leaves_entry(self, tmp_path):
        tp = _plan(tmp_path, [{"action": "act0", "why": "w0"}])
        r = tp.add_forbidden([{"action": "act0"}])
        stored = [e for e in tp.get_active()["forbidden"] if e["action"] == "act0"][0]
        assert stored["why"] == "w0" and r["refreshed"] == 0


class TestN3CapDisclosure:
    def test_cap_state_named(self, tmp_path):
        from flagscale_agent.react.tools.plan_create import PlanCreateTool
        t = PlanCreateTool.__new__(PlanCreateTool)
        plan = {"forbidden": [{"action": f"a{i}"} for i in range(8)],
                "forbidden_meta": {"requested": 10, "dropped": 2}}
        out = PlanCreateTool._forbidden_report(t, plan)
        assert "上限" in out and "不要重发" in out

    def test_malformed_state_still_advises_resent(self, tmp_path):
        from flagscale_agent.react.tools.plan_create import PlanCreateTool
        t = PlanCreateTool.__new__(PlanCreateTool)
        plan = {"forbidden": [{"action": "a"}],
                "forbidden_meta": {"requested": 3, "dropped": 2}}
        out = PlanCreateTool._forbidden_report(t, plan)
        assert "缺 action 文本或重复" in out


class TestN4NoCallerMutation:
    def test_caller_list_unmutated_and_meta_exact(self, tmp_path):
        tp = TaskPlan(plan_dir=str(tmp_path))
        caller = [json.dumps([{"action": "a1", "why": "w1"},
                              {"action": "a2"}, {"action": "a3"}])]
        tp.create(title="T", steps=["s1"], forbidden=caller)
        assert len(caller) == 1  # caller's list untouched
        p = tp.get_active()
        assert p["forbidden_meta"] == {"requested": 1, "dropped": 0}
        assert len(p["forbidden"]) == 3

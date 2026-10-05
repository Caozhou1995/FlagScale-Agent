# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the RoleSpec mechanism: reviewer preset, live-tree
pointer check, auto-injected reviewer role tag, and the report-completeness
gate — one mechanism across multi_agent files.

Proposals under test:
  [81c81515] reviewer preset: deadline >=10min; inputs must carry the
             deliverable abs path + rev; acceptance forces a non-empty
             findings.md with a per-finding table; re-review role that only
             verifies diff fixes.
  [76b18db8] review contracts verify input pointers are the LIVE tree (the
             location the running system actually imports).
  [b56f8276] spawn auto-injects constraints.reviewer=true on a review goal;
             suppression keyed on the EXPLICIT role tag, never prose.
  [91aab676 + fa9b99a4] report-completeness gate: >=1 finding or an explicit
             'no findings' line, no dangling '(IN PROGRESS' header; enforced
             at WRITE time in report_result.py and mirrored in the reviewer
             contract template acceptance.
"""

import subprocess
import uuid

import pytest

from flagscale_agent.react.multi_agent.contract import (
    Contract,
    ContractError,
    ROLE_RE_REVIEWER,
    ROLE_REVIEWER,
    check_report_completeness,
    has_dangling_in_progress,
    has_no_findings_line,
    is_reviewer_role,
    report_has_findings,
    resolve_role,
)
from flagscale_agent.react.multi_agent.ledger import RUNNING, TaskLedger
from flagscale_agent.react.multi_agent.report_result import ReportResultTool
from flagscale_agent.react.multi_agent.spawn import (
    SpawnWorkerTool,
    _is_review_goal,
    _render_contract,
    apply_reviewer_preset,
    maybe_auto_inject_reviewer,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FLAGSCALE_TASK_ID", raising=False)
    monkeypatch.delenv("FLAGSCALE_LIVE_TREE_CHECK", raising=False)


def _deliverable(tmp_path, name="deliverable.py"):
    d = tmp_path / name
    d.write_text("x = 1\n", encoding="utf-8")
    return d


def _reviewer_contract(tmp_path, deliverable=None, goal=None, cons_over=None,
                       **over):
    cons = {
        "writable": [str(tmp_path)],
        "forbidden": ["modify any file"],
        "reviewer": True,
        "max_minutes": 10,
    }
    if cons_over:
        cons.update(cons_over)
    deliverable = deliverable or _deliverable(tmp_path)
    kw = dict(
        goal=goal or "Review the deliverable for correctness bugs",
        constraints=cons,
        acceptance=[{"check": "test -s out", "kind": "check_command"}],
        output_ptr=str(tmp_path / "findings.md"),
        inputs=[{"kind": "path", "value": str(deliverable)},
                {"kind": "value", "value": "rev=abc1234"}],
    )
    kw.update(over)
    return Contract.build(**kw)


class TestResolveRole:
    def test_reviewer_flag_true(self):
        assert resolve_role({"reviewer": True}) == ROLE_REVIEWER

    def test_reviewer_flag_false_is_normal(self):
        # Explicit suppression: reviewer:false behaves like a normal task.
        assert resolve_role({"reviewer": False}) is None

    def test_role_key_re_reviewer(self):
        assert resolve_role({"role": ROLE_RE_REVIEWER}) == ROLE_RE_REVIEWER

    def test_role_key_beats_reviewer_flag(self):
        assert resolve_role(
            {"role": ROLE_RE_REVIEWER, "reviewer": False}
        ) == ROLE_RE_REVIEWER

    def test_absent_constraints_is_normal(self):
        assert resolve_role({}) is None
        assert resolve_role(None) is None

    def test_unknown_role_is_normal(self):
        assert resolve_role({"role": "worker"}) is None

    def test_is_reviewer_role(self):
        assert is_reviewer_role({"reviewer": True})
        assert is_reviewer_role({"role": ROLE_RE_REVIEWER})
        assert not is_reviewer_role({})
        assert not is_reviewer_role({"reviewer": False})


class TestIsReviewGoal:
    def test_review_words_match(self):
        assert _is_review_goal("Review the change for bugs")
        assert _is_review_goal("independently REVIEW the diff")
        assert _is_review_goal("re-review the fixes")
        assert _is_review_goal("audit the config")

    def test_word_boundary_no_substring(self):
        # b56f8276: suppression/detection must not be prose-substring matching;
        # "preview" is NOT a review goal.
        assert not _is_review_goal("preview the report")
        assert not _is_review_goal("count files")

    def test_empty_goal(self):
        assert not _is_review_goal("")
        assert not _is_review_goal(None)


class TestAutoInjectReviewerFlag:
    def test_review_goal_without_tag_injects(self):
        cons = {"writable": ["/tmp"]}
        maybe_auto_inject_reviewer("review the diff", cons)
        assert cons["reviewer"] is True

    def test_explicit_false_suppresses(self):
        # The explicit tag wins — injection never overrides it.
        cons = {"writable": ["/tmp"], "reviewer": False}
        maybe_auto_inject_reviewer("review the diff", cons)
        assert cons["reviewer"] is False

    def test_explicit_true_untouched(self):
        cons = {"writable": ["/tmp"], "reviewer": True}
        maybe_auto_inject_reviewer("review the diff", cons)
        assert cons["reviewer"] is True

    def test_explicit_role_key_suppresses(self):
        cons = {"writable": ["/tmp"], "role": "worker"}
        maybe_auto_inject_reviewer("review the diff", cons)
        assert "reviewer" not in cons

    def test_non_review_goal_no_inject(self):
        cons = {"writable": ["/tmp"]}
        maybe_auto_inject_reviewer("count files", cons)
        assert "reviewer" not in cons
    def test_re_review_goal_injects_reviewer_flag(self):
        # Only the boolean flag is auto-injected; the re-review ROLE stays an
        # explicit choice (role key), per the tag-not-prose rule.
        cons = {"writable": ["/tmp"]}
        maybe_auto_inject_reviewer("re-review the fixes", cons)
        assert cons["reviewer"] is True
        assert "role" not in cons


class TestReviewerPreset:
    def _cons(self, tmp_path, **over):
        cons = {"writable": [str(tmp_path)], "reviewer": True, "max_minutes": 5}
        cons.update(over)
        return cons

    def _inputs(self, tmp_path):
        d = _deliverable(tmp_path)
        return [{"kind": "path", "value": str(d)},
                {"kind": "value", "value": "rev=abc1234"}]

    def test_deadline_bumped_to_10(self, tmp_path):
        cons = self._cons(tmp_path)
        _, dm = apply_reviewer_preset(
            "review the deliverable", cons, [], str(tmp_path / "findings.md"),
            5, inputs=self._inputs(tmp_path))
        assert dm == 10
        assert cons["max_minutes"] == 10

    def test_longer_deadline_kept(self, tmp_path):
        cons = self._cons(tmp_path, max_minutes=25)
        _, dm = apply_reviewer_preset(
            "review the deliverable", cons, [], str(tmp_path / "findings.md"),
            25, inputs=self._inputs(tmp_path))
        assert dm == 25

    def test_acceptance_injected(self, tmp_path):
        cons = self._cons(tmp_path)
        ptr = str(tmp_path / "findings.md")
        acc, _ = apply_reviewer_preset(
            "review the deliverable", cons, [], ptr, 5,
            inputs=self._inputs(tmp_path))
        checks = [a["check"] for a in acc]
        # non-empty findings.md + completeness (same helper as the write
        # gate) — the injected completeness command covers the table-or-
        # no-findings AND no-dangling-(IN PROGRESS rules.
        assert any(f"test -s {ptr}" in c for c in checks)
        assert any("check_report_completeness" in c for c in checks)
        assert all(a.get("kind") == "check_command" for a in acc)

    def test_missing_deliverable_input_rejected(self, tmp_path):
        cons = self._cons(tmp_path)
        with pytest.raises(ContractError, match="deliverable"):
            apply_reviewer_preset(
                "review", cons, [], str(tmp_path / "findings.md"), 5,
                inputs=[{"kind": "value", "value": "rev=x"}])

    def test_missing_rev_rejected(self, tmp_path):
        cons = self._cons(tmp_path)
        d = _deliverable(tmp_path)
        with pytest.raises(ContractError, match="rev"):
            apply_reviewer_preset(
                "review", cons, [], str(tmp_path / "findings.md"), 5,
                inputs=[{"kind": "path", "value": str(d)}])

    def test_rev_via_constraints_ok(self, tmp_path):
        cons = self._cons(tmp_path, rev="abc1234")
        d = _deliverable(tmp_path)
        _, dm = apply_reviewer_preset(
            "review", cons, [], str(tmp_path / "findings.md"), 5,
            inputs=[{"kind": "path", "value": str(d)}])
        assert dm == 10

    def test_re_review_role_gets_preset(self, tmp_path):
        cons = self._cons(tmp_path, role=ROLE_RE_REVIEWER)
        acc, dm = apply_reviewer_preset(
            "verify diff fixes", cons, [], str(tmp_path / "findings.md"), 5,
            inputs=self._inputs(tmp_path))
        assert dm == 10
        assert acc

    def test_normal_role_untouched(self, tmp_path):
        cons = {"writable": [str(tmp_path)], "max_minutes": 3}
        base = [{"check": "true", "kind": "check_command"}]
        acc, dm = apply_reviewer_preset(
            "count files", cons, base, str(tmp_path / "out.json"), 3, inputs=[])
        assert dm == 3 and acc == base
        assert cons["max_minutes"] == 3


class TestInjectedAcceptanceRuns:
    """The injected acceptance items are REAL shell commands the parent runs —
    they must pass on a complete report and fail on an incomplete one."""

    def _acc(self, tmp_path, role=None):
        cons = {"writable": [str(tmp_path)], "reviewer": True,
                "max_minutes": 10}
        if role:
            cons["role"] = role
        d = _deliverable(tmp_path)
        ptr = str(tmp_path / "findings.md")
        acc, _ = apply_reviewer_preset(
            "review the deliverable", cons, [], ptr, 10,
            inputs=[{"kind": "path", "value": str(d)},
                    {"kind": "value", "value": "rev=abc1234"}])
        return acc, ptr

    def _run_all(self, acc, cwd):
        for a in acc:
            r = subprocess.run(a["check"], shell=True, cwd=str(cwd))
            assert r.returncode == 0, a["check"]

    def test_complete_report_passes_all(self, tmp_path):
        acc, ptr = self._acc(tmp_path)
        (tmp_path / "findings.md").write_text(
            "# Review\n\n| F1 | missing guard | fix in spawn.py |\n",
            encoding="utf-8")
        self._run_all(acc, tmp_path)

    def test_no_findings_line_passes_all(self, tmp_path):
        acc, ptr = self._acc(tmp_path)
        (tmp_path / "findings.md").write_text(
            "# Review\n\nno findings\n", encoding="utf-8")
        self._run_all(acc, tmp_path)

    def test_in_progress_report_fails(self, tmp_path):
        acc, ptr = self._acc(tmp_path)
        (tmp_path / "findings.md").write_text(
            "## Findings (IN PROGRESS\n\n| F1 | x | y |\n", encoding="utf-8")
        bad = [a["check"] for a in acc
               if "check_report_completeness" in a["check"]]
        assert bad
        for c in bad:
            rc = subprocess.run(c, shell=True, cwd=str(tmp_path)).returncode
            assert rc != 0, c

    def test_findings_less_report_fails(self, tmp_path):
        acc, ptr = self._acc(tmp_path)
        (tmp_path / "findings.md").write_text(
            "# Review\n\nlooks fine overall\n", encoding="utf-8")
        bad = [a["check"] for a in acc
               if "check_report_completeness" in a["check"]]
        assert bad
        for c in bad:
            rc = subprocess.run(c, shell=True, cwd=str(tmp_path)).returncode
            assert rc != 0, c

    def test_skeleton_table_fails_completeness_cmd(self, tmp_path):
        # F2 (reviewer round): header+separator rows with zero data rows must
        # fail the injected completeness command, not just the Python gate.
        acc, ptr = self._acc(tmp_path)
        (tmp_path / "findings.md").write_text(
            "# Review\n\n| ID | Sev | Fix |\n| --- | --- | --- |\n",
            encoding="utf-8")
        bad = [a["check"] for a in acc
               if "check_report_completeness" in a["check"]]
        assert bad
        for c in bad:
            rc = subprocess.run(c, shell=True, cwd=str(tmp_path)).returncode
            assert rc != 0, c
class TestLiveTreePointerCheck:
    """[76b18db8] Review contracts must verify input pointers are the LIVE
    tree — the location the CURRENT process actually imports. Recorded
    failure: the review targeted a stale package copy while the harness
    imported the real one, so the first diff pass read the wrong file."""

    def _make_pkg(self, base, name):
        pkg = base / name
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        return pkg

    def _setup(self, tmp_path, monkeypatch):
        # Two same-named package copies; the STALE one is what this process
        # imports (prepended to sys.path) — mirroring the recorded incident
        # where the stale copy was the importable one.
        name = f"livechk_{uuid.uuid4().hex[:8]}"
        live = self._make_pkg(tmp_path / "live", name)
        stale = self._make_pkg(tmp_path / "stale", name)
        monkeypatch.syspath_prepend(str(tmp_path / "stale"))
        return live, stale

    def test_stale_copy_rejected(self, tmp_path, monkeypatch):
        live, stale = self._setup(tmp_path, monkeypatch)
        # Input points at the copy the process does NOT import -> stale.
        c = _reviewer_contract(tmp_path, deliverable=live)
        with pytest.raises(ContractError, match="LIVE tree"):
            c.validate()

    def test_imported_copy_passes(self, tmp_path, monkeypatch):
        live, stale = self._setup(tmp_path, monkeypatch)
        c = _reviewer_contract(tmp_path, deliverable=stale)
        c.validate()  # no raise

    def test_file_inside_live_pkg_checked(self, tmp_path, monkeypatch):
        live, stale = self._setup(tmp_path, monkeypatch)
        mod = live / "mod.py"
        mod.write_text("", encoding="utf-8")
        c = _reviewer_contract(tmp_path, deliverable=mod)
        with pytest.raises(ContractError, match="LIVE tree"):
            c.validate()

    def test_non_package_input_skipped(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        c = _reviewer_contract(tmp_path, deliverable=plain)
        c.validate()  # no raise: not a python package, nothing to compare

    def test_unimportable_name_skipped(self, tmp_path):
        name = f"notimported_{uuid.uuid4().hex[:8]}"
        pkg = self._make_pkg(tmp_path / "elsewhere", name)
        c = _reviewer_contract(tmp_path, deliverable=pkg)
        c.validate()  # no raise: the process imports no such module

    def test_normal_role_not_gated(self, tmp_path, monkeypatch):
        # The live-tree check is reviewer-role-gated: normal contracts with a
        # coincidentally-shadowing path stay valid.
        live, stale = self._setup(tmp_path, monkeypatch)
        d = _deliverable(tmp_path)
        c = Contract.build(
            goal="count files",
            constraints={"writable": [str(tmp_path)], "max_minutes": 5},
            acceptance=[{"check": "true", "kind": "check_command"}],
            output_ptr=str(tmp_path / "out.json"),
            inputs=[{"kind": "path", "value": str(live)}],
        )
        c.validate()  # no raise

    def test_kill_switch_env(self, tmp_path, monkeypatch):
        live, stale = self._setup(tmp_path, monkeypatch)
        monkeypatch.setenv("FLAGSCALE_LIVE_TREE_CHECK", "0")
        c = _reviewer_contract(tmp_path, deliverable=live)
        c.validate()  # no raise: check disabled by env


class TestCompletenessHelpers:
    def test_table_row_counts_as_finding(self):
        assert report_has_findings("| F1 | bug | fix |")
        assert report_has_findings("text\n| F1 | a | b |\n")
        assert report_has_findings("| F1 | note |\nno findings elsewhere")

    def test_plain_prose_is_not_a_table(self):
        assert not report_has_findings("no findings at all")
        assert not report_has_findings("")

    def test_table_skeleton_is_not_a_finding(self):
        # F2 (reviewer round): header + separator rows with no data row must
        # NOT count as findings — otherwise a zero-finding skeleton passes.
        assert not report_has_findings("| ID | Sev | Fix |\n| --- | --- | --- |")
        assert not report_has_findings("| --- | --- |")
        assert report_has_findings("| F1 | med | fix the gate |")

    def test_skeleton_report_rejected_end_to_end(self, tmp_path):
        p = tmp_path / "r.md"
        p.write_text(
            "# Review\n\n| ID | Sev | Fix |\n| --- | --- | --- |\n",
            encoding="utf-8")
        out = check_report_completeness(str(p))
        assert out is not None and "finding" in out

    def test_no_findings_line(self):
        assert has_no_findings_line(
            "## Findings\n\nNo findings — the diff is clean.")
        assert not has_no_findings_line("findings table follows")
        # F3 (reviewer round): 'no findings' echoed inside guidance prose
        # must NOT satisfy the check — line-anchored, markdown-prefix aware.
        assert not has_no_findings_line(
            "...or state 'no findings', then finalize")
        assert not has_no_findings_line(
            "add a table or state no findings, then finalize")
        assert has_no_findings_line("- no findings")
        assert has_no_findings_line("**no findings** — clean diff")

    def test_dangling_in_progress(self):
        assert has_dangling_in_progress("## Findings (IN PROGRESS")
        assert has_dangling_in_progress("(IN PROGRESS: fill table)")
        assert has_dangling_in_progress("  ### findings table (IN PROGRESS")
        assert not has_dangling_in_progress(
            "we saw a marker (IN PROGRESS) mid-sentence")
        assert not has_dangling_in_progress("| F1 | ok |")

    def test_check_report_completeness(self, tmp_path):
        p = tmp_path / "r.md"
        assert check_report_completeness(str(p)) is not None  # missing file
        p.write_text("", encoding="utf-8")
        assert "empty" in check_report_completeness(str(p))
        p.write_text("## Findings (IN PROGRESS\n", encoding="utf-8")
        assert "IN PROGRESS" in check_report_completeness(str(p))
        p.write_text("nothing to report\n", encoding="utf-8")
        assert check_report_completeness(str(p)) is not None
        p.write_text("| F1 | x | y |\n", encoding="utf-8")
        assert check_report_completeness(str(p)) is None
        p.write_text("no findings\n", encoding="utf-8")
        assert check_report_completeness(str(p)) is None
    def test_no_findings_or_table_rejected(self, tmp_path):
        p = tmp_path / "r.md"
        p.write_text("| F1 | x | y |\n", encoding="utf-8")
        assert check_report_completeness(str(p)) is None
        p.write_text("plain prose only\n", encoding="utf-8")
        out = check_report_completeness(str(p))
        assert out is not None and "finding" in out


class TestReportResultCompletenessGate:
    """[fa9b99a4] At WRITE time in report_result.py: a reviewer-role task
    whose output report fails the completeness check gets the report_result
    call REJECTED with guidance to finalize; the task stays RUNNING."""

    @pytest.fixture
    def led(self, tmp_path):
        return TaskLedger(str(tmp_path / "tasks"))

    def _running(self, led, tmp_path, role=None, reviewer=True):
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        d = _deliverable(work)
        cons = {"writable": [str(work)], "max_minutes": 10}
        if reviewer:
            cons["reviewer"] = True
        if role:
            cons["role"] = role
        c = Contract.build(
            goal="Review the deliverable for correctness bugs",
            constraints=cons,
            acceptance=[{"check": "test -s out", "kind": "check_command"}],
            output_ptr=str(work / "findings.md"),
            inputs=[{"kind": "path", "value": str(d)},
                    {"kind": "value", "value": "rev=abc1234"}],
        )
        led.create(c)
        led.transition(c.id, RUNNING, pid=111)
        return c

    def _report(self, tmp_path, text):
        (tmp_path / "work" / "findings.md").write_text(text, encoding="utf-8")

    def _execute(self, led, task_id):
        import os as _os
        _os.environ["FLAGSCALE_TASK_ID"] = task_id
        try:
            return ReportResultTool(ledger=led).execute(summary="done")
        finally:
            _os.environ.pop("FLAGSCALE_TASK_ID", None)

    def test_empty_report_rejected(self, led, tmp_path):
        c = self._running(led, tmp_path)
        self._report(tmp_path, "")
        out = self._execute(led, c.id)
        assert out.startswith("ERROR")
        assert "finalize" in out
        assert led.get(c.id).status == RUNNING

    def test_dangling_in_progress_rejected(self, led, tmp_path):
        c = self._running(led, tmp_path)
        self._report(tmp_path, "## Findings (IN PROGRESS\n| F1 | a | b |\n")
        out = self._execute(led, c.id)
        assert out.startswith("ERROR")
        assert "IN PROGRESS" in out
        assert led.get(c.id).status == RUNNING

    def test_findings_less_report_rejected(self, led, tmp_path):
        c = self._running(led, tmp_path)
        self._report(tmp_path, "looks good overall\n")
        out = self._execute(led, c.id)
        assert out.startswith("ERROR")
        assert led.get(c.id).status == RUNNING

    def test_findings_table_reported(self, led, tmp_path):
        c = self._running(led, tmp_path)
        self._report(tmp_path, "| F1 | bug | fixed |\n")
        assert self._execute(led, c.id).startswith("reported")

    def test_no_findings_reported(self, led, tmp_path):
        c = self._running(led, tmp_path)
        self._report(tmp_path, "no findings\n")
        assert self._execute(led, c.id).startswith("reported")

    def test_missing_report_rejected(self, led, tmp_path):
        c = self._running(led, tmp_path)
        out = self._execute(led, c.id)
        assert out.startswith("ERROR")
        assert led.get(c.id).status == RUNNING

    def test_non_reviewer_not_gated(self, led, tmp_path):
        c = self._running(led, tmp_path, reviewer=False)
        self._report(tmp_path, "")
        assert self._execute(led, c.id).startswith("reported")

    def test_re_reviewer_gated(self, led, tmp_path):
        c = self._running(led, tmp_path, role=ROLE_RE_REVIEWER)
        self._report(tmp_path, "")
        out = self._execute(led, c.id)
        assert out.startswith("ERROR")
        assert led.get(c.id).status == RUNNING


class TestRenderRoleLines:
    def test_re_review_render_lines(self, tmp_path):
        c = _reviewer_contract(
            tmp_path, cons_over={"role": ROLE_RE_REVIEWER})
        text = _render_contract(c)
        assert "## Reviewer discipline" in text
        assert "RE-REVIEW ROLE" in text
        assert "verify ONLY that each previously reported" in text

    def test_role_tag_only_render_gets_discipline(self, tmp_path):
        # F1 (reviewer round): a contract that carries ONLY the explicit role
        # tag (reviewer flag false/absent) must still get the full reviewer
        # discipline render — resolve_role, not the boolean flag, is the gate.
        c = _reviewer_contract(
            tmp_path, cons_over={"role": ROLE_RE_REVIEWER,
                                 "reviewer": False})
        text = _render_contract(c)
        assert "## Reviewer discipline" in text
        assert "RE-REVIEW ROLE" in text

    def test_reviewer_render_has_completeness_expectation(self, tmp_path):
        # [91aab676] the completeness check belongs in the reviewer contract
        # template as an acceptance suggestion.
        c = _reviewer_contract(tmp_path)
        text = _render_contract(c)
        assert "no findings" in text
        assert "IN PROGRESS" in text

    def test_normal_render_has_no_role_lines(self, tmp_path):
        c = _reviewer_contract(
            tmp_path,
            goal="count files",
            cons_over={"reviewer": False},
            inputs=[],
        )
        text = _render_contract(c)
        assert "## Reviewer discipline" not in text


class _FakeProc:
    def __init__(self, pid=4321):
        self.pid = pid


class TestSpawnIntegration:
    """spawn.execute wires injection + preset into the frozen contract."""

    @pytest.fixture
    def led(self, tmp_path):
        return TaskLedger(str(tmp_path / "tasks"))

    def _spawn(self, led, tmp_path, monkeypatch, goal, cons, inputs=None):
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProc())
        tool = SpawnWorkerTool(ledger=led)
        return tool.execute(
            goal=goal,
            constraints=cons,
            acceptance=[{"kind": "check_command", "check": "true"}],
            output_ptr=str(tmp_path / "work" / "out.md"),
            deadline_minutes=5,
            inputs=inputs,
        )

    def test_auto_inject_and_preset_in_contract(self, led, tmp_path,
                                                monkeypatch):
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        d = _deliverable(work)
        out = self._spawn(
            led, tmp_path, monkeypatch, "review the diff for bugs",
            {"writable": [str(work)]},
            inputs=[{"kind": "path", "value": str(d)},
                    {"kind": "value", "value": "rev=abc1234"}])
        assert out.startswith("spawned"), out
        tid = out.split()[2]
        rec = led.get(tid)
        assert rec.contract.constraints.get("reviewer") is True
        # 1 base + 2 preset-injected acceptance items (test -s + the
        # check_report_completeness command that shares the write-gate logic).
        assert len(rec.contract.acceptance) == 3
        assert rec.contract.constraints["max_minutes"] == 10

    def test_preset_missing_rev_fails_spawn(self, led, tmp_path, monkeypatch):
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        d = _deliverable(work)
        out = self._spawn(
            led, tmp_path, monkeypatch, "review the diff for bugs",
            {"writable": [str(work)]},
            inputs=[{"kind": "path", "value": str(d)}])
        assert out.startswith("ERROR")
        assert "rev" in out

    def test_no_injection_for_plain_goal(self, led, tmp_path, monkeypatch):
        work = tmp_path / "work"
        work.mkdir(exist_ok=True)
        out = self._spawn(led, tmp_path, monkeypatch, "write a report",
                          {"writable": [str(work)]})
        assert out.startswith("spawned"), out
        tid = out.split()[2]
        rec = led.get(tid)
        assert "reviewer" not in rec.contract.constraints
        assert len(rec.contract.acceptance) == 1

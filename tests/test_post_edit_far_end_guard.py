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

"""Tests for PostEditFarEndGuard — the every-successful-edit far-end reminder."""

import os

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.post_edit_far_end import PostEditFarEndGuard


def _ctx(tool_name="edit_file", path="a.py", result="Successfully edited a.py"):
    return GuardContext(
        tool_name=tool_name,
        tool_args={"path": path},
        tool_result=result,
    )


@pytest.fixture
def guard():
    return PostEditFarEndGuard()


class TestFiresOnSuccessOnly:
    def test_py_edit_success_fires_with_py_compile_hint(self, guard):
        v = guard.check_post(
            _ctx(path="flagscale_agent/react/guard/unit_test.py",
                 result="Successfully edited flagscale_agent/react/guard/unit_test.py")
        )
        assert v is not None
        assert v.action == "inject"
        assert "py_compile" in v.message
        assert "/reload" in v.message  # agent-source reload branch
        assert "flagscale_agent/" in v.message

    def test_yaml_write_success_fires_with_parse_hint_no_reload(self, guard):
        v = guard.check_post(
            _ctx(tool_name="write_file", path="cfg/exp.yaml",
                 result="Wrote 100 chars to cfg/exp.yaml (total file size: 100 bytes)")
        )
        assert v is not None and v.action == "inject"
        assert "yaml.safe_load" in v.message
        assert "/reload" not in v.message  # not agent source

    def test_sh_edit_success_fires_with_bash_n(self, guard):
        v = guard.check_post(
            _ctx(path="scripts/run.sh", result="Successfully edited scripts/run.sh")
        )
        assert v is not None and "bash -n" in v.message

    def test_unknown_ext_falls_back_to_re_read(self, guard):
        v = guard.check_post(_ctx(path="README.md", result="Successfully edited README.md"))
        assert v is not None and "re-read" in v.message

    def test_uppercase_extension_matched(self, guard):
        v = guard.check_post(_ctx(path="DATA.JSON", result="Successfully edited DATA.JSON"))
        assert v is not None and "json.load" in v.message


class TestSilentCases:
    @pytest.mark.parametrize("result", [
        "ERROR: old_string not found in a.py",
        "ERROR: Cannot write to protected system path: /etc/hosts",
        "Error executing tool: boom",
        "",
        None,
    ])
    def test_error_or_empty_results_stay_silent(self, guard, result):
        assert guard.check_post(_ctx(result=result)) is None

    def test_non_write_tool_stays_silent(self, guard):
        assert guard.check_post(_ctx(tool_name="shell", path="x.py")) is None
        assert guard.check_post(_ctx(tool_name="shell")) is None

    def test_missing_path_stays_silent(self, guard):
        ctx = GuardContext(tool_name="write_file", tool_args={},
                           tool_result="Wrote 5 chars to x (total file size: 5 bytes)")
        assert guard.check_post(ctx) is None

    def test_pre_check_never_blocks(self, guard):
        # Inject-only contract: check_pre returns None regardless of input.
        assert guard.check_pre(_ctx()) is None
        assert guard.check_pre(GuardContext(tool_name="write_file",
                                            tool_args={"path": "/etc/passwd"},
                                            tool_result="x")) is None


class TestCategoryIndependence:
    def test_category_is_post_edit_far_end(self, guard):
        v = guard.check_post(_ctx())
        assert v is not None
        assert v.category == "post_edit_far_end"
        # Not colliding with the other post-edit guard's category
        assert v.category != "unit_test_reminder"

    def test_fires_every_edit_no_latch(self, guard):
        # User decision: fire on EVERY successful edit — no per-path latch,
        # no suppression across repeats of the same file.
        for _ in range(3):
            v = guard.check_post(_ctx(path="same.py"))
            assert v is not None, "must re-fire on every successful edit"

    def test_reset_turn_is_safe_noop(self, guard):
        guard.reset_turn()
        assert guard.check_post(_ctx()) is not None


class TestRegistryWiring:
    def test_registered_in_agent_kernel(self):
        # The agent's kernel must wire the guard so it actually runs in prod.
        from flagscale_agent.react import agent as agent_mod
        import inspect
        src = inspect.getsource(agent_mod)
        assert "PostEditFarEndGuard" in src
        # both import and registration present
        assert "from flagscale_agent.react.guard.post_edit_far_end import" in src
        assert "guard_registry.register(PostEditFarEndGuard())" in src

    def test_registry_dedup_keeps_distinct_categories(self):
        from flagscale_agent.react.guard import GuardRegistry
        from flagscale_agent.react.guard.unit_test import UnitTestGuard
        reg = GuardRegistry()
        reg.register(PostEditFarEndGuard())
        reg.register(UnitTestGuard())
        # Both guards survive registration
        names = [g.name for g in reg.guards]
        assert "post_edit_far_end" in names and "unit_test_reminder" in names
        v = reg.check_post(_ctx(path="flagscale_agent/react/guard/unit_test.py",
                                result="Successfully edited unit_test.py"))
        # merged inject from both guards (each has its own category); the
        # unit-test reminder needs >=2 sources so only the far-end one fires here
        assert v is not None
        assert "FAR end" in v.message


class TestProcessBoundaryHint:
    """A .py that launches processes / sets env must nudge toward a REAL E2E."""

    def _write(self, tmp_path, name, body):
        p = tmp_path / name
        p.write_text(body, encoding="utf-8")
        return str(p)

    def test_subprocess_source_gets_boundary_nudge(self, guard, tmp_path):
        p = self._write(tmp_path, "spawner.py",
                        "import subprocess\nsubprocess.Popen(['ls'])\n")
        v = guard.check_post(_ctx(path=p, result=f"Successfully edited {p}"))
        assert v is not None
        assert "PROCESS BOUNDARY" in v.message
        assert "REAL" in v.message and "subprocess E2E" in v.message

    def test_os_environ_source_gets_boundary_nudge(self, guard, tmp_path):
        p = self._write(tmp_path, "envset.py", "import os\nos.environ['X'] = '1'\n")
        v = guard.check_post(_ctx(path=p, result=f"Successfully edited {p}"))
        assert v is not None and "PROCESS BOUNDARY" in v.message

    def test_plain_source_has_no_boundary_nudge(self, guard, tmp_path):
        p = self._write(tmp_path, "plain.py", "def add(a, b):\n    return a + b\n")
        v = guard.check_post(_ctx(path=p, result=f"Successfully edited {p}"))
        assert v is not None and "PROCESS BOUNDARY" not in v.message

    def test_non_py_never_reads_for_boundary(self, guard, tmp_path):
        p = self._write(tmp_path, "note.md", "subprocess.Popen and os.environ\n")
        v = guard.check_post(_ctx(path=p, result=f"Successfully edited {p}"))
        assert v is not None and "PROCESS BOUNDARY" not in v.message

    def test_missing_file_is_silent_not_crash(self, guard):
        # idempotent path that does not exist → no boundary hint, no exception
        v = guard.check_post(_ctx(path="/no/such/dir/x.py",
                                  result="Successfully edited /no/such/dir/x.py"))
        assert v is not None and "PROCESS BOUNDARY" not in v.message



class TestFormContractNudge:
    """Form/contract drift nudge: HOW + WHY + GAIN in the FORM line."""

    def test_form_contract_line_present_with_anchor_phrases(self, guard):
        v = guard.check_post(_ctx(path="cfg/exp.yaml",
                                  result="Wrote 10 chars to cfg/exp.yaml"))
        assert v is not None and v.action == "inject"
        assert "FORM contract" in v.message
        # HOW: verbatim rule list + check against written bytes
        assert "VERBATIM" in v.message
        assert "written bytes" in v.message
        # WHY: silent failure with functional green
        assert "SILENTLY" in v.message
        assert "paraphrase" in v.message
        # GAIN: catch at write time
        assert "write time" in v.message

    def test_existing_far_end_anchors_preserved(self, guard):
        v = guard.check_post(_ctx(path="scripts/run.sh",
                                  result="Successfully edited scripts/run.sh"))
        assert v is not None
        assert "FAR end" in v.message
        assert "valid-for-type" in v.message
        assert "will the consumer actually read it at this exact path?" in v.message

    def test_cold_consumer_probe_present(self, guard):
        """Before done, become a stranger who just received the artifact: cat the
        ACTUAL product file (not your own narration) and confirm it is really
        there and really in the required format."""
        v = guard.check_post(_ctx(path="cfg/exp.yaml",
                                  result="Wrote 10 chars to cfg/exp.yaml"))
        assert v is not None
        assert "COLD-CONSUMER" in v.message
        assert "cat" in v.message

    def test_side_effect_sweep_present(self, guard):
        """Before delivering, run git status/diff and READ the list for any
        unintended change — a scratch/byproduct or a file that should not have
        been touched — and revert it."""
        v = guard.check_post(_ctx(path="cfg/exp.yaml",
                                  result="Wrote 10 chars to cfg/exp.yaml"))
        assert v is not None
        assert "SIDE-EFFECT sweep" in v.message
        assert "git status --short" in v.message


class TestMeasuredChecks:
    """Measurement upgrade: the guard RUNS the cheap validity check itself and
    reports the measured PASS/FAIL in the inject (user-approved design).

    These tests hit REAL subprocesses (no mocking) — the boundary under test
    is the measurement itself.
    """

    def _prompt_py(self, tmp_path, body):
        d = tmp_path / "flagscale_agent" / "react"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "prompt.py"
        p.write_text(body, encoding="utf-8")
        return str(p)

    def test_stray_brace_prompt_reports_render_failure(self, guard, tmp_path):
        # py_compile passes (the string literal is valid Python) but the
        # runtime .format explodes with KeyError — the exact class the
        # render probe exists to catch.
        p = self._prompt_py(
            tmp_path,
            'SYSTEM_PROMPT_STATIC = "hdr {cwd}\\ntrailing {"\n'
            'DASHBOARD_TEMPLATE = "x {dashboard_content}"\n',
        )
        v = guard.check_post(_ctx(path=p, result=f"Successfully edited {p}"))
        assert v is not None
        assert "RENDER FAILED" in v.message
        assert "stray" in v.message  # names the fix: the brace pair

    def test_missing_prompt_constant_reports_render_failure(self, guard, tmp_path):
        # F1 regression (reviewer-confirmed): a removed/renamed prompt constant
        # is PROVABLE runtime breakage (prompt_builder imports both names at
        # startup) — it must be REPORTED, not swallowed as an env error.
        d = tmp_path / "flagscale_agent" / "react"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "prompt.py"
        p.write_text('SYSTEM_PROMPT_STATIC = "hdr {cwd}"\n', encoding="utf-8")
        v = guard.check_post(_ctx(path=str(p), result=f"Successfully edited {p}"))
        assert v is not None
        assert "RENDER FAILED" in v.message
        assert "constant missing" in v.message

    def test_module_import_env_error_stays_silent(self, guard, tmp_path):
        # The HONEST uncheckable case: the edited file's own imports cannot be
        # satisfied in the probe env (ModuleNotFoundError) — the probe cannot
        # check the file here, which is not evidence it is broken: stay silent.
        p = self._prompt_py(
            tmp_path,
            'import module_that_does_not_exist_xyz\n'
            'SYSTEM_PROMPT_STATIC = "x"\nDASHBOARD_TEMPLATE = "y"\n',
        )
        v = guard.check_post(_ctx(path=p, result=f"Successfully edited {p}"))
        assert v is not None
        assert "RENDER FAILED" not in v.message
        assert "measured now: PASS" in v.message

    def test_foreign_prompt_path_not_probed(self, guard, tmp_path):
        # F3 regression: the anchor must sit at a path-separator boundary —
        # /x/notflagscale_agent/react/prompt.py is NOT a repo prompt file and
        # must not trigger the render probe (no RENDER FAILED even with a
        # brace that would break a prompt template).
        d = tmp_path / "x" / "notflagscale_agent" / "react"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "prompt.py"
        p.write_text('SYSTEM_PROMPT_STATIC = "x {"\n', encoding="utf-8")
        v = guard.check_post(_ctx(path=str(p), result=f"Successfully edited {p}"))
        assert v is not None
        assert "RENDER FAILED" not in v.message
        assert "measured now: PASS" in v.message  # py_compile truthfully passes

    def test_is_prompt_file_boundary_anchored(self):
        from flagscale_agent.react.guard.post_edit_far_end import (
            PostEditFarEndGuard as G,
        )

        assert G._is_prompt_file("flagscale_agent/react/prompt.py")
        assert G._is_prompt_file("/repo/flagscale_agent/react/prompt.py")
        assert G._is_prompt_file("/repo/flagscale_agent/react/prompt_builder.py")
        # boundary: foreign tree whose name merely ENDS with the anchor
        assert not G._is_prompt_file("/x/notflagscale_agent/react/prompt.py")
        assert not G._is_prompt_file("flagscale_agent/react/sub_prompt.py")

    def test_live_prompt_py_measures_pass(self, guard):
        import flagscale_agent.react.prompt as prompt_mod

        v = guard.check_post(_ctx(path=str(prompt_mod.__file__),
                                  result="Successfully edited prompt.py"))
        assert v is not None
        assert "measured now: PASS" in v.message

    def test_syntax_error_py_reports_failed_with_stderr_tail(self, guard, tmp_path):
        p = tmp_path / "broken.py"
        p.write_text("def f(:\n    pass\n", encoding="utf-8")
        v = guard.check_post(_ctx(path=str(p), result=f"Successfully edited {p}"))
        assert v is not None
        assert "measured now: FAILED" in v.message
        assert "py_compile" in v.message  # the far-end hint line survives

    def test_valid_json_reports_pass(self, guard, tmp_path):
        p = tmp_path / "good.json"
        p.write_text('{"a": 1}', encoding="utf-8")
        v = guard.check_post(_ctx(path=str(p), result=f"Wrote 8 chars to {p}"))
        assert v is not None and "measured now: PASS" in v.message

    def test_broken_json_reports_failed(self, guard, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json", encoding="utf-8")
        v = guard.check_post(_ctx(path=str(p), result=f"Wrote 9 chars to {p}"))
        assert v is not None and "measured now: FAILED" in v.message

    def test_no_check_type_has_no_measurement_line(self, guard):
        v = guard.check_post(_ctx(path="README.md",
                                  result="Successfully edited README.md"))
        assert v is not None and "measured now" not in v.message

    def test_missing_file_reports_failed_not_crash(self, guard):
        # A nonexistent path is a MEASURED failure (the check ran and failed),
        # never an exception.
        v = guard.check_post(_ctx(path="/no/such/dir/x.py",
                                  result="Successfully edited /no/such/dir/x.py"))
        assert v is not None and "measured now: FAILED" in v.message

    def test_subprocess_timeout_falls_back_to_plain_reminder(self, guard, tmp_path,
                                                             monkeypatch):
        # Real (non-mocked) timeout: shrink the budget so py_compile cannot
        # finish -> _measure returns None -> the inject stays the plain
        # reminder with NO measurement line. Guard must not raise.
        import flagscale_agent.react.guard.post_edit_far_end as pefe

        monkeypatch.setattr(pefe, "_RUN_TIMEOUT_SECONDS", 0.001)
        p = tmp_path / "slow.py"
        p.write_text("V = 1\n", encoding="utf-8")
        v = guard.check_post(_ctx(path=str(p), result=f"Successfully edited {p}"))
        assert v is not None
        assert "measured now" not in v.message
        assert "valid-for-type" in v.message  # reminder intact

    def test_relative_path_same_subtree_prompt_probe_clean(self, guard):
        # The EXACT conditions that exposed the commonpath-root bug live: a
        # RELATIVE path whose tree shares the guard's subtree — commonpath
        # derived flagscale_agent/react as "repo root", cwd switched there,
        # and the double-prefixed path made the probe FileNotFoundError,
        # misreported as a stray-brace RENDER FAILED (a false positive on
        # a healthy repo: 2477 tests green + runtime fine). Root must come
        # from the guard's own fixed address instead. "Pre-fix" evidence is
        # the live pre-mortem observation that triggered this fix.
        import flagscale_agent.react.guard.post_edit_far_end as pefe

        root = os.path.abspath(pefe.__file__)
        for _ in range(4):
            root = os.path.dirname(root)
        rel = os.path.relpath(
            os.path.join(root, "flagscale_agent/react/prompt.py"), os.getcwd()
        )
        assert not os.path.isabs(rel)  # the bug-triggering condition, confirmed
        r = guard._render_probe(rel)
        assert r is None  # healthy file: render OK, no false RENDER FAILED


class TestCitationDrift:
    """prop_60077fda: an edit that shifts a file's line count must warn that any
    held `file:line` anchor to that file may now be stale (the recorded failure:
    a doc body cited :1102/:620 that later edits moved to :1126/:623)."""

    def _git_file(self, tmp_path):
        """A file committed to a throwaway git repo, so HEAD:<path> resolves."""
        import subprocess
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
        f = repo / "doc.md"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        subprocess.run(["git", "add", "doc.md"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
        return f

    def test_drift_note_on_line_shift(self, guard, tmp_path):
        f = self._git_file(tmp_path)
        # add two lines -> delta +2
        f.write_text("line1\nline2\nline3\nnew4\nnew5\n", encoding="utf-8")
        note = guard._citation_drift_note(_ctx(path=str(f)), str(f))
        assert note is not None
        assert "CITATION DRIFT" in note
        assert "+2 line" in note
        assert "re-grep" in note

    def test_no_note_when_no_line_change(self, guard, tmp_path):
        f = self._git_file(tmp_path)
        # rewrite without changing the line count -> delta 0 -> silent
        f.write_text("LINE1\nLINE2\nLINE3\n", encoding="utf-8")
        assert guard._citation_drift_note(_ctx(path=str(f)), str(f)) is None

    def test_silent_on_non_anchor_file_types(self, guard, tmp_path):
        import subprocess
        repo = tmp_path / "r2"
        repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        binfile = repo / "img.png"
        binfile.write_bytes(b"x")
        subprocess.run(["git", "add", "img.png"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "b"], cwd=repo, check=True)
        binfile.write_bytes(b"xx\nyy\n")
        assert guard._citation_drift_note(_ctx(path=str(binfile)), str(binfile)) is None

    def test_silent_without_git_baseline(self, guard, tmp_path):
        f = tmp_path / "untracked.md"
        f.write_text("a\nb\n", encoding="utf-8")
        assert guard._citation_drift_note(_ctx(path=str(f)), str(f)) is None

    def test_message_includes_drift(self, guard, tmp_path):
        f = self._git_file(tmp_path)
        f.write_text("line1\nline2\nline3\nx\ny\n", encoding="utf-8")
        note = guard._citation_drift_note(_ctx(path=str(f)), str(f))
        msg = guard._message(str(f), "measured now: PASS", note)
        assert "CITATION DRIFT" in msg
        assert "valid-for-type" in msg  # base reminder not lost

    def test_relpath_for_git_resolves(self, guard, tmp_path):
        f = self._git_file(tmp_path)
        rel = guard._relpath_for_git(str(f))
        assert rel == "doc.md"

    def test_relpath_for_git_fallback_without_repo(self, guard, tmp_path):
        f = tmp_path / "loose.md"
        f.write_text("a\n", encoding="utf-8")
        assert guard._relpath_for_git(str(f)) == "loose.md"

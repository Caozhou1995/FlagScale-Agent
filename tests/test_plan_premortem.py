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

"""Tests for the feature-keyed premortem injection in PlanGuard."""

import os
import shutil
import tempfile

import pytest

from flagscale_agent.react.guard.plan import PlanGuard, _premortem_block


@pytest.fixture
def ws():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _point_at(monkeypatch, d):
    monkeypatch.setenv("FLAGSCALE_AGENT_TASK_DIR", d)


class TestPremortemBlock:
    def test_empty_workspace_no_inject(self, monkeypatch, ws):
        _point_at(monkeypatch, ws)
        assert _premortem_block() == ""

    def test_missing_dir_degrades_to_empty(self, monkeypatch):
        _point_at(monkeypatch, "/nonexistent_premortem_dir_xyz")
        assert _premortem_block() == ""

    def test_reference_artifact_triggers_ref_note(self, monkeypatch, ws):
        open(os.path.join(ws, "expected_output.txt"), "w").write("out")
        _point_at(monkeypatch, ws)
        b = _premortem_block()
        assert "Reference artifacts" in b
        assert "diff" in b.lower()

    def test_expected_prefix_triggers_ref_note(self, monkeypatch, ws):
        open(os.path.join(ws, "expected.txt"), "w").write("out")
        _point_at(monkeypatch, ws)
        b = _premortem_block()
        assert "Reference artifacts" in b

    def test_harness_triggers_harness_note(self, monkeypatch, ws):
        open(os.path.join(ws, "verify.sh"), "w").write("#!/bin/sh\n")
        _point_at(monkeypatch, ws)
        b = _premortem_block()
        assert "USE IT as the" in b

    def test_test_prefix_triggers_harness_note(self, monkeypatch, ws):
        open(os.path.join(ws, "test_main.py"), "w").write("")
        _point_at(monkeypatch, ws)
        b = _premortem_block()
        assert "USE IT as the" in b

    def test_pyproject_counts_as_harness(self, monkeypatch, ws):
        open(os.path.join(ws, "pyproject.toml"), "w").write("")
        _point_at(monkeypatch, ws)
        b = _premortem_block()
        assert "USE IT as the" in b

    def test_both_features_two_notes(self, monkeypatch, ws):
        open(os.path.join(ws, "expected_output.txt"), "w").write("out")
        open(os.path.join(ws, "Makefile"), "w").write("")
        _point_at(monkeypatch, ws)
        b = _premortem_block()
        assert "Reference artifacts" in b
        assert "USE IT as the" in b

    def test_no_noise_for_unrelated_workspace(self, monkeypatch, ws):
        # Files that match neither feature family -> no inject.
        open(os.path.join(ws, "main.py"), "w").write("")
        open(os.path.join(ws, "data.csv"), "w").write("")
        _point_at(monkeypatch, ws)
        assert _premortem_block() == ""

    def test_injected_in_plan_create_inject(self, monkeypatch, ws):
        open(os.path.join(ws, "expected_output.txt"), "w").write("out")
        _point_at(monkeypatch, ws)
        g = PlanGuard()
        v = g.check_pre(
            type("C", (), {"tool_name": "plan_create", "tool_args": {}})()
        )
        assert v is not None and v.action == "inject"
        assert "Reference artifacts" in v.message

    def test_write_file_gate_carries_premortem(self, monkeypatch, ws):
        open(os.path.join(ws, "verify.sh"), "w").write("#!/bin/sh\n")
        _point_at(monkeypatch, ws)
        g = PlanGuard(task_plan=None, single_shot=True)
        v = g.check_pre(
            type("C", (), {
                "tool_name": "write_file",
                "tool_args": {"path": os.path.join(ws, "out.txt")},
            })()
        )
        assert v is not None and v.action == "block"
        assert "USE IT as the" in v.message

    def test_single_shot_block_has_no_duplicate_portfolio_text(self, monkeypatch, ws):
        # Regression: _WALL_AWARE_FIRST_PLAN must appear exactly once in the
        # single-shot block message (it is included via _wall_aware_block()).
        g = PlanGuard(task_plan=None, single_shot=True)
        v = g.check_pre(
            type("C", (), {
                "tool_name": "write_file",
                "tool_args": {"path": os.path.join(ws, "out.txt")},
            })()
        )
        assert v is not None
        assert v.message.count("PORTFOLIO") == 1

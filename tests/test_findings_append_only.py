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

"""prop_810728da — the audit findings file is APPEND-ONLY.

The audit report is a cross-round shared deliverable (one measured session
reached 160 KB / 909 lines, built by appending one `## Audit #N` section per
round). Two things made a whole-file rewrite likely rather than hypothetical:
the tool's own instructions say "first call with mode='write'", and the auditor
has its OWN empty `findings.md` shell in <session>/audit/reviewer/ — a path that
is one segment away.

A shrinking `mode='write'` is now refused at the tool layer (so every caller is
covered, not just the agent). Growing rewrites and appends still work: the rule
targets DESTRUCTION, not the write itself.

Note: a non-findings path keeps the pre-existing >50%-shrink guard, which emits
a different message ("would shrink"). These tests assert the two are
distinguishable, so a regression in either one is attributable.
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.tools.write_file import (
    WriteFileTool, _is_append_only_audit_report)


@pytest.fixture
def tool():
    return WriteFileTool()


@pytest.fixture
def findings(tmp_path):
    p = tmp_path / "audit" / "findings.md"
    p.parent.mkdir(parents=True)
    WriteFileTool().execute(path=str(p), content="# Audit #1\nVIOLATION: a\n")
    return p


# ── path predicate ──────────────────────────────────────────────────────────

def test_findings_md_is_recognised_at_any_depth(tmp_path):
    assert _is_append_only_audit_report("/x/audit/findings.md")
    assert _is_append_only_audit_report(str(tmp_path / "a" / "b" / "findings.md"))


def test_other_filenames_are_not_affected(tmp_path):
    assert not _is_append_only_audit_report(str(tmp_path / "notes.md"))
    assert not _is_append_only_audit_report(str(tmp_path / "findings.txt"))


# ── the rule ────────────────────────────────────────────────────────────────

def test_shrinking_overwrite_of_the_audit_report_is_refused(tool, findings):
    out = tool.execute(path=str(findings), content="short\n")
    assert out.startswith("ERROR:")
    assert "APPEND-ONLY" in out
    assert "mode='append'" in out, "the refusal must name the way forward"
    # The earlier audit must survive byte-for-byte.
    assert "# Audit #1" in findings.read_text()


def test_first_write_to_a_new_report_is_allowed(tool, tmp_path):
    # Nothing to protect yet — refusing here would block the normal creation.
    p = tmp_path / "audit2" / "findings.md"
    p.parent.mkdir(parents=True)
    out = tool.execute(path=str(p), content="# Audit #1\nVIOLATION: a\n")
    assert out.startswith("Wrote")
    assert p.read_text().startswith("# Audit #1")


def test_append_is_the_supported_way_to_add_a_round(tool, findings):
    out = tool.execute(path=str(findings), content="# Audit #2\nVIOLATION: b\n",
                       mode="append")
    assert out.startswith("Appended")
    text = findings.read_text()
    assert "# Audit #1" in text and "# Audit #2" in text


def test_growing_rewrite_passes(tool, findings):
    # The rule targets destruction, not the write itself: content that does not
    # shrink the file cannot lose an audit.
    body = "# Audit #1\nVIOLATION: a\nVIOLATION: b\nVIOLATION: c\n"
    out = tool.execute(path=str(findings), content=body)
    assert out.startswith("Wrote"), out


def test_non_audit_file_is_governed_only_by_the_shrink_guard(tool, tmp_path):
    # Distinguishability: the two refusals must not be confused, or a regression
    # in one would be attributed to the other.
    q = tmp_path / "notes.md"
    tool.execute(path=str(q), content="long content here\n")
    out = tool.execute(path=str(q), content="x\n")
    assert "would shrink" in out
    assert "APPEND-ONLY" not in out

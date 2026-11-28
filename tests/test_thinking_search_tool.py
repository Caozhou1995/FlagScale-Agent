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

"""thinking_search — retrieval over the session's on-disk reasoning log.

The log is written in FULL (no truncation, no dropped entries), so it can grow
to megabytes. These tests pin the retrieval half of that trade: an agent must be
able to locate the entry that contains a decision, and read ONE entry at full
fidelity, without reading the whole file.

They also pin compatibility with logs written by the PREVIOUS code, whose
headers read "## reasoning N (truncated)" — an existing session's history must
stay searchable, not be lost to a parser detail.
"""

import os

from flagscale_agent.react.tools.thinking_search import (
    ThinkingSearchTool,
    _split_entries,
)


def _write_log(tmp_path, body):
    sess = tmp_path / "sess"
    sess.mkdir(parents=True, exist_ok=True)
    path = sess / "thinking_log.md"
    path.write_text(body, encoding="utf-8")
    return str(sess)


def _tool(sess):
    return ThinkingSearchTool(lambda: sess)


_LOG = """# Session reasoning log (auto-written by the harness)

- entries: 3

## reasoning 1
Filler about the weather.

## reasoning 2
I looked at the file and the 5000 bytes limit is a hard constraint. Approach:
write the C file, then wc -c it.

## reasoning 3
Unrelated note about primers.
"""


class TestSnippetMode:
    def test_locates_the_entry_holding_the_decision(self, tmp_path):
        sess = _write_log(tmp_path, _LOG)
        out = _tool(sess).execute(query="5000 bytes")
        assert "1 hit(s)" in out
        assert "[reasoning 2 |" in out, "the hit must name entry 2"
        assert "wc -c" in out, "the snippet must show the decision itself"

    def test_and_semantics_requires_all_keywords(self, tmp_path):
        sess = _write_log(tmp_path, _LOG)
        # both words live in entry 2 only
        assert "reasoning 2" in _tool(sess).execute(query="5000 constraint")
        # 'primers' is entry 3, '5000' is entry 2 -> no single entry has both
        out = _tool(sess).execute(query="primers 5000")
        assert "0 hit(s)" in out

    def test_missing_keyword_reports_zero_hits_with_a_tip(self, tmp_path):
        sess = _write_log(tmp_path, _LOG)
        out = _tool(sess).execute(query="nonexistentterm")
        assert "0 hit(s)" in out
        assert "fewer keywords" in out


class TestEntryModeReturnsTheWholeEntry:
    def test_full_entry_text_is_returned_uncut(self, tmp_path):
        # The reason the file is untruncated: the decision is at the END, so a
        # head-only slice would drop exactly the line being searched for.
        filler = "z" * 9000
        body = "## reasoning 1\n" + filler + "THE_DECISION_LINE\n"
        sess = _write_log(tmp_path, body)
        out = _tool(sess).execute(query="THE_DECISION_LINE", mode="entry", entry=1)
        assert filler in out, "the entry body was cut"
        assert "THE_DECISION_LINE" in out

    def test_entry_mode_defaults_to_the_newest_match(self, tmp_path):
        body = ("## reasoning 1\nold mention of primer\n"
                "## reasoning 2\nnew mention of primer\n")
        sess = _write_log(tmp_path, body)
        out = _tool(sess).execute(query="primer", mode="entry")
        assert "new mention" in out

    def test_unknown_entry_number_is_an_explicit_error(self, tmp_path):
        sess = _write_log(tmp_path, _LOG)
        out = _tool(sess).execute(query="5000", mode="entry", entry=99)
        assert "ERROR" in out and "99" in out


class TestLegacyHeaderCompatibility:
    def test_truncated_suffix_headers_are_parsed(self, tmp_path):
        # Logs written before the truncation removal carry this header form.
        body = ("## reasoning 1 (truncated)\nfirst\n"
                "## reasoning 2 (truncated)\nsecond has DECISION\n")
        sess = _write_log(tmp_path, body)
        out = _tool(sess).execute(query="DECISION")
        assert "reasoning 2" in out and "1 hit(s)" in out

    def test_split_entries_counts_legacy_headers(self):
        body = "## reasoning 1 (truncated)\na\n## reasoning 2 (truncated)\nb\n"
        assert len(_split_entries(body)) == 2

    def test_file_without_markers_is_still_searchable(self, tmp_path):
        sess = _write_log(tmp_path, "just a hand-written note about BsaI\n")
        out = _tool(sess).execute(query="BsaI")
        assert "1 hit(s)" in out


class TestDegradesCleanly:
    def test_missing_log_reports_why(self, tmp_path):
        out = ThinkingSearchTool(lambda: str(tmp_path / "nowhere")).execute(query="x")
        assert "ERROR" in out and "post-edit reminder fires" in out

    def test_no_session_dir_reports_an_error(self):
        assert "ERROR" in ThinkingSearchTool(lambda: None).execute(query="x")

    def test_raising_dir_fn_does_not_crash(self):
        def boom():
            raise RuntimeError("gone")
        assert "ERROR" in ThinkingSearchTool(boom).execute(query="x")

    def test_empty_query_is_rejected(self, tmp_path):
        sess = _write_log(tmp_path, _LOG)
        assert "ERROR" in _tool(sess).execute(query="   ")

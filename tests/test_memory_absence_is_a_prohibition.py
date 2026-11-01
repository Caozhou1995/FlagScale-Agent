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

"""prop_8ad3a23b — an absence must be reported as a PROHIBITION.

Measured defect (audits #78/#81): an agent wrote "Observed from world:
memory_read returned …" and quoted a body for a key whose read had actually
returned "No memory found" — twice in the same session. The bare miss message
reads as "look elsewhere", so the agent filled the gap with a plausible body.

Same shape in memory_list: the header said "Showing 15/378" and the body simply
stopped, so an unlisted key was read as an absent one.

Both now carry an explicit prohibition on the missing thing. These tests pin the
PROHIBITION (the operative part), not the exact sentence, so rewording is free.
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.memory import Memory
from flagscale_agent.react.tools.memory_list import MemoryListTool
from flagscale_agent.react.tools.memory_read import MemoryReadTool


@pytest.fixture
def mem(tmp_path):
    m = Memory(str(tmp_path / "mem"))
    m.put("fact/demo/alpha", "fact", "value: alpha")
    m.put("fact/demo/beta", "fact", "value: beta")
    return m


# ── memory_read: a miss must forbid citation ────────────────────────────────

def test_miss_forbids_citing_the_key(mem):
    out = MemoryReadTool(mem).execute(key="fact/demo/nope")
    assert "No memory found" in out
    assert "Do NOT cite this key" in out, (
        "a bare 'No memory found' invites the agent to invent a body — the miss "
        "must explicitly forbid attributing content to the key")
    assert "invented" in out


def test_miss_offers_the_nearest_existing_keys(mem):
    # A typo'd key is the common cause; pointing at the neighbours turns a dead
    # end into a one-step fix.
    out = MemoryReadTool(mem).execute(key="fact/demo/alpah")
    assert "fact/demo/alpha" in out
    assert "fact/demo/beta" in out


def test_hit_output_unchanged_by_the_miss_wording(mem):
    out = MemoryReadTool(mem).execute(key="fact/demo/alpha")
    assert out.startswith("[fact] fact/demo/alpha")
    assert "Do NOT cite" not in out


def test_prefix_miss_is_also_explicit(mem):
    out = MemoryReadTool(mem).execute(key="nothing/here/")
    assert "No entries found with prefix" in out


def test_suggestion_helper_never_raises_on_a_broken_backend():
    class Boom:
        def get(self, _):
            return None

        def list_by_prefix(self, _):
            raise RuntimeError("backend down")

    tool = MemoryReadTool(Boom())
    assert tool._similar_keys("fact/demo/x") == []
    # The miss message must still be produced.
    assert "Do NOT cite this key" in tool.execute(key="fact/demo/x")


# ── memory_list: a truncation must forbid citing the remainder ──────────────

def test_truncated_list_says_the_remainder_is_not_evidence(mem):
    out = MemoryListTool(mem).execute(limit=1)
    assert "Showing 1/2" in out
    assert "TRUNCATED at limit=1" in out
    assert "must NOT be cited" in out
    assert "not evidence of its absence" in out


def test_untruncated_list_carries_no_warning(mem):
    # A complete listing must stay clean — otherwise the warning loses meaning.
    out = MemoryListTool(mem).execute(limit=50)
    assert "Showing 2/2" in out
    assert "TRUNCATED" not in out


def test_truncation_still_shows_the_entries_it_did_show(mem):
    out = MemoryListTool(mem).execute(limit=1)
    assert "fact/demo/" in out

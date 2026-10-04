# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for memory_write supersedes two-phase confirm (proposal eb7a59cb).

supersedes is DESTRUCTIVE: the first call must return a BLOCKED preview of
what each old key holds; only a re-call with confirm=true executes the
deletion. The success receipt must echo each retired key's first line so a
wrong supersede is visible the moment it happens.
"""

import pytest

from flagscale_agent.react.memory import Memory
from flagscale_agent.react.paths import get_memory_dir
from flagscale_agent.react.tools.memory_write import MemoryWriteTool


OLD_ALPHA = "fact/old/alpha"
OLD_BETA = "fact/old/beta"


@pytest.fixture
def tool(tmp_path, monkeypatch):
    monkeypatch.setenv("FLAGSCALE_HOME", str(tmp_path))
    mem = Memory(get_memory_dir())
    mem.put(OLD_ALPHA, "fact", "value: A1\napplies: old entry one")
    mem.put(OLD_BETA, "fact", "value: B2")
    return MemoryWriteTool(mem), mem


class TestSupersedesPrecheck:
    def test_without_confirm_returns_blocked_preview(self, tool):
        t, mem = tool
        r = t.execute(
            key="fact/new/one", type="fact", content="value: NEW",
            supersedes=[OLD_ALPHA],
        )
        assert r.startswith("BLOCKED")
        assert "confirm=true" in r
        assert "A1" in r  # preview shows what the old key holds

    def test_blocked_call_deletes_nothing(self, tool):
        t, mem = tool
        t.execute(key="fact/new/one", type="fact", content="value: NEW",
                  supersedes=[OLD_ALPHA])
        assert mem.get(OLD_ALPHA) is not None
        assert mem.get(OLD_BETA) is not None
        assert mem.get("fact/new/one") is None  # nothing written either

    def test_confirm_executes_deletion_with_receipt(self, tool):
        t, mem = tool
        r = t.execute(
            key="fact/new/one", type="fact", content="value: NEW",
            supersedes=[OLD_ALPHA, OLD_BETA], confirm=True,
        )
        assert r.startswith("Memorized")
        assert mem.get(OLD_ALPHA) is None
        assert mem.get(OLD_BETA) is None
        assert mem.get("fact/new/one") is not None
        assert "A1" in r and "B2" in r  # receipt echoes the retired content

    def test_no_supersedes_unaffected_by_confirm_gate(self, tool):
        t, mem = tool
        r = t.execute(key="fact/new/two", type="fact", content="value: T2")
        assert r.startswith("Memorized")
        assert "BLOCKED" not in r

    def test_supersede_missing_key_with_confirm_still_writes(self, tool):
        t, mem = tool
        r = t.execute(
            key="fact/new/three", type="fact", content="value: T3",
            supersedes=["fact/old/ghost"], confirm=True,
        )
        assert r.startswith("Memorized")
        assert mem.get("fact/new/three") is not None

    def test_parameters_expose_confirm_and_destructive_supersedes(self, tool):
        t, _ = tool
        props = t.parameters["properties"]
        assert "confirm" in props
        assert props["confirm"]["type"] == "boolean"
        assert "DESTRUCTIVE" in props["supersedes"]["description"]

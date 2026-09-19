# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the cross-session improvement-proposal registry."""

import os
import shutil
import tempfile

import pytest

from flagscale_agent.react.proposals import (
    ProposalRegistry,
    VALID_STATUSES,
    OPEN_STATUSES,
    TERMINAL_STATUSES,
)


@pytest.fixture
def reg():
    d = tempfile.mkdtemp()
    yield ProposalRegistry(d)
    shutil.rmtree(d, ignore_errors=True)


class TestAdd:
    def test_add_defaults_to_proposed(self, reg):
        e = reg.add("Add X guard", container="agent-code", session_id="s1",
                    topic="x")
        assert e["status"] == "proposed"
        assert e["id"].startswith("prop_")
        assert e["description"] == "Add X guard"
        assert e["created_session"] == "s1"
        assert e["status_history"][0]["status"] == "proposed"

    def test_add_persists_and_reloads(self, reg):
        e = reg.add("Add Y", session_id="s1")
        # A fresh registry instance over the same dir sees it (cross-session).
        reg2 = ProposalRegistry(reg._dir)
        assert reg2.get(e["id"])["description"] == "Add Y"

    def test_add_invalid_status(self, reg):
        with pytest.raises(ValueError):
            reg.add("bad", status="nonsense")


class TestStatus:
    def test_set_status_appends_history(self, reg):
        e = reg.add("Add Z", session_id="s1")
        reg.set_status(e["id"], "approved", session_id="s2", note="ok")
        got = reg.get(e["id"])
        assert got["status"] == "approved"
        assert [h["status"] for h in got["status_history"]] == ["proposed", "approved"]

    def test_set_status_unknown_id(self, reg):
        with pytest.raises(ValueError):
            reg.set_status("prop_deadbeef", "done")

    def test_set_status_invalid_status(self, reg):
        e = reg.add("A", session_id="s1")
        with pytest.raises(ValueError):
            reg.set_status(e["id"], "bogus")

    def test_invalid_id_rejected_for_get(self, reg):
        # Path-traversal-ish ids must not resolve.
        assert reg.get("../../etc/passwd") is None
        assert reg.get("not_a_prop_id") is None


class TestListing:
    def test_open_vs_terminal(self, reg):
        a = reg.add("A", session_id="s1")
        b = reg.add("B", session_id="s1")
        c = reg.add("C", session_id="s1")
        reg.set_status(b["id"], "done")
        reg.set_status(c["id"], "rejected")
        open_ids = {e["id"] for e in reg.list_open()}
        assert open_ids == {a["id"]}
        assert {e["id"] for e in reg.list_status("done")} == {b["id"]}

    def test_open_statuses_constant(self):
        assert set(OPEN_STATUSES).isdisjoint(TERMINAL_STATUSES)
        assert set(OPEN_STATUSES) | set(TERMINAL_STATUSES) == set(VALID_STATUSES)

    def test_render_open_empty(self, reg):
        assert reg.render_open() == ""

    def test_render_open_lists_open_only(self, reg):
        a = reg.add("A", container="skill", session_id="s1")
        reg.add("B", session_id="s1")  # no container
        text = reg.render_open()
        assert a["id"] in text
        assert "skills" in text or "[skill]" in text
        assert text.startswith("Open improvement proposals")

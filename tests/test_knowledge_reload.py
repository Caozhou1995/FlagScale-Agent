"""The knowledge manager must see a group added AFTER the manager was built.

A manager lives for the whole process (Agent.__init__ constructs it once), while
knowledge groups arrive later — a git pull, a deployment sync. If the config is
only read in __init__, the new group stays invisible until a restart and the
load_knowledge tool answers "Unknown group" for a group that is already on disk.

These tests build a throwaway knowledge root so they never touch the real tree.
"""
import os
import time
import pathlib

import pytest

from flagscale_agent.knowledge import KnowledgeManager


def _root(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "indexes").mkdir()
    (tmp_path / "knowledge_config.yaml").write_text(
        "know-alpha:\n"
        '  description: "first"\n'
        "  docs:\n"
        "    - alpha/01.md\n",
        encoding="utf-8",
    )
    return tmp_path


def _add_group(root: pathlib.Path, name: str = "know-beta") -> None:
    cfg = root / "knowledge_config.yaml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8")
        + f"{name}:\n"
        '  description: "second"\n'
        "  docs:\n"
        f"    - {name.split('-', 1)[1]}/01.md\n",
        encoding="utf-8",
    )
    (root / "indexes" / f"{name}.idx").write_text(f"# {name}\n", encoding="utf-8")
    # A later mtime than the initial load, so the staleness check trips even on
    # filesystems with coarse timestamp resolution.
    future = time.time() + 5
    os.utime(cfg, (future, future))


def test_new_group_is_visible_on_the_same_instance(tmp_path):
    root = _root(tmp_path)
    km = KnowledgeManager(str(root))
    assert km.available_groups == ["know-alpha"]

    _add_group(root)

    assert "know-beta" in km.available_groups, (
        "a group added after construction must be visible without a process restart"
    )
    assert km.get_index("know-beta") == "# know-beta\n"
    assert km.get_group_docs("know-beta") == ["beta/01.md"]
    assert km.load_knowledge("know-beta") is not None


def test_removed_group_disappears_on_the_same_instance(tmp_path):
    root = _root(tmp_path)
    _add_group(root)
    km = KnowledgeManager(str(root))
    assert "know-beta" in km.available_groups

    # Rewrite the config without know-beta -> the stale entry must not survive.
    cfg = root / "knowledge_config.yaml"
    cfg.write_text(
        "know-alpha:\n"
        '  description: "first"\n'
        "  docs:\n"
        "    - alpha/01.md\n",
        encoding="utf-8",
    )
    future = time.time() + 10
    os.utime(cfg, (future, future))

    assert km.available_groups == ["know-alpha"]
    assert km.load_knowledge("know-beta") is None


def test_unchanged_config_is_not_an_error(tmp_path):
    """The steady state must keep working: no reload, no group loss."""
    root = _root(tmp_path)
    km = KnowledgeManager(str(root))
    for _ in range(3):
        assert km.available_groups == ["know-alpha"]
        assert km.get_index("know-alpha") is None  # index file absent by design
        assert km.get_group_docs("know-alpha") == ["alpha/01.md"]


def test_missing_config_is_tolerated(tmp_path):
    """A root with no config must not raise on lookup."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "indexes").mkdir()
    km = KnowledgeManager(str(tmp_path))
    assert km.available_groups == []
    assert km.load_knowledge("know-alpha") is None

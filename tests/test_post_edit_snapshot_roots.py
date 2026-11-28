"""Regression pins for the effect-branch snapshot roots.

Measured failure (2026-11-28): a shell call issued from `/` made os.getcwd() ==
"/", so _snapshot_roots walked the WHOLE root filesystem. The effect branch then
reported /run/containerd/io.containerd.runtime.v2.task/.../config.json (11666
bytes) as a changed deliverable — pure runtime noise, and the newly added size
line amplified it into the agent's context (17 such lines in one inject).

The fix refuses system roots outright. Both halves are asserted:
  (a) NEGATIVE: a system root yields nothing (this is what was broken)
  (b) POSITIVE: a real task workspace is still walked — without this half, a fix
      that simply disabled the walk entirely would look green.
"""
import os

from flagscale_agent.react.guard.post_edit_far_end import (
    _is_workspace_root,
    _snapshot_roots,
)


def test_system_roots_are_refused():
    for root in ("/", "/run", "/proc", "/sys", "/usr", "/etc", "/boot",
                 "/lib", "/bin", "/sbin"):
        assert _is_workspace_root(root) is False, f"{root} must never be walked"


def test_runtime_subtrees_are_refused_at_any_depth():
    # containerd's per-container state lives deep under /run; the depth must not
    # make it look like a workspace.
    for root in (
        "/run/containerd/io.containerd.runtime.v2.task/moby/abc123",
        "/proc/self",
        "/sys/fs/cgroup",
        "/dev/shm",
    ):
        assert _is_workspace_root(root) is False, f"{root} must never be walked"


def test_real_workspaces_are_allowed():
    for root in ("/app", "/tmp", "/root", "/nfs/caozhou/glm/jobs",
                 "/nfs/caozhou/fa_eval/FlagScale-Agent"):
        assert _is_workspace_root(root) is True, f"{root} is a workspace"


def test_cwd_root_produces_no_snapshot_entries(monkeypatch):
    """The exact bug: cwd == '/' must snapshot zero files, not the whole disk."""
    monkeypatch.chdir("/")
    snap = _snapshot_roots()
    assert snap == {}, f"snapshot from '/' must be empty, got {len(snap)} entries"
    for prefix in ("/run/", "/proc/", "/sys/", "/usr/", "/etc/"):
        leaked = [p for p in snap if p.startswith(prefix)]
        assert not leaked, f"runtime/vendor paths leaked into snapshot: {leaked[:3]}"


def test_cwd_workspace_still_snapshots_its_files(monkeypatch, tmp_path):
    """POSITIVE half: the fix must not disable the walk where it is needed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deliverable.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "note.md").write_text("# not a deliverable ext\n", encoding="utf-8")
    snap = _snapshot_roots()
    assert any(p.endswith("deliverable.py") for p in snap), snap
    assert not any(p.endswith("note.md") for p in snap), snap


def test_explicit_extra_root_still_works_from_a_refused_cwd(monkeypatch, tmp_path):
    """A shell run from '/' onto an explicit /tmp target must still be seen."""
    monkeypatch.chdir("/")
    (tmp_path / "out.c").write_text("int main(){return 0;}\n", encoding="utf-8")
    snap = _snapshot_roots(extra_roots=[str(tmp_path)])
    assert any(p.endswith("out.c") for p in snap), snap

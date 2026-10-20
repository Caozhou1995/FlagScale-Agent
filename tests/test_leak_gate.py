"""CI gate: no graded-task identity may reach the agent.

This wires analysis/leak_gate.py into the pytest suite so the four leak channels
are actually checked in CI. It complements the existing per-phrase tests (which
only cover the phrases someone happened to remember) by checking the whole
channel: static prompt, all agent-facing runtime text, the staging/mount tree,
and every launcher.

Channels (each one independently leaked during 2026-10-20):
  (a) static prompt carries a task token
  (b) staging tree carries graded-task material (tests/, analysis/, tokens)
  (c) agent-facing runtime text carries task tokens or "<n> <unit>" thresholds
  (d) a launcher bind-mounts the repo root, making tests/ readable in-container

Host-dependent checks skip (not fail) when the launcher dir or staging tree is
absent, so the gate is usable on a clean checkout while staying strict here.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GATE_PY = REPO / "analysis" / "leak_gate.py"
LAUNCH_DIR = pathlib.Path("/nfs/caozhou/glm")
STAGE = pathlib.Path("/nfs/caozhou/glm/agent_stage2")


def _gate():
    if not GATE_PY.is_file():
        pytest.skip(f"leak_gate.py absent at {GATE_PY}")
    spec = importlib.util.spec_from_file_location("_leak_gate", GATE_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_leak_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_static_prompt_clean():
    assert not _gate().check_prompt(REPO)


def test_agent_facing_text_clean():
    assert not _gate().check_agent_facing(REPO)


def test_launchers_do_not_mount_repo_root():
    if not LAUNCH_DIR.is_dir():
        pytest.skip("launcher dir absent")
    assert not _gate().check_mounts(LAUNCH_DIR)


def test_staging_tree_clean():
    if not STAGE.is_dir():
        pytest.skip("staging tree absent")
    assert not _gate().check_stage(STAGE)


# --- the gate must FAIL on injected leaks, or it is decoration -------------

def _repo_with_prompt(tmp_path, extra: str) -> pathlib.Path:
    """A minimal repo whose prompt.py is the real one plus `extra`."""
    repo = tmp_path / "repo"
    (repo / "flagscale_agent/react/guard").mkdir(parents=True)
    src = (REPO / "flagscale_agent/react/prompt.py").read_text()
    (repo / "flagscale_agent/react/prompt.py").write_text(src + extra)
    for rel in ("oracle.py", "judge.py", "guard/verification.py"):
        real = REPO / "flagscale_agent/react" / rel
        (repo / "flagscale_agent/react" / rel).write_text(
            real.read_text() if real.is_file() else "")
    return repo


def test_gate_detects_task_token_in_prompt(tmp_path):
    gate = _gate()
    assert gate.check_prompt(_repo_with_prompt(tmp_path, "\n# use mteb\n")), \
        "gate missed a task token in the prompt"


def test_gate_detects_threshold_in_agent_facing_text(tmp_path):
    gate = _gate()
    repo = _repo_with_prompt(tmp_path, "")
    judge = repo / "flagscale_agent/react/judge.py"
    judge.write_text(judge.read_text() + '\n_LIMIT = "bound 15-45 nt"\n')
    assert gate.check_agent_facing(repo), \
        "gate missed a threshold in agent-facing text"


def test_gate_detects_forbidden_dir_in_staging(tmp_path):
    gate = _gate()
    stage = tmp_path / "stage"
    (stage / "flagscale_agent").mkdir(parents=True)
    (stage / "pyproject.toml").write_text("x")
    (stage / "README.md").write_text("x")
    assert not gate.check_stage(stage)          # clean baseline is clean
    (stage / "tests").mkdir()
    assert gate.check_stage(stage), "gate missed tests/ inside staging"


def test_gate_detects_repo_root_mount(tmp_path):
    gate = _gate()
    ld = tmp_path / "launch"
    ld.mkdir()
    (ld / "ok.sh").write_text('MOUNTS=\'[{"source":"/nfs/caozhou/glm/agent_stage2"}]\'\n')
    assert not gate.check_mounts(ld)            # staging mount is accepted
    (ld / "bad.sh").write_text(
        'MOUNTS=\'[{"source":"/nfs/caozhou/fa_eval/FlagScale-Agent"}]\'\n')
    assert gate.check_mounts(ld), "gate missed a repo-root mount"

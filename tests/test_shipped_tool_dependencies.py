"""The in-package tools the prompt points at must be RUNNABLE, not merely importable.

Measured motivation (2026-10-20, live m50 trial container): the prompt told the
agent to run

    /opt/fsagent-venv/bin/python3 -c "import flagscale_agent.tools.cardinality_probe ..."

and that import succeeded (PROBE_OK), but executing the probe died with
`ModuleNotFoundError: No module named 'numpy'`. The harness installs the agent
into a dedicated venv with `python3 -m venv` + `pip install <staged tree>`, so
the venv contains ONLY what the project metadata declares. numpy was not a
declared dependency (it appears only in an optional extra and in lazy imports),
so the tool was un-importable-in-practice for the one interpreter the agent is
told to use.

A test that only checked the prompt *text* would have passed while the tool was
broken. These tests bind the tool modules' real imports to the declared
dependency set.
"""

from __future__ import annotations

import ast
import sys
import pathlib
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "flagscale_agent" / "tools"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _declared_dependencies() -> set[str]:
    data = tomllib.loads(PYPROJECT.read_text())
    raw = data["project"]["dependencies"]
    names = set()
    for spec in raw:
        name = spec
        for sep in ("[", ">", "<", "=", "!", "~", ";"):
            name = name.split(sep)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def _top_level_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
    return found


def test_the_two_shipped_diagnostics_exist():
    names = {p.name for p in TOOLS_DIR.glob("*.py")}
    assert "cardinality_probe.py" in names
    assert "assembled_deliverable_replay.py" in names


def test_third_party_imports_of_the_tools_are_declared_dependencies():
    """Every non-stdlib top-level import of a shipped tool must be declared.

    This is what makes the prompt's locator command actually work: the venv the
    harness builds installs only the declared dependency set.
    """
    declared = _declared_dependencies()
    # The authoritative list, so an incomplete hand-written whitelist cannot make
    # this test either falsely red (stdlib flagged) or falsely green.
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    missing: dict[str, set[str]] = {}
    for tool in ("cardinality_probe.py", "assembled_deliverable_replay.py"):
        path = TOOLS_DIR / tool
        assert path.is_file(), f"missing shipped tool: {tool}"
        for module in _top_level_imports(path):
            if module in stdlib or module == "flagscale_agent":
                continue
            if module.lower().replace("_", "-") not in declared:
                missing.setdefault(tool, set()).add(module)
    assert not missing, (
        "tools import modules that the harness venv will not install: "
        f"{missing} (declared: {sorted(declared)})"
    )


def test_numpy_is_a_hard_dependency_not_an_extra():
    """numpy must be in [project].dependencies, not optional-dependencies.

    The agent's interpreter IS the venv built from these dependencies; an extra
    is not installed, which is exactly the failure this file documents.
    """
    data = tomllib.loads(PYPROJECT.read_text())
    hard = {d.split(">")[0].split("=")[0].strip().lower()
            for d in data["project"]["dependencies"]}
    assert "numpy" in hard
    optional = data["project"].get("optional-dependencies", {}) or {}
    for group in optional.values():
        for dep in group:
            assert not dep.lower().startswith("numpy"), (
                "numpy must not be optional: the agent's venv only installs "
                "[project].dependencies"
            )

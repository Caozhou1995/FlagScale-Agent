"""The in-package probes must be locatable by an interpreter that actually has them.

Measured failure this locks (m48 containers): the system `python3` on PATH raises
ModuleNotFoundError for `flagscale_agent`, because the adapter pip-installs the
package into `/opt/fsagent-venv`. A pointer written as `python3 -c "import ..."`
is therefore unexecutable text inside the task container.
"""
import re
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

PROMPT = SYSTEM_PROMPT_STATIC


def test_both_probes_are_named():
    assert "flagscale_agent.tools.cardinality_probe" in PROMPT
    assert "flagscale_agent.tools.assembled_deliverable_replay" in PROMPT


def test_locator_uses_the_venv_interpreter_not_bare_python3():
    # every `-c "import flagscale_agent...` locator must be prefixed by the venv path
    for m in re.finditer(r'`([^`]*)-c "import [^`]*flagscale_agent[^`]*`', PROMPT):
        assert m.group(1).strip().endswith("/opt/fsagent-venv/bin/python3"), m.group(0)
    assert PROMPT.count("/opt/fsagent-venv/bin/python3") >= 2


def test_prompt_forbids_trusting_bare_python3():
    assert "the bare `python3` on PATH does NOT have the package" in PROMPT


def test_no_bare_python3_locator_for_the_probes():
    assert not re.search(r'locate (it|either one) with `python3 -c', PROMPT)


def test_locator_never_cwd_relative():
    assert "never a path relative to your task's working directory" in PROMPT

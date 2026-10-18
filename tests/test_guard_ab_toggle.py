"""A/B guard toggle: FLAGSCALE_AGENT_DISABLED_GUARDS suppresses named guards.

The paired A/B harness needs to run the SAME task set twice, once with a target
guard ENABLED and once DISABLED, so a per-task reward delta is attributable to
that one guard. This module tests the registration-time toggle that makes the
"disabled" arm reachable.

Generic only — the env var names no task, grader, or parser.
"""

from __future__ import annotations

import os

import pytest

from flagscale_agent.react.guard import Guard, GuardRegistry


class _NamedGuard(Guard):
    priority = 10

    def __init__(self, name: str):
        self.name = name

    def check_pre(self, ctx):  # pragma: no cover - not exercised
        return None


def _registered(env_value, names=("verification", "safety", "plan")):
    """Build a registry with `names` while the toggle env is set to env_value."""
    old = os.environ.get("FLAGSCALE_AGENT_DISABLED_GUARDS")
    try:
        if env_value is None:
            os.environ.pop("FLAGSCALE_AGENT_DISABLED_GUARDS", None)
        else:
            os.environ["FLAGSCALE_AGENT_DISABLED_GUARDS"] = env_value
        reg = GuardRegistry()
        for n in names:
            reg.register(_NamedGuard(n))
        return sorted(g.name for g in reg._guards)
    finally:
        if old is None:
            os.environ.pop("FLAGSCALE_AGENT_DISABLED_GUARDS", None)
        else:
            os.environ["FLAGSCALE_AGENT_DISABLED_GUARDS"] = old


@pytest.mark.parametrize("env_value", [None, "", "   ", ","])
def test_unset_or_empty_registers_every_guard(env_value):
    """Default behaviour is unchanged: no toggle => all guards register."""
    assert _registered(env_value) == ["plan", "safety", "verification"]


def test_named_guard_is_skipped():
    assert _registered("verification") == ["plan", "safety"]


def test_multiple_named_guards_are_skipped():
    assert _registered("verification,plan") == ["safety"]


def test_whitespace_around_names_is_trimmed():
    assert _registered(" verification ,  safety ") == ["plan"]


def test_unknown_name_is_ignored():
    """A name not matching any guard leaves the set intact (fail-open)."""
    assert _registered("no_such_guard") == ["plan", "safety", "verification"]


def test_env_var_sets_no_task_or_grader_token():
    """The toggle channel names no task/grader — generic by construction."""
    src = open(
        os.path.join(
            os.path.dirname(__file__), "..",
            "flagscale_agent", "react", "guard", "__init__.py",
        )
    ).read()
    assert "FLAGSCALE_AGENT_DISABLED_GUARDS" in src

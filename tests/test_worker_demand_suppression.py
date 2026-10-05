# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for reviewer/diverger demand self-reference suppression
(proposals 6824c6c3 + d23da412).

The reviewer demand ("spawn ONE read-only reviewer per passing step_done")
and the diverger demand ("propose 2-3 genuinely different framings") are
PARENT-side habits. A spawned reviewer/diverger worker must NOT receive the
nested demand: it would self-reference the demand chain and contend for the
shared concurrency gate. Detection runs off the contract file spawned into
the worker's env (FLAGSCALE_CONTRACT_PATH).

conftest.py clears the worker-identity env per test; the worker branch is
exercised explicitly via monkeypatch.setenv (house pattern).
"""

import json

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.plan import PlanGuard
from flagscale_agent.react.guard.verification import VerificationGuard

REVIEWER_CONTRACT = (
    "# Task\n\ngoal: independently review the diff\n\n"
    "## Reviewer discipline\n- read-only\n"
)
DIVERGER_CONTRACT = (
    "# Task\n\ngoal: Propose 2-3 genuinely different framings of the task "
    "— ideas only\n"
)
NORMAL_CONTRACT = "# Task\n\ngoal: implement the 8 proposals\n"

REVIEWER_DEMAND_MARKER = "spawn one NOW"
DIVERGER_DEMAND_MARKER = "Propose 2-3 genuinely different framings"


def _worker_env(monkeypatch, contract_text, tmp_path, reviewer=None):
    """Mimic a spawned worker env: rendered prompt + structured contract.json.

    Production suppression reads the STRUCTURED constraints from the sibling
    contract.json (written by the ledger at spawn time, L179) — never the
    rendered prose. The fixture must write both files with the same shape.
    """
    contract = tmp_path / "contract.md"
    contract.write_text(contract_text)
    constraints = {}
    if reviewer is not None:
        constraints["reviewer"] = reviewer
    (tmp_path / "contract.json").write_text(
        json.dumps({"constraints": constraints})
    )
    monkeypatch.setenv("FLAGSCALE_TASK_ID", "t-suppress")
    monkeypatch.setenv("FLAGSCALE_CONTRACT_PATH", str(contract))


def _step_done_ctx():
    return GuardContext(
        tool_name="plan_update",
        tool_args={"action": "step_done", "step_id": 1,
                   "_override_reason": "ran tests, all pass"},
        override_reason="ran tests, all pass",
    )


class TestReviewerDemandSuppression:
    def test_reviewer_worker_gets_no_nested_demand(self, tmp_path, monkeypatch):
        """A reviewer worker's passing step_done carries the pre-mortem only."""
        _worker_env(monkeypatch, REVIEWER_CONTRACT, tmp_path, reviewer=True)
        guard = VerificationGuard()
        guard._step_done_recheck_reminded = True  # let the step_done pass
        assert guard.check_pre(_step_done_ctx()) is None
        v = guard.check_post(_step_done_ctx())
        assert v is not None and v.action == "inject"
        assert v.reason == "step_done_premortem"
        assert REVIEWER_DEMAND_MARKER not in v.message

    def test_parent_still_gets_the_demand(self):
        """Without worker env, the reviewer demand fires as before."""
        guard = VerificationGuard()
        guard._step_done_recheck_reminded = True
        assert guard.check_pre(_step_done_ctx()) is None
        v = guard.check_post(_step_done_ctx())
        assert v is not None
        assert REVIEWER_DEMAND_MARKER in v.message


class TestDivergerDemandSuppression:
    def test_diverger_worker_gets_no_nested_demand(self, tmp_path, monkeypatch):
        """A diverger worker's plan_create inject omits the framing demand."""
        _worker_env(monkeypatch, DIVERGER_CONTRACT, tmp_path, reviewer=True)
        guard = PlanGuard()
        v = guard.check_pre(GuardContext(tool_name="plan_create",
                                         tool_args={"title": "x",
                                                    "steps": ["s"]}))
        assert v is not None and v.action == "inject"
        assert DIVERGER_DEMAND_MARKER not in v.message

    def test_normal_worker_still_gets_the_demand(self, tmp_path, monkeypatch):
        """A non-diverger worker contract keeps the demand (marker absent)."""
        _worker_env(monkeypatch, NORMAL_CONTRACT, tmp_path)
        guard = PlanGuard()
        v = guard.check_pre(GuardContext(tool_name="plan_create",
                                         tool_args={"title": "x",
                                                    "steps": ["s"]}))
        assert v is not None
        assert DIVERGER_DEMAND_MARKER in v.message

    def test_parent_still_gets_the_demand(self):
        """Without worker env, the diverger demand fires as before."""
        guard = PlanGuard()
        v = guard.check_pre(GuardContext(tool_name="plan_create",
                                         tool_args={"title": "x",
                                                    "steps": ["s"]}))
        assert v is not None
        assert DIVERGER_DEMAND_MARKER in v.message

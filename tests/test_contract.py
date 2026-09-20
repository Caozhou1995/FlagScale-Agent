# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0

"""Tests for the multi-agent Contract (design §1.1/§1.3)."""

import os
import time

import pytest

from flagscale_agent.react.multi_agent.contract import (
    Contract,
    ContractError,
    compute_id,
    GOAL_MAX_LEN,
)


def _mk(tmp_path, **over):
    """Build a valid contract rooted under tmp_path."""
    writable = str(tmp_path)
    out = os.path.join(writable, "out.json")
    kw = dict(
        goal="Write a JSON report of the repo file count",
        constraints={"writable": [writable], "forbidden": ["no network"],
                     "max_minutes": 10},
        acceptance=[{"check": "test -f out.json", "kind": "check_command"}],
        output_ptr=out,
        inputs=[],
    )
    kw.update(over)
    return Contract.build(**kw)


class TestContentAddressing:
    def test_id_is_deterministic(self):
        c1 = compute_id("goal A", {"w": ["/tmp"]}, [{"check": "true"}])
        c2 = compute_id("goal A", {"w": ["/tmp"]}, [{"check": "true"}])
        assert c1 == c2
        assert len(c1) == 12

    def test_id_changes_with_goal(self):
        a = compute_id("goal A", {}, [{"check": "true"}])
        b = compute_id("goal B", {}, [{"check": "true"}])
        assert a != b

    def test_goal_only_whitespace_rejected(self, tmp_path):
        with pytest.raises(ContractError):
            _mk(tmp_path, goal="   ").validate()

    def test_goal_too_long_rejected(self, tmp_path):
        with pytest.raises(ContractError):
            _mk(tmp_path, goal="x" * (GOAL_MAX_LEN + 1)).validate()


class TestValidate:
    def test_valid_contract_passes(self, tmp_path):
        _mk(tmp_path).validate()  # no raise

    def test_empty_acceptance_rejected(self, tmp_path):
        with pytest.raises(ContractError):
            _mk(tmp_path, acceptance=[]).validate()

    def test_acceptance_missing_check_rejected(self, tmp_path):
        with pytest.raises(ContractError):
            _mk(tmp_path, acceptance=[{"desc": "no check key"}]).validate()

    def test_output_ptr_must_be_absolute(self, tmp_path):
        with pytest.raises(ContractError):
            _mk(tmp_path, output_ptr="relative/out.json").validate()

    def test_output_ptr_outside_writable_rejected_inv4(self, tmp_path):
        # INV4: output_ptr must live inside constraints.writable
        with pytest.raises(ContractError):
            _mk(tmp_path, output_ptr="/etc/evil.json").validate()

    def test_input_path_must_exist(self, tmp_path):
        c = _mk(tmp_path, inputs=[
            {"kind": "path", "value": str(tmp_path / "nope.txt")}
        ])
        with pytest.raises(ContractError):
            c.validate(check_inputs_exist=True)
        # but not required when the worker side validates (remote input)
        c.validate(check_inputs_exist=False)

    def test_input_relative_path_rejected(self, tmp_path):
        with pytest.raises(ContractError):
            _mk(tmp_path, inputs=[{"kind": "path", "value": "rel.txt"}]).validate(
                check_inputs_exist=False)

    def test_deadline_in_past_rejected(self, tmp_path):
        with pytest.raises(ContractError):
            _mk(tmp_path, deadline_epoch=int(time.time()) - 10).validate()

    def test_depth_zero_rejected(self, tmp_path):
        with pytest.raises(ContractError):
            _mk(tmp_path, depth=0).validate()

    def test_tampered_id_rejected(self, tmp_path):
        c = _mk(tmp_path)
        # rebuild with a wrong carried id
        bad = Contract(
            id="deadbeefdead", goal=c.goal, constraints=c.constraints,
            acceptance=c.acceptance, inputs=c.inputs, output_ptr=c.output_ptr,
            deadline_epoch=c.deadline_epoch, depth=c.depth, parent=c.parent,
        )
        with pytest.raises(ContractError):
            bad.validate()


class TestWire:
    def test_round_trip_preserves_fields(self, tmp_path):
        c = _mk(tmp_path, inputs=[])
        wire = c.to_wire()
        assert wire["task_id"] == c.id
        assert wire["goal"] == c.goal
        assert wire["deadline_sec"] > 0
        back = Contract.from_wire(wire)
        assert back.id == c.id
        assert back.goal == c.goal
        assert back.output_ptr == c.output_ptr
        assert back.acceptance == c.acceptance

    def test_wire_without_carriers_best_effort(self, tmp_path):
        c = _mk(tmp_path)
        wire = c.to_wire()
        # strip non-normative carriers to simulate a foreign-language peer
        for k in ("_inputs", "_acceptance_all", "_constraints_full", "_parent"):
            wire.pop(k, None)
        back = Contract.from_wire(wire)
        assert back.id == c.id
        assert back.output_ptr == c.output_ptr

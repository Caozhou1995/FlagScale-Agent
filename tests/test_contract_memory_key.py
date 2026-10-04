# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Tests for Contract kind=="memory_key" existence precheck (proposal 8aa20979).

A contract input {kind: "memory_key", value: "fact/dom/name"} promises the
worker a memory entry. The parent-side pre-spawn check (validate with
check_inputs_exist=True) must fail closed when the key does not exist —
with candidate keys from the same domain — and must be skipped entirely on
the worker side (check_inputs_exist=False).
"""

import time

import pytest

from flagscale_agent.react.memory import Memory
from flagscale_agent.react.multi_agent.contract import (
    Contract,
    ContractError,
    compute_id,
)
from flagscale_agent.react.paths import get_memory_dir


EXISTING_KEY = "fact/demo/real_key"
DOMAIN = "fact/demo"


@pytest.fixture
def seeded_memory(tmp_path, monkeypatch):
    """Isolated FLAGSCALE_HOME with one seeded memory entry."""
    monkeypatch.setenv("FLAGSCALE_HOME", str(tmp_path))
    Memory(get_memory_dir()).put(
        EXISTING_KEY, "fact", "value: demo-entry\nverify cmd: none"
    )
    return tmp_path


def _contract(memory_input=None):
    inputs = []
    if memory_input is not None:
        kind, value = memory_input
        inputs.append({"kind": kind, "value": value})
    goal = "exercise the memory_key precheck"
    constraints = {"writable": ["/tmp/fsa_test_out"]}
    acceptance = [{"check": "echo ok"}]
    return Contract(
        id=compute_id(goal, constraints, acceptance),
        goal=goal,
        constraints=constraints,
        acceptance=acceptance,
        inputs=inputs,
        output_ptr="/tmp/fsa_test_out/result.md",
        deadline_epoch=int(time.time()) + 600,
        depth=1,
    )


class TestMemoryKeyExists:
    def test_existing_key_passes(self, seeded_memory):
        """A seeded, existing memory_key passes the parent-side check."""
        c = _contract(("memory_key", EXISTING_KEY))
        c.validate(check_inputs_exist=True)  # must not raise

    def test_missing_key_raises_with_candidates(self, seeded_memory):
        """A missing key raises ContractError naming same-domain candidates."""
        c = _contract(("memory_key", "fact/demo/does_not_exist"))
        with pytest.raises(ContractError) as exc:
            c.validate(check_inputs_exist=True)
        msg = str(exc.value)
        assert "memory_key does not exist" in msg
        assert EXISTING_KEY in msg  # the candidate list points at the fix

    def test_worker_side_skips_check(self, seeded_memory):
        """check_inputs_exist=False (worker re-validate) skips the precheck."""
        c = _contract(("memory_key", "fact/demo/does_not_exist"))
        c.validate(check_inputs_exist=False)  # must not raise

    def test_empty_value_rejected_unconditionally(self, seeded_memory):
        """A blank memory_key value is rejected even without existence check."""
        c = _contract(("memory_key", "   "))
        with pytest.raises(ContractError) as exc:
            c.validate(check_inputs_exist=False)
        assert "non-empty string" in str(exc.value)

    def test_path_inputs_unaffected(self, seeded_memory):
        """kind=="path" keeps the original absolute-path/existence semantics."""
        c = _contract(("path", "/definitely/not/a/real/path"))
        with pytest.raises(ContractError) as exc:
            c.validate(check_inputs_exist=True)
        assert "path does not exist" in str(exc.value)

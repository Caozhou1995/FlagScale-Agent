# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task contract — the single shared substrate of the multi-agent design.

Every layer (single-machine spawn, cross-machine summon, market, swarm) is a
consumer of this one data structure: a "game" is a contract plus a set of
players. Freezing the contract (its Python form for in-process use + its JSON
wire form for cross-machine use) makes every upper layer a fill-in-the-blank.

Two representations of the SAME contract:
  - Python dataclass `Contract` (§2 single-machine layer, in-process)
  - JSON wire `TaskContract` (contract.to_wire() / Contract.from_wire(),
    §3+ cross-machine layer)

The id is CONTENT-ADDRESSED: sha256(goal|constraints|acceptance)[:12]. This is
what makes idempotent acceptance (INV6) work — the same task text always maps
to the same id, so a duplicate delivery is detectable without a registry lookup.

Invariants enforced here (see design doc §0.3):
  - INV4: output_ptr must live inside constraints["writable"]
  - INV7: the contract is self-contained (no host-private jargon references)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# goal length cap (design §1.1: "<=200 chars, single sentence, decidable")
GOAL_MAX_LEN = 200


class ContractError(ValueError):
    """Raised when a contract fails validation. Fail-closed: no partial accept."""


def _is_abs(p: str) -> bool:
    return isinstance(p, str) and p.startswith("/") and os.path.isabs(p)


def compute_id(goal: str, constraints: Dict[str, Any], acceptance: List[Dict[str, Any]]) -> str:
    """Content-addressed id: sha256(goal|constraints_json|acceptance_json)[:12].

    Canonical form (sorted keys, no whitespace drift) so the SAME logical
    contract always yields the SAME id across processes and machines — the
    precondition for idempotent acceptance (INV6).
    """
    payload = "|".join([
        goal or "",
        json.dumps(constraints or {}, sort_keys=True, ensure_ascii=False),
        json.dumps(acceptance or [], sort_keys=True, ensure_ascii=False),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Contract:
    """An immutable task contract (design §1.1).

    Frozen because a contract is a promise: once handed to a worker it must not
    mutate under it. A changed scope is a NEW contract (new id), not an edit.
    """

    id: str
    goal: str
    constraints: Dict[str, Any]
    acceptance: List[Dict[str, Any]]
    inputs: List[Dict[str, Any]]
    output_ptr: str
    deadline_epoch: int
    depth: int
    parent: Dict[str, Any] = field(default_factory=dict)

    # ── validation ───────────────────────────────────────────────────────────
    def validate(self, check_inputs_exist: bool = True) -> None:
        """Validate per design §1.1, in order. Raises ContractError on any fail.

        Args:
            check_inputs_exist: when True (parent-side pre-spawn check) every
                `inputs[*].value` path must currently exist on disk. The worker
                side re-validates the CONTRACT but not filesystem state (the
                path may be remote / fetched lazily).
        """
        # 1. goal non-empty, <= 200 chars; acceptance non-empty, each check non-empty
        if not self.goal or not self.goal.strip():
            raise ContractError("goal must be non-empty")
        if len(self.goal) > GOAL_MAX_LEN:
            raise ContractError(
                f"goal exceeds {GOAL_MAX_LEN} chars (got {len(self.goal)})"
            )
        if not self.acceptance:
            raise ContractError("acceptance must be non-empty")
        for i, a in enumerate(self.acceptance):
            if not isinstance(a, dict) or not (a.get("check") or "").strip():
                raise ContractError(f"acceptance[{i}].check must be non-empty")

        # 2. inputs/output_ptr are absolute paths; inputs exist (parent-side)
        if not _is_abs(self.output_ptr):
            raise ContractError(f"output_ptr must be an absolute path: {self.output_ptr!r}")
        for i, inp in enumerate(self.inputs):
            if not isinstance(inp, dict):
                raise ContractError(f"inputs[{i}] must be a dict")
            val = inp.get("value", "")
            if inp.get("kind") == "path" and not _is_abs(val):
                raise ContractError(f"inputs[{i}].value must be an absolute path: {val!r}")
            if check_inputs_exist and inp.get("kind") == "path" and not os.path.exists(val):
                raise ContractError(f"inputs[{i}] path does not exist: {val!r}")

        # 3. output_ptr ∈ writable (INV4)
        writable = (self.constraints or {}).get("writable") or []
        if not any(_is_abs(w) and _within(self.output_ptr, w) for w in writable):
            raise ContractError(
                f"output_ptr {self.output_ptr!r} not inside any constraints.writable {writable!r} (INV4)"
            )

        # 4. deadline in the future; depth >= 1
        if self.deadline_epoch <= int(time.time()):
            raise ContractError(
                f"deadline_epoch {self.deadline_epoch} is not in the future"
            )
        if self.depth < 1:
            raise ContractError(f"depth must be >= 1 (got {self.depth})")

        # 5. recomputed id matches the carried id (tamper detection)
        expect = compute_id(self.goal, self.constraints, self.acceptance)
        if self.id != expect:
            raise ContractError(
                f"id mismatch: carried {self.id!r} != recomputed {expect!r} (tampered?)"
            )

    # ── wire (cross-machine, design §1.3) ────────────────────────────────────
    def to_wire(self) -> Dict[str, Any]:
        """Serialize to the language-neutral JSON wire form (§1.3).

        Note the wire uses `deadline_sec` (a duration) while the dataclass uses
        `deadline_epoch` (an instant) — the wire carries a relative deadline so
        a remote worker's clock skew cannot instantly expire the task.
        """
        now = int(time.time())
        remaining = max(1, self.deadline_epoch - now)
        # acceptance on the wire is a single object; local form is a list of
        # checks. Carry the first check as the canonical spec, keep the full
        # list under a non-normative key for lossless round-trip.
        acc = self.acceptance[0] if self.acceptance else {}
        return {
            "task_id": self.id,
            "goal": self.goal,
            "constraints": self.constraints.get("forbidden", []),
            "acceptance": {
                "kind": acc.get("kind", "check_command"),
                "spec": acc.get("check", ""),
                "cwd": acc.get("cwd", ""),
            },
            "output_spec": {"path_or_url": self.output_ptr, "format": "json"},
            "deadline_sec": remaining,
            "depth": self.depth,
            "parent_trace": self.parent.get("parent_trace", []),
            # non-normative round-trip carriers
            "_inputs": self.inputs,
            "_acceptance_all": self.acceptance,
            "_constraints_full": self.constraints,
            "_parent": self.parent,
        }

    @classmethod
    def from_wire(cls, wire: Dict[str, Any]) -> "Contract":
        """Reconstruct a Contract from the wire form.

        Uses the non-normative carriers when present (lossless); otherwise
        rebuilds a best-effort contract from the normative wire fields.
        """
        if "_constraints_full" in wire:
            constraints = wire["_constraints_full"]
            acceptance = wire.get("_acceptance_all") or []
            inputs = wire.get("_inputs") or []
            output_ptr = wire["output_spec"]["path_or_url"]
            deadline_epoch = int(time.time()) + int(wire.get("deadline_sec", 60))
            parent = wire.get("_parent") or {}
        else:
            # Best-effort from normative fields only.
            constraints = {"writable": [], "forbidden": wire.get("constraints", [])}
            acc = wire.get("acceptance", {})
            acceptance = [{"check": acc.get("spec", ""), "kind": acc.get("kind", "check_command")}]
            inputs = []
            output_ptr = wire["output_spec"]["path_or_url"]
            deadline_epoch = int(time.time()) + int(wire.get("deadline_sec", 60))
            parent = {"parent_trace": wire.get("parent_trace", [])}
        return cls(
            id=wire["task_id"],
            goal=wire["goal"],
            constraints=constraints,
            acceptance=acceptance,
            inputs=inputs,
            output_ptr=output_ptr,
            deadline_epoch=deadline_epoch,
            depth=wire.get("depth", 1),
            parent=parent,
        )

    # ── construction helper ──────────────────────────────────────────────────
    @classmethod
    def build(
        cls,
        goal: str,
        constraints: Dict[str, Any],
        acceptance: List[Dict[str, Any]],
        output_ptr: str,
        inputs: Optional[List[Dict[str, Any]]] = None,
        deadline_epoch: Optional[int] = None,
        depth: int = 1,
        parent: Optional[Dict[str, Any]] = None,
    ) -> "Contract":
        """Build a Contract with a computed content-addressed id."""
        constraints = constraints or {}
        acceptance = acceptance or []
        inputs = inputs or []
        if deadline_epoch is None:
            max_min = (constraints or {}).get("max_minutes", 60)
            deadline_epoch = int(time.time()) + int(max_min) * 60
        cid = compute_id(goal, constraints, acceptance)
        return cls(
            id=cid,
            goal=goal,
            constraints=constraints,
            acceptance=acceptance,
            inputs=inputs,
            output_ptr=output_ptr,
            deadline_epoch=deadline_epoch,
            depth=depth,
            parent=parent or {},
        )


def _within(path: str, root: str) -> bool:
    """True if `path` is `root` itself or a descendant of it (no .. escape)."""
    path = os.path.normpath(path)
    root = os.path.normpath(root)
    return path == root or path.startswith(root.rstrip("/") + "/")

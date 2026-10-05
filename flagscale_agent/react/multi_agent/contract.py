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
  - Python dataclass `Contract` (single-machine layer, in-process)
  - JSON wire `TaskContract` (contract.to_wire() / Contract.from_wire(),
    cross-machine layer)

The id is CONTENT-ADDRESSED: sha256(goal|constraints|acceptance)[:12]. This is
what makes idempotent acceptance work — the same task text always maps
to the same id, so a duplicate delivery is detectable without a registry lookup.

Invariants enforced here:
  - output_ptr must live inside constraints["writable"]
  - the contract is self-contained (no host-private jargon references)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# goal length cap ("<=200 chars, single sentence, decidable")
GOAL_MAX_LEN = 200

# ── RoleSpec — one mechanism for the reviewer role and everything it implies ──
# A task's ROLE is derived from EXPLICIT constraint tags only (never from
# prose-substring detection): `constraints.role` when present, else the
# boolean `constraints.reviewer` flag. The role then governs the reviewer
# preset (spawn side), the live-tree pointer check (parent side) and the
# write-time report-completeness gate (worker side).
ROLE_REVIEWER = "reviewer"
ROLE_RE_REVIEWER = "re_reviewer"
_ROLE_TAGS = (ROLE_REVIEWER, ROLE_RE_REVIEWER)

# reviewer preset: a review needs room to read the artifact and finalize.
REVIEWER_MIN_MINUTES = 10


def resolve_role(constraints: Optional[Dict[str, Any]]) -> Optional[str]:
    """The task role from EXPLICIT tags, or None for a normal task.

    Precedence: `role` (explicit role tag) wins over the boolean `reviewer`
    flag — `{"role": "re_reviewer", "reviewer": False}` is a re-review, not a
    suppressed review. A present-but-false `reviewer` means "normal task" and
    is the ONLY suppression signal (never prose).
    """
    if not isinstance(constraints, dict):
        return None
    role = constraints.get("role")
    if isinstance(role, str) and role in _ROLE_TAGS:
        return role
    if constraints.get("reviewer") is True:
        return ROLE_REVIEWER
    return None


def is_reviewer_role(constraints: Optional[Dict[str, Any]]) -> bool:
    """True for any reviewer-class role (reviewer or re-reviewer)."""
    return resolve_role(constraints) is not None


@dataclass(frozen=True)
class RoleSpec:
    """What a role implies, in one place, consumed by all three gates."""

    role: Optional[str] = None
    report_gate: bool = False        # write-time completeness gate
    live_tree_check: bool = False    # verify path inputs are the LIVE tree
    min_minutes: int = 0             # preset deadline floor

    @property
    def is_review(self) -> bool:
        return self.role is not None


def role_spec(constraints: Optional[Dict[str, Any]]) -> RoleSpec:
    """Derive the RoleSpec for a contract's constraints."""
    role = resolve_role(constraints)
    if role is None:
        return RoleSpec()
    return RoleSpec(role=role, report_gate=True, live_tree_check=True,
                    min_minutes=REVIEWER_MIN_MINUTES)


# ── report completeness (shared by the write gate and the acceptance template) ─
# Line-anchored: 'no findings' must be (the start of) its own line, so a
# report that merely ECHOES the gate's guidance ("...or state 'no findings',
# then finalize") does not satisfy the check. Optional markdown prefixes:
# bullet (-, *, +), ordered list (1.), bold (**).
_NO_FINDINGS_RE = re.compile(
    r"^[ \t]*(?:[-*+]|\d+\.)?[ \t]*\**no findings\b",
    re.IGNORECASE | re.MULTILINE)


def report_has_findings(text: str) -> bool:
    """True when the report carries a per-finding markdown table DATA row.

    A markdown table block is consecutive `|`-led rows (>=2 pipes). When the
    second row is a delimiter row (cells of dashes/colons only), only rows
    AFTER it count as data — a HEADER `| ID | Sev | Fix |` plus separator
    with zero data rows is a skeleton, not a finding. Within a data row a
    cell must carry content besides spaces/dashes/colons.
    """

    def _cells(row: str) -> List[str]:
        return [c.strip() for c in row.strip("|").split("|")]

    def _is_delim(row: str) -> bool:
        cells = _cells(row)
        return bool(cells) and all(re.fullmatch(r"[-: ]*", c or "")
                                   for c in cells)

    def _has_content(row: str) -> bool:
        return any(c and not re.fullmatch(r"[-: ]*", c) for c in _cells(row))

    lines = (text or "").splitlines()
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not (s.startswith("|") and s.count("|") >= 2):
            i += 1
            continue
        j = i
        block = []
        while j < n:
            t = lines[j].strip()
            if t.startswith("|") and t.count("|") >= 2:
                block.append(t)
                j += 1
            else:
                break
        data_rows = block[2:] if len(block) >= 2 and _is_delim(block[1]) \
            else block
        if any(_has_content(r) for r in data_rows):
            return True
        i = j
    return False


def has_no_findings_line(text: str) -> bool:
    """True when the report explicitly states there are no findings."""
    return _NO_FINDINGS_RE.search(text or "") is not None


def has_dangling_in_progress(text: str) -> bool:
    """True when a report carries a dangling '(IN PROGRESS' header/placeholder.

    A header or placeholder line (starts with '#' or '(') that still says
    IN PROGRESS means the report was reported before it was finalized.
    A mid-sentence mention is NOT a dangling header.
    """
    for line in (text or "").splitlines():
        if "(IN PROGRESS" not in line:
            continue
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith("("):
            return True
    return False


def check_report_completeness(path: str) -> Optional[str]:
    """Return None if the reviewer report at `path` is complete, else guidance.

    Complete = exists, non-empty, no dangling '(IN PROGRESS' header, and
    either >=1 finding (a per-finding table row) or an explicit 'no findings'
    line.
    """
    if not path or not os.path.exists(path):
        return (f"reviewer report {path!r} does not exist — write the report "
                "to output_ptr, then finalize and call report_result again.")
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError as e:
        return f"reviewer report {path!r} could not be read: {e}"
    if not text.strip():
        return (f"reviewer report {path!r} is empty — finalize it before "
                "reporting.")
    if has_dangling_in_progress(text):
        return (f"reviewer report {path!r} still carries a dangling "
                "'(IN PROGRESS' header — finish that section before reporting.")
    if not (report_has_findings(text) or has_no_findings_line(text)):
        return (f"reviewer report {path!r} contains no finding and no explicit "
                "'no findings' line — add a per-finding table (one `| ... |` "
                "row per finding) or state 'no findings', then finalize.")
    return None


class ContractError(ValueError):
    """Raised when a contract fails validation. Fail-closed: no partial accept."""


def _is_abs(p: str) -> bool:
    return isinstance(p, str) and p.startswith("/") and os.path.isabs(p)


def live_tree_check_enabled() -> bool:
    """The live-tree check can be disabled by env (escape hatch for ports /
    non-importable layouts). Default: enabled."""
    return os.environ.get("FLAGSCALE_LIVE_TREE_CHECK", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _package_root_for(path: str) -> Optional[str]:
    """The topmost package dir containing `path`, or None if not a package.

    Walks up while each level carries `__init__.py`, so both a package dir and
    a module file inside one resolve to the package root.
    """
    cur = path if os.path.isdir(path) else os.path.dirname(path)
    root: Optional[str] = None
    while cur and os.path.isfile(os.path.join(cur, "__init__.py")):
        root = cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return root


def check_live_tree(input_path: str) -> Optional[str]:
    """If `input_path` names an importable python package, verify it is the
    location the CURRENT process actually imports. Returns an error message
    when it is a stale duplicate, else None.

    Why: a review that points at a stale copy (a different checkout of the same
    package) reads the wrong file, so its first diff pass targets code the
    running system never executes. Comparing the input's package root against
    the resolved module location (importlib) turns that silent wrong-file
    review into a pre-spawn refusal.
    """
    if not live_tree_check_enabled():
        return None
    root = _package_root_for(input_path)
    if root is None:
        return None
    name = os.path.basename(root.rstrip("/"))
    if not name.isidentifier():
        return None
    try:
        import importlib.util
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError, AttributeError):
        return None
    if spec is None:
        return None
    import_root = None
    locs = getattr(spec, "submodule_search_locations", None)
    if locs:
        import_root = list(locs)[0]
    elif spec.origin and spec.origin not in ("built-in", "frozen"):
        import_root = os.path.dirname(spec.origin)
    if not import_root:
        return None
    if os.path.realpath(import_root) != os.path.realpath(root):
        return (
            f"input path {input_path!r} is not the LIVE tree: this process "
            f"imports package {name!r} from {import_root!r} — the input points "
            "at a stale duplicate copy. Point the review at the imported "
            "location (or set FLAGSCALE_LIVE_TREE_CHECK=0 to override)."
        )
    return None


def _check_memory_key_exists(i: int, key: str) -> None:
    """Pre-spawn existence check for kind=="memory_key" inputs.

    A contract that promises the worker a memory key must name one that
    EXISTS at freeze time — a typo'd or stale key silently hands the worker
    an empty dict and wastes its first turns rediscovering the miss. Fail
    closed here, at the parent, with the nearest candidate keys so the fix
    is one lookup away. Lazy imports keep this pure-stdlib module
    import-cycle-free (react.memory ← react.paths only).
    """
    from flagscale_agent.react.memory import Memory
    from flagscale_agent.react.paths import get_memory_dir

    mem = Memory(get_memory_dir())
    entry = mem.get(key)
    if entry is not None:
        return
    candidates = [
        e["key"] for e in mem.list_by_prefix(key.rsplit("/", 1)[0] + "/")
    ][:8]
    hint = (
        "\n  candidates under this domain: " + ", ".join(candidates)
        if candidates
        else "\n  (no entries under this domain)"
    )
    raise ContractError(
        f"inputs[{i}] memory_key does not exist: {key!r}{hint}"
    )


def compute_id(goal: str, constraints: Dict[str, Any], acceptance: List[Dict[str, Any]]) -> str:
    """Content-addressed id: sha256(goal|constraints_json|acceptance_json)[:12].

    Canonical form (sorted keys, no whitespace drift) so the SAME logical
    contract always yields the SAME id across processes and machines — the
    precondition for idempotent acceptance.
    """
    payload = "|".join([
        goal or "",
        json.dumps(constraints or {}, sort_keys=True, ensure_ascii=False),
        json.dumps(acceptance or [], sort_keys=True, ensure_ascii=False),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Contract:
    """An immutable task contract.

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
        """Validate in order. Raises ContractError on any fail.

        Args:
            check_inputs_exist: when True (parent-side pre-spawn check) every
                `inputs[*].value` path must currently exist on disk. The worker
                side re-validates the CONTRACT but not filesystem state (the
                path may be remote / fetched lazily).
        """
        # 1. goal non-empty, <= 200 chars; acceptance non-empty, each check non-empty
        if not self.goal or not self.goal.strip():
            raise ContractError(f"goal must be non-empty (got: {self.goal!r})")
        if len(self.goal) > GOAL_MAX_LEN:
            raise ContractError(
                f"goal exceeds {GOAL_MAX_LEN} chars (got {len(self.goal)}; "
                f"received: {self.goal[:120]!r}) — goal must be a one-sentence "
                f"task title; put the detailed charter/context into inputs "
                f"as kind=value items"
            )
        if not self.acceptance:
            raise ContractError("acceptance must be non-empty")
        for i, a in enumerate(self.acceptance):
            if not isinstance(a, dict) or not (a.get("check") or "").strip():
                raise ContractError(f"acceptance[{i}].check must be non-empty")

        # 2. inputs/output_ptr are absolute paths; inputs exist (parent-side)
        if not _is_abs(self.output_ptr):
            raise ContractError(f"output_ptr must be an absolute path: {self.output_ptr!r}")
        spec = role_spec(self.constraints)
        for i, inp in enumerate(self.inputs):
            if not isinstance(inp, dict):
                raise ContractError(f"inputs[{i}] must be a dict")
            val = inp.get("value", "")
            if inp.get("kind") == "path" and not _is_abs(val):
                raise ContractError(f"inputs[{i}].value must be an absolute path: {val!r}")
            if check_inputs_exist and inp.get("kind") == "path" and not os.path.exists(val):
                raise ContractError(f"inputs[{i}] path does not exist: {val!r}")
            if (check_inputs_exist and spec.live_tree_check
                    and inp.get("kind") == "path" and os.path.exists(val)):
                problem = check_live_tree(val)
                if problem:
                    raise ContractError(problem)
            if inp.get("kind") == "memory_key":
                if not isinstance(val, str) or not val.strip():
                    raise ContractError(f"inputs[{i}].value must be a non-empty string: {val!r}")
                if check_inputs_exist:
                    _check_memory_key_exists(i, val.strip())

        # 3. output_ptr ∈ writable
        writable = (self.constraints or {}).get("writable") or []
        if not any(_is_abs(w) and _within(self.output_ptr, w) for w in writable):
            raise ContractError(
                f"output_ptr {self.output_ptr!r} not inside any constraints.writable {writable!r}"
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

    # ── wire (cross-machine) ────────────────────────────────────
    def to_wire(self) -> Dict[str, Any]:
        """Serialize to the language-neutral JSON wire form.

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
    """True if `path` is `root` itself or a descendant of it.

    Both sides are canonicalized with `os.path.realpath` before the prefix test
    so a symlink cannot escape the root: a path like `root/link/secret` where
    `link -> /etc` normalizes to `root/link/secret` (normpath passes) but
    realpaths to `/etc/secret` and is correctly rejected. `realpath` also
    collapses any `..` segments, so it subsumes the old normpath check.
    """
    path = os.path.realpath(path)
    root = os.path.realpath(root)
    return path == root or path.startswith(root.rstrip("/") + "/")

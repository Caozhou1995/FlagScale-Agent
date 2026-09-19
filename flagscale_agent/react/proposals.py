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

"""Improvement-proposal registry — cross-session, stateful tracking of the
harness-improvement proposals the agent raises at wrap-up.

The wrap-up routine (VerificationGuard item 5, HARNESS GAP & CAPTURE) asks the
agent to propose improvements to the agent harness itself. Before this registry
those proposals lived only in the session transcript + a memory insight: they
had no persistent status, so a proposal the human never answered simply vanished
and was never resurfaced. The registry fixes that:

- PERSISTENT — one YAML file per proposal under a GLOBAL directory
  (~/.flagscale/proposals), so a later session sees an earlier session's still-
  open proposals.
- STATEFUL — each proposal has a status from a closed state machine:
      proposed  → raised, awaiting human review (default)
      approved  → human agreed; not yet implemented
      done      → human agreed AND it has been implemented
      rejected  → human declined
      superseded→ folded into another proposal
  OPEN_STATUSES = {proposed, approved} are the ones that must RESURFACE at the
  next wrap-up; TERMINAL_STATUSES = {done, rejected, superseded} are closed.

This module is the storage layer only. The `proposal` tool wraps it for the
agent, and the wrap-up prompt instructs the agent to reconcile open proposals.
"""

import os
import re
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import yaml


VALID_STATUSES = ("proposed", "approved", "done", "rejected", "superseded")
OPEN_STATUSES = ("proposed", "approved")
TERMINAL_STATUSES = ("done", "rejected", "superseded")

# Proposal id must be safe for use as a filename (no path traversal).
_ID_RE = re.compile(r"^prop_[a-z0-9]{4,12}$")


class ProposalRegistry:
    """Flat-file, per-proposal YAML store with a status state machine."""

    def __init__(self, proposals_dir: str):
        self._dir = proposals_dir
        self._lock = threading.RLock()

    # ── paths ────────────────────────────────────────────────────────────────
    def _path(self, proposal_id: str) -> str:
        if not _ID_RE.match(proposal_id or ""):
            raise ValueError(
                f"Invalid proposal id: {proposal_id!r} — must match {_ID_RE.pattern}"
            )
        return os.path.join(self._dir, f"{proposal_id}.yaml")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ── write ────────────────────────────────────────────────────────────────
    def _save(self, entry: dict):
        os.makedirs(self._dir, exist_ok=True)
        entry["updated"] = self._now()
        path = self._path(entry["id"])
        # Atomic write: tmp + os.replace, so a crash never leaves a half file.
        fd, tmp = tempfile.mkstemp(dir=self._dir, prefix=".tmp_prop_", suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.dump(entry, f, allow_unicode=True, default_flow_style=False)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def add(self, description: str, container: str = "", session_id: str = "",
            topic: str = "", status: str = "proposed") -> dict:
        """Register a new proposal. Returns the stored entry.

        Args:
            description: one-line description of the proposed improvement.
            container: routing — one of 'agent-code' / 'skill' / 'knowledge'
                (mirrors the three wrap-up proposal containers). Free-form,
                but these three are the expected values.
            session_id: session that raised it.
            topic: short slug for the proposal subject (used in summaries).
            status: initial status; default 'proposed'.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        with self._lock:
            now = self._now()
            pid = f"prop_{uuid.uuid4().hex[:8]}"
            entry = {
                "id": pid,
                "description": (description or "").strip(),
                "container": (container or "").strip(),
                "topic": (topic or "").strip(),
                "status": status,
                "created_session": session_id,
                "created_at": now,
                "updated": now,
                "status_history": [
                    {"status": status, "at": now, "by": session_id, "note": "created"}
                ],
            }
            self._save(entry)
            return entry

    def set_status(self, proposal_id: str, status: str, session_id: str = "",
                   note: str = "") -> dict:
        """Transition a proposal's status. Appends to status_history."""
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status: {status} — one of {VALID_STATUSES}"
            )
        with self._lock:
            entry = self.get(proposal_id)
            if not entry:
                raise ValueError(f"No such proposal: {proposal_id}")
            entry["status"] = status
            entry.setdefault("status_history", []).append({
                "status": status,
                "at": self._now(),
                "by": session_id,
                "note": note or "",
            })
            self._save(entry)
            return entry

    # ── read ─────────────────────────────────────────────────────────────────
    def get(self, proposal_id: str) -> Optional[dict]:
        try:
            path = self._path(proposal_id)
        except ValueError:
            return None
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def list_all(self) -> List[dict]:
        if not os.path.isdir(self._dir):
            return []
        out = []
        for fname in sorted(os.listdir(self._dir)):
            if not fname.startswith("prop_") or not fname.endswith(".yaml"):
                continue
            try:
                with open(os.path.join(self._dir, fname), "r", encoding="utf-8") as f:
                    e = yaml.safe_load(f)
                if isinstance(e, dict) and e.get("id"):
                    out.append(e)
            except Exception:
                continue
        return out

    def list_open(self) -> List[dict]:
        """Proposals still awaiting resolution (proposed/approved)."""
        return [e for e in self.list_all() if e.get("status") in OPEN_STATUSES]

    def list_status(self, status: str) -> List[dict]:
        return [e for e in self.list_all() if e.get("status") == status]

    # ── rendering ────────────────────────────────────────────────────────────
    @staticmethod
    def _fmt(e: dict) -> str:
        c = f"[{e.get('container')}] " if e.get("container") else ""
        return f"{e['id']} ({e.get('status','?')}): {c}{e.get('description','')}"

    def render_open(self) -> str:
        """Human/LLM-readable list of still-open proposals, or '' if none."""
        open_ = self.list_open()
        if not open_:
            return ""
        lines = ["Open improvement proposals still awaiting review:"]
        for e in sorted(open_, key=lambda x: x.get("created_at", "")):
            lines.append("  • " + self._fmt(e))
        return "\n".join(lines)

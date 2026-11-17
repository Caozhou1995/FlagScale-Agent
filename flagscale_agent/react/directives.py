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

"""User directives — the Do / Do NOT block on the dashboard.

SCOPE: SESSION, NOT DEPLOYMENT
------------------------------
The store lives inside the session directory (``<session_dir>/directives.json``),
exactly like ``conversation.json``. Not ``FLAGSCALE_HOME``, not a deployment-wide
path, not a shared work directory.

Why: the dashboard is a per-session artifact. Two sessions never share a prompt,
so a directive recorded in one must not appear in another — a Do/DoNOT line that
leaks across sessions is worse than no line at all, because it is an order the
user never gave for that task. The concrete failure that forced this scope:
storing under ``FLAGSCALE_HOME`` looked right for the interactive session and
collapsed for batch trials — each trial gets its own ``FLAGSCALE_HOME``
(``/logs/agent/.flagscale``), so the file existed for exactly one trial and the
others rendered an empty block. Seeding files by hand was the wrong fix: it is
unreproducible and does not survive the next launch.

WHAT THIS IS
------------
A small, durable store of rules the USER stated, in a form the agent sees
every turn. Two properties separate it from every other memory channel:

1. PROVENANCE IS THE USER. Each entry must carry the user's own words
   (``quote``) verbatim. Memory entries are the agent's own conclusions about
   the world; a directive is an ORDER about how to work. The distinction is
   load-bearing: measured failures of the "I already know better" kind
   (editing the deploy tree instead of the source tree; re-running the full
   suite when told to run only the neighbours) are not gaps in knowledge —
   the agent KNEW the rule and applied its own judgement instead. Knowledge
   competes with judgement; an order does not.

2. IT RIDES THE SYSTEM PROMPT, NOT THE MEMORY TOOL. Memory is retrieved
   on demand, so a rule nobody thinks to look up is a rule that never
   applies. The dashboard is rebuilt every turn and sits at the end of the
   prompt, so a directive is in front of the agent while it decides.

STORAGE
-------
``<session_dir>/directives.json``: a list of
``{"kind": "do"|"dont", "rule": str, "quote": str, "why": str,
   "added": ISO8601}``.
Best-effort and failure-isolated: a malformed or missing file degrades to an
empty list, because a directive-rendering hiccup must never break a run.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional

#: Hard caps. The block is rendered EVERY turn, so it is a recurring cost, not
#: a one-off. Measured: a readings block of ~21k chars per turn was already a
#: real context cost; directives must stay an order of magnitude below that.
MAX_DIRECTIVES = 24
MAX_RULE_CHARS = 240
MAX_QUOTE_CHARS = 300
MAX_WHY_CHARS = 200

#: Filename inside the session directory.
STORE_NAME = "directives.json"


def directives_path(session_dir: str) -> str:
    """Absolute path of THIS SESSION's directive store.

    ``session_dir`` is the same directory that holds ``conversation.json``.
    Required, never defaulted: a default would silently reinstate a
    deployment-wide store, which is the cross-session leak this scope fixes.
    """
    if not session_dir:
        raise ValueError("session_dir is required: directives are session-scoped")
    return os.path.join(os.path.abspath(session_dir), STORE_NAME)


def _read_raw(session_dir: str) -> List[Dict]:
    """Read the store, degrading to [] on any failure."""
    try:
        with open(directives_path(session_dir), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [e for e in data if isinstance(e, dict)]


def _write_raw(session_dir: str, entries: List[Dict]) -> None:
    """Atomically replace the store.

    Write-to-temp-then-rename: a crash mid-write must not leave a truncated
    JSON that silently drops every directive recorded in this session.
    """
    path = directives_path(session_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def normalize_kind(kind: str) -> str:
    """Map the many ways a caller can say "do not" onto two values."""
    k = (kind or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if k in ("dont", "donot", "dontdo", "never", "avoid", "not"):
        return "dont"
    return "do"


def _dedupe_key(text: str) -> str:
    """Normalise rule text for dedupe.

    Whitespace AND punctuation are stripped: the user restates the same order
    across sessions with a different comma ("改源码树，不要改部署树" vs
    "改源码树, 不要改部署树"), and two near-identical lines in a block rendered
    every turn is recurring cost with no added signal.  Measured: a
    whitespace-only key treated those two as different rules.
    """
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def list_directives(session_dir: str) -> List[Dict]:
    """Return this session's directives, newest first, already clipped."""
    out = []
    for e in _read_raw(session_dir):
        rule = _clip(str(e.get("rule", "")), MAX_RULE_CHARS)
        if not rule:
            continue
        out.append(
            {
                "kind": normalize_kind(str(e.get("kind", "do"))),
                "rule": rule,
                "quote": _clip(str(e.get("quote", "")), MAX_QUOTE_CHARS),
                "why": _clip(str(e.get("why", "")), MAX_WHY_CHARS),
                "added": str(e.get("added", "")),
            }
        )
    return out


def upsert_directive(
    session_dir: str, kind: str, rule: str, quote: str = "", why: str = ""
) -> Dict:
    """Add a directive, or REPLACE one whose rule text already exists.

    Dedupe is by normalised rule text, not by exact bytes, so restating an
    order in slightly different words updates the existing line instead of
    growing the block.
    """
    kind = normalize_kind(kind)
    rule = _clip(rule, MAX_RULE_CHARS)
    if not rule:
        raise ValueError("a directive needs a non-empty rule")
    quote = _clip(quote, MAX_QUOTE_CHARS)
    why = _clip(why, MAX_WHY_CHARS)
    key = _dedupe_key(rule)

    entries = _read_raw(session_dir)
    kept = [e for e in entries if _dedupe_key(str(e.get("rule", ""))) != key]
    record = {
        "kind": kind,
        "rule": rule,
        "quote": quote,
        "why": why,
        "added": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    kept.insert(0, record)
    # Newest-first, drop the oldest beyond the cap: a block that grows without
    # bound would eventually cost more context than the mistakes it prevents.
    _write_raw(session_dir, kept[:MAX_DIRECTIVES])
    return record


def remove_directive(session_dir: str, rule: str) -> bool:
    """Remove by normalised rule text. Returns True when something was removed."""
    key = _dedupe_key(rule)
    if not key:
        return False
    entries = _read_raw(session_dir)
    kept = [e for e in entries if _dedupe_key(str(e.get("rule", ""))) != key]
    if len(kept) == len(entries):
        return False
    _write_raw(session_dir, kept)
    return True


def render_directives_block(session_dir: Optional[str] = None) -> str:
    """Render the Do / Do NOT block for the dashboard.

    Empty string when there is nothing to say — an empty heading every turn
    would be noise that teaches the agent to skip the section entirely. Also
    empty when ``session_dir`` is unknown: no session, no orders.
    """
    if not session_dir:
        return ""
    entries = list_directives(session_dir)
    if not entries:
        return ""
    dos = [e for e in entries if e["kind"] == "do"]
    donts = [e for e in entries if e["kind"] == "dont"]

    lines = [
        "Directives from the USER (extracted from what the user actually said; "
        "these are ORDERS, not findings — they outrank your own judgement and "
        "your own memory entries when the two disagree):"
    ]
    if dos:
        lines.append("  DO:")
        for e in dos:
            line = f"    - {e['rule']}"
            if e["why"]:
                line += f" ({e['why']})"
            lines.append(line)
    if donts:
        lines.append("  DO NOT:")
        for e in donts:
            line = f"    - {e['rule']}"
            if e["why"]:
                line += f" ({e['why']})"
            lines.append(line)
    return "\n".join(lines)

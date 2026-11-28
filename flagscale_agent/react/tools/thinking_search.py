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

"""thinking_search tool — locate entries in the session's thinking_log.md.

The harness writes the session's reasoning to <session_dir>/thinking_log.md in
FULL (no truncation, no dropped entries). That is deliberate: a long reasoning
block states its conclusion at the END, so a head cap destroys exactly the
decision the far-end reminder asks about. The cost is that the file can grow to
megabytes, and slurping it into context is wasteful.

This tool closes that gap: it finds entries by keyword (AND semantics) and
returns either a short anchored snippet or ONE entry's full text. The reader
therefore gets the entry it needs, at full fidelity, without reading the file.
"""

import os
import re
from typing import Any, Dict, List, Optional

from flagscale_agent.react.tools.base import Tool

# Hard cap on the whole returned result (chars). An entry longer than this is
# returned as a head+tail window around the keyword instead of verbatim, so one
# pathological reasoning block cannot blow the context budget.
_MAX_RESULT_CHARS = 24000
# Snippet window when mode='snippet'.
_SNIPPET_CHARS = 700

# Tolerates a suffix after the number: logs written by the pre-2026-11-28 code
# carry "## reasoning N (truncated)" headers, and those files must stay
# searchable so an existing session's history is not lost to a parser details.
_ENTRY_RE = re.compile(r"^## reasoning (\d+).*$", re.MULTILINE)


def _split_entries(text: str) -> List[Dict[str, Any]]:
    """Split the log into entries: [{index, header, body}], 1-based index.

    A document with no `## reasoning N` marker yields one entry covering the
    whole file, so a hand-written or legacy log is still searchable.
    """
    marks = list(_ENTRY_RE.finditer(text))
    if not marks:
        return [{"index": 1, "body": text}]
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(marks):
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append({"index": int(m.group(1)), "body": text[start:end]})
    return out


def _matches(body: str, keywords: List[str]) -> bool:
    lowered = body.lower()
    return all(kw.lower() in lowered for kw in keywords)


def _snippet(body: str, keywords: List[str]) -> str:
    """Window around the earliest keyword hit, with a hit count."""
    lowered = body.lower()
    anchor = -1
    for kw in keywords:
        pos = lowered.find(kw.lower())
        if pos != -1 and (anchor == -1 or pos < anchor):
            anchor = pos
    if anchor == -1:
        return body[:_SNIPPET_CHARS].strip()
    start = max(0, anchor - _SNIPPET_CHARS // 4)
    end = start + _SNIPPET_CHARS
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(body) else ""
    return (prefix + body[start:end] + suffix).strip()


class ThinkingSearchTool(Tool):
    """Keyword search over the session's own thinking_log.md."""

    name = "thinking_search"
    description = (
        "Search THIS session's on-disk reasoning log (<session_dir>/"
        "thinking_log.md) by keyword. The log holds every reasoning block in "
        "FULL — nothing truncated — so a decision stated at the end of a long "
        "block is present verbatim. Use it whenever a reminder tells you to "
        "point at the earlier turn where a decision was made: search the "
        "decision's own words here instead of trying to recall.\n\n"
        "Modes: mode='snippet' (default) returns a window around the hit;\n"
        "mode='entry' returns the FULL text of the numbered entry, so nothing "
        "is cut.\n\n"
        "Examples:\n"
        "- thinking_search(query='5000 bytes limit') → which reasoning block "
        "stated that constraint\n"
        "- thinking_search(query='BsaI clamp', mode='entry', entry=1) → the full "
        "block, uncut\n"
        "- thinking_search(query='primer') → earliest matching window"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Keywords, space-separated = AND semantics (every keyword "
                    "must appear in the same reasoning entry). Case-insensitive."
                ),
            },
            "mode": {
                "type": "string",
                "description": (
                    "'snippet' (default): a window around the first hit per "
                    "entry. 'entry': the FULL text of one entry (needs entry=N, "
                    "or defaults to the newest match)."
                ),
            },
            "entry": {
                "type": "integer",
                "description": (
                    "With mode='entry': the reasoning entry number to dump in "
                    "full (as shown by a previous search's hits). Omit to dump "
                    "the newest matching entry."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max hits to list (default 8, max 30).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, session_dir_fn):
        # A callable (not a snapshot): resume/reload rebinds the session dir.
        self._session_dir_fn = session_dir_fn

    def _log_path(self) -> Optional[str]:
        try:
            session_dir = self._session_dir_fn() if self._session_dir_fn else None
        except Exception:
            session_dir = None
        if not session_dir:
            return None
        path = os.path.join(session_dir, "thinking_log.md")
        return path if os.path.exists(path) else None

    def execute(self, **kwargs) -> str:
        query = (kwargs.get("query") or "").strip()
        if not query:
            return "ERROR: query is required (space-separated keywords)."
        keywords = [k for k in query.split() if k]
        mode = kwargs.get("mode") or "snippet"
        if mode not in ("snippet", "entry"):
            mode = "snippet"
        limit = max(1, min(int(kwargs.get("limit", 8) or 8), 30))

        path = self._log_path()
        if path is None:
            return ("ERROR: thinking_log.md not found in this session dir — it is "
                    "written right before a post-edit reminder fires, so there is "
                    "nothing to search yet.")

        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            return "ERROR: cannot read %s: %s" % (path, e)

        entries = _split_entries(text)
        hits = [e for e in entries if _matches(e["body"], keywords)]

        if mode == "entry":
            want = kwargs.get("entry")
            if want is not None:
                chosen = next((e for e in entries if e["index"] == int(want)), None)
                if chosen is None:
                    return ("ERROR: no reasoning %s in %s (entries present: %d)."
                            % (want, path, len(entries)))
                return self._render_full(chosen, keywords, path)
            if not hits:
                return self._no_hit(keywords, path, len(entries))
            return self._render_full(hits[-1], keywords, path)

        lines = ["thinking_search: %d hit(s) for %r in %s (%d entries, %d bytes)"
                 % (len(hits), " ".join(keywords), path, len(entries), len(text))]
        if not hits:
            lines.append("No reasoning entry contains ALL keywords: %r" % keywords)
            lines.append("Tip: try fewer keywords — one distinctive noun from the "
                         "decision usually suffices.")
            return "\n".join(lines)
        lines.append("Call thinking_search(query=%r, mode='entry', entry=N) for "
                     "that entry's FULL uncut text." % " ".join(keywords))
        for e in hits[:limit]:
            snip = _snippet(e["body"], keywords).replace("\n", " ")
            lines.append("")
            lines.append("[reasoning %d | %d chars] %s"
                         % (e["index"], len(e["body"]), snip))
        if len(hits) > limit:
            lines.append("")
            lines.append("(%d more hit(s) not shown; raise limit or narrow the "
                         "query)" % (len(hits) - limit))
        return "\n".join(lines)

    def _no_hit(self, keywords: List[str], path: str, n_entries: int) -> str:
        return ("No reasoning entry in %s (%d entries) contains ALL keywords: %r"
                % (path, n_entries, keywords))

    def _render_full(self, entry: Dict[str, Any], keywords: List[str],
                     path: str) -> str:
        body = entry["body"].strip()
        header = ("thinking_search: FULL text of reasoning %d (%d chars) from %s"
                  % (entry["index"], len(body), path))
        if len(body) <= _MAX_RESULT_CHARS:
            return header + "\n\n" + body
        # Windowed fallback for one pathological entry: keep the head AND the
        # tail (conclusions live at the end) plus a window around the keyword.
        half = _MAX_RESULT_CHARS // 3
        lowered = body.lower()
        anchor = -1
        for kw in keywords:
            pos = lowered.find(kw.lower())
            if pos != -1 and (anchor == -1 or pos < anchor):
                anchor = pos
        mid_start = max(0, (anchor if anchor != -1 else len(body) // 2) - half // 2)
        return (header + " — TOO LONG FOR ONE RESULT: showing head + keyword "
                "window + tail\n\n"
                + "[head]\n" + body[:half]
                + "\n\n[…]\n\n[keyword window]\n"
                + body[mid_start:mid_start + half]
                + "\n\n[…]\n\n[tail — conclusions live here]\n" + body[-half:])

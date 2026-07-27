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

"""Evict summary store — tracks summaries of evicted messages for recall navigation.

Stores LLM-generated one-line summaries of evicted messages in a JSON file.
Allows the agent to browse evicted content by summary before deciding what to recall.
"""

import json
import os
import time
from typing import Optional


class EvictSummaryStore:
    """Manages evict_summaries.json — append-only summaries of evicted messages.

    File format: JSON array of objects:
    [
        {"index": 5, "role": "tool", "tool_name": "shell", "summary": "...", "ts": 1234567890},
        {"index": 6, "role": "assistant", "tool_name": null, "summary": "...", "ts": 1234567890},
        ...
    ]
    """

    def __init__(self, store_dir: str):
        """Initialize summary store.

        Args:
            store_dir: Directory for evict_summaries.json (same as swap_store dir's parent).
        """
        self._path = os.path.join(store_dir, "evict_summaries.json")
        self._cache: dict[int, dict] = {}  # index -> entry
        self._load()

    def _load(self):
        """Load existing summaries from disk."""
        if os.path.isfile(self._path):
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                for entry in entries:
                    self._cache[entry["index"]] = entry
            except (json.JSONDecodeError, IOError, KeyError):
                self._cache = {}

    def has(self, index: int) -> bool:
        """Check if a summary already exists for this index (idempotent guard)."""
        return index in self._cache

    def add(self, index: int, role: str, summary: str, tool_name: Optional[str] = None):
        """Add a summary entry. Skips if index already exists (idempotent).

        Args:
            index: Message index that was evicted.
            role: Message role (user, assistant, tool).
            summary: One-line LLM-generated summary.
            tool_name: Tool name if role is 'tool'.
        """
        if index in self._cache:
            return  # Already summarized, skip

        entry = {
            "index": index,
            "role": role,
            "tool_name": tool_name,
            "summary": summary,
            "ts": int(time.time()),
        }
        self._cache[index] = entry
        self._save()

    def add_batch(self, entries: list[dict]):
        """Add multiple entries at once. Each entry: {index, role, summary, tool_name?}.

        Skips entries whose index already exists.
        """
        added = False
        for entry in entries:
            idx = entry["index"]
            if idx in self._cache:
                continue
            self._cache[idx] = {
                "index": idx,
                "role": entry.get("role", "unknown"),
                "tool_name": entry.get("tool_name"),
                "summary": entry.get("summary", ""),
                "ts": int(time.time()),
            }
            added = True
        if added:
            self._save()

    def list_all(self) -> list[dict]:
        """Return all summaries sorted by index."""
        return sorted(self._cache.values(), key=lambda e: e["index"])

    def get(self, index: int) -> Optional[dict]:
        """Get summary for a specific index."""
        return self._cache.get(index)

    def _save(self):
        """Persist to disk (full rewrite — entries are small)."""
        entries = sorted(self._cache.values(), key=lambda e: e["index"])
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=1)

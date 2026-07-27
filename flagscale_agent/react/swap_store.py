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

"""Swap store for context management V3.

Stores evicted message content to local filesystem.
Messages are indexed by their position in the messages list.
"""

import json
import os
from typing import Optional


class SwapStore:
    """Local filesystem store for evicted message content.

    Each evicted message is stored as a JSON file: {index}.json
    containing the original content and metadata.
    """

    def __init__(self, store_dir: str):
        """Initialize swap store.

        Args:
            store_dir: Directory to store evicted content.
                       Typically session_dir/swap_store/
        """
        self._store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

    @property
    def store_dir(self) -> str:
        return self._store_dir

    def save(self, index: int, content, metadata: Optional[dict] = None) -> str:
        """Save evicted content to disk.

        Args:
            index: Message index in the messages list.
            content: The full message content to store (str or serializable).
            metadata: Optional metadata (tool_name, tool_input, etc.)

        Returns:
            Path to the stored file.
        """
        # Ensure content is a string for consistent load behavior
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        path = self._path_for(index)
        data = {
            "index": index,
            "content": content,
        }
        if metadata:
            data["metadata"] = metadata
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return path

    def load(self, index: int) -> Optional[str]:
        """Load evicted content from disk.

        Args:
            index: Message index to recall.

        Returns:
            The original content, or None if not found.
        """
        path = self._path_for(index)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("content")
        except (json.JSONDecodeError, IOError):
            return None

    def has(self, index: int) -> bool:
        """Check if an index exists in the store."""
        return os.path.isfile(self._path_for(index))

    def load_metadata(self, index: int) -> Optional[dict]:
        """Load metadata for an evicted message.

        Returns:
            The metadata dict, or None if not found or no metadata.
        """
        path = self._path_for(index)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("metadata")
        except (json.JSONDecodeError, IOError):
            return None

    def delete(self, index: int) -> bool:
        """Delete a stored entry (e.g., after permanent eviction).

        Returns:
            True if deleted, False if not found.
        """
        path = self._path_for(index)
        if os.path.isfile(path):
            os.remove(path)
            return True
        return False

    def list_indexes(self) -> list:
        """List all stored indexes."""
        indexes = []
        if not os.path.isdir(self._store_dir):
            return indexes
        for fname in os.listdir(self._store_dir):
            if fname.endswith(".json"):
                try:
                    indexes.append(int(fname[:-5]))
                except ValueError:
                    pass
        return sorted(indexes)

    def size_bytes(self) -> int:
        """Total bytes used by swap store."""
        total = 0
        if not os.path.isdir(self._store_dir):
            return 0
        for fname in os.listdir(self._store_dir):
            path = os.path.join(self._store_dir, fname)
            if os.path.isfile(path):
                total += os.path.getsize(path)
        return total

    def _path_for(self, index: int) -> str:
        """Get file path for a given index."""
        return os.path.join(self._store_dir, f"{index}.json")

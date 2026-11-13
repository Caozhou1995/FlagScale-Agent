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

"""Read file tool with line range support and short-term caching."""

import itertools
import os
import time

from flagscale_agent.react.tools.base import Tool


class FileCache:
    """Short-term file content cache with TTL and invalidation on write.

    Design principles:
    - TTL-based expiry: cached content expires after _TTL seconds
    - mtime-based validation: if file mtime changes, cache is invalid
    - Explicit invalidation: write_file/edit_file calls invalidate the path
    - Memory-bounded: max _MAX_ENTRIES, LRU eviction
    """

    _TTL = 30  # seconds
    _MAX_ENTRIES = 50

    def __init__(self):
        self._store = {}  # normpath -> (content_lines, mtime, cached_at)

    @staticmethod
    def _key(path: str) -> str:
        """Canonical cache key for a caller-supplied path.

        The cache is keyed by the path STRING, but callers spell the same file
        many ways (`sub/k.py`, `./sub/k.py`, an absolute path, a symlinked
        path). Keying on the raw string made invalidation miss: a write via
        `./sub/k.py` left the entry cached under `sub/k.py`, so a later read
        returned the PRE-write content while the file on disk was new
        (measured: read returned A=1, file held A=777).

        realpath (not just abspath) also collapses symlinks, and it does not
        require the file to exist, so a read of a not-yet-written path and a
        later write to it still share one key.
        """
        try:
            return os.path.realpath(path)
        except (OSError, ValueError):
            return os.path.abspath(path)

    def get(self, path: str):
        """Return cached lines or None if cache miss/stale."""
        key = self._key(path)
        entry = self._store.get(key)
        if entry is None:
            return None
        content_lines, cached_mtime, cached_at = entry
        # TTL check
        if time.time() - cached_at > self._TTL:
            del self._store[key]
            return None
        # mtime check — file may have been modified externally
        try:
            current_mtime = os.path.getmtime(path)
            if current_mtime != cached_mtime:
                del self._store[key]
                return None
        except OSError:
            del self._store[key]
            return None
        return content_lines

    def put(self, path: str, lines: list):
        """Cache file content."""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        key = self._key(path)
        # LRU eviction: remove oldest entry if at capacity
        if len(self._store) >= self._MAX_ENTRIES and key not in self._store:
            oldest_key = min(self._store, key=lambda k: self._store[k][2])
            del self._store[oldest_key]
        self._store[key] = (lines, mtime, time.time())

    def invalidate(self, path: str):
        """Explicitly invalidate a path (called after write/edit).

        Uses the same canonical key as get/put, so a write spelled differently
        from the earlier read (`./sub/k.py` vs `sub/k.py`) still drops the
        entry. Callers that pass several spellings of one path are harmless.
        """
        self._store.pop(self._key(path), None)

    def invalidate_all(self):
        """Clear entire cache."""
        self._store.clear()


# Module-level singleton so it's shared across tool instances
_file_cache = FileCache()


def get_file_cache() -> FileCache:
    """Access the shared file cache (for invalidation from write/edit tools)."""
    return _file_cache


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a file. Returns content with line numbers for easy reference. "
        "Hard limit: 500 lines per call. Files over 500 lines MUST be read in segments "
        "using start_line/end_line. For files under 500 lines, omit start_line/end_line "
        "to read in full. Re-reading the same file wastes tokens — save key findings "
        "to memory instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The file path to read.",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-based). Default: 1",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (inclusive). Default: end of file. Max 500 lines per call.",
            },
            "numbered": {
                "type": "boolean",
                "description": "Include line numbers. Default: true",
            },
        },
        "required": ["path"],
    }

    MAX_LINES = 500

    # Only files up to this size are cached at all. The cache stored the WHOLE
    # line list per path with no byte bound, so a few dozen MB-scale files
    # (agent logs are routinely that size) pinned hundreds of MB: measured
    # 8 distinct 20MB paths held 1.84M lines / 308MB RSS, and _MAX_ENTRIES=50
    # extrapolates to ~1.9GB. Above the cap we simply do not cache.
    _MAX_CACHE_BYTES = 4 * 1024 * 1024

    @staticmethod
    def _count_lines(fb) -> int:
        """Count newline-terminated lines in a binary handle, in fixed-size chunks.

        Streaming so the count costs O(1) memory: the previous code learned the
        line count only by materialising every line via f.readlines().
        A trailing partial line counts as one line.
        """
        block = 1024 * 1024
        count = 0
        last_byte = b""
        while True:
            chunk = fb.read(block)
            if not chunk:
                break
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
        if last_byte and last_byte != b"\n":
            count += 1
        return count

    def execute(self, **kwargs) -> str:
        path = kwargs["path"]
        start = kwargs.get("start_line", 1)
        end = kwargs.get("end_line")
        numbered = kwargs.get("numbered", True)

        if not os.path.exists(path):
            return f"ERROR: File not found: {path}"
        if os.path.isdir(path):
            return f"ERROR: Path is a directory: {path}"

        # Try cache first (only for full-file reads without line range)
        is_full_read = (start <= 1 and end is None)
        lines = None
        if is_full_read:
            lines = _file_cache.get(path)

        if lines is not None:
            total = len(lines)
            start = max(1, start)
            if end is None:
                end = min(start + self.MAX_LINES - 1, total)
            end = min(end, total)
            selected = lines[start - 1:end]
        else:
            # Cache miss: read ONLY the requested window. f.readlines() used to
            # pull the entire file into memory no matter how few lines were
            # asked for (measured: 500 lines out of an 80MB log = 129MB peak).
            try:
                size = os.path.getsize(path)
            except OSError as e:
                return f"ERROR: {e}"
            try:
                with open(path, "rb") as fb:
                    total = self._count_lines(fb)
            except Exception as e:
                return f"ERROR: {e}"

            start = max(1, start)
            if end is None:
                end = min(start + self.MAX_LINES - 1, total)
            end = min(end, total)

            if start > total:
                return f"File has {total} lines, requested start_line={start} is past end."

            need = end - start + 1
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    selected = list(itertools.islice(f, start - 1, end))
            except Exception as e:
                return f"ERROR: {e}"

            # Cache only when the whole file is small enough to bound memory.
            # A full read is the common repeat case, so populate the cache then;
            # a ranged read of a small file is cached too (cheap, bounded).
            if size <= self._MAX_CACHE_BYTES:
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        all_lines = f.readlines()
                    if len(all_lines) == total:
                        _file_cache.put(path, all_lines)
                except Exception:
                    pass  # cache population is best-effort, never fatal

        truncated = end < total and (end - start + 1) >= self.MAX_LINES

        if numbered:
            width = len(str(end))
            output_lines = [f"{i:{width}d}| {line}" for i, line in enumerate(selected, start=start)]
        else:
            output_lines = selected

        header = f"[{path}] lines {start}-{end} of {total}"
        result = header + "\n" + "".join(output_lines)

        if truncated:
            result += f"\n... truncated at {self.MAX_LINES} lines. Use start_line={end + 1} to continue."

        return result

"""Context management: evict and recall operations.

Extracted from agent.py to reduce its size and isolate context-window
management logic into a dedicated module.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from flagscale_agent.react import display

if TYPE_CHECKING:
    from flagscale_agent.react.history import HistoryManager
    from flagscale_agent.react.swap_store import SwapStore


class ContextManager:
    """Handles evict/recall tool operations.

    Dependencies are injected at construction so this class is testable
    without a full Agent instance.
    """

    def __init__(
        self,
        history: "HistoryManager",
        swap_store: "SwapStore",
    ):
        self.history = history
        self.swap_store = swap_store

    def handle_evict(self, arguments: dict) -> str:
        """Process an evict tool call against the message history."""
        indexes = arguments.get("indexes", []) if arguments else []
        if indexes:
            display.tool_start("evict", f"indexes={indexes[:10]}{'...' if len(indexes) > 10 else ''}")
        else:
            display.tool_start("evict", "")
        t0 = time.time()

        if not arguments:
            result_msg = "ERROR: 'indexes' parameter is required and must be a non-empty list."
            display.tool_done("evict", time.time() - t0, detail="missing indexes", error=True)
            return result_msg

        if not indexes:
            result_msg = "ERROR: 'indexes' parameter is required and must be a non-empty list."
            display.tool_done("evict", time.time() - t0, detail="empty indexes", error=True)
            return result_msg

        if not isinstance(indexes, list):
            if isinstance(indexes, (int, float)):
                indexes = [int(indexes)]
            else:
                result_msg = "ERROR: 'indexes' must be a list of integers."
                display.tool_done("evict", time.time() - t0, detail="invalid type", error=True)
                return result_msg

        evicted_count = 0
        freed_tokens = 0
        errors = []

        for idx in indexes:
            if isinstance(idx, float) and idx == int(idx):
                idx = int(idx)
            if not isinstance(idx, int):
                errors.append(f"index {idx}: not an integer")
                continue

            # Evict the message (history generates placeholder locally)
            result = self.history.evict_message(idx)
            if result is None:
                errors.append(f"index {idx}: not evictable (already evicted, system prompt, protected tail)")
                continue

            content = result["content"]
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            self.swap_store.save(idx, content, result.get("metadata"))
            evicted_count += 1
            freed_tokens += result.get("metadata", {}).get("tokens", 0)

            # Handle paired eviction (tool_use + tool_result must stay paired)
            paired = result.get("paired_evict")
            if paired:
                paired_idx = paired["index"]
                paired_content = paired["content"]
                paired_content_str = paired_content
                if not isinstance(paired_content_str, str):
                    paired_content_str = json.dumps(paired_content, ensure_ascii=False)
                paired_meta = {
                    "role": "user",
                    "tokens": paired["tokens"],
                    "paired_with": idx,
                }
                self.swap_store.save(paired_idx, paired_content_str, paired_meta)
                evicted_count += 1
                freed_tokens += paired["tokens"]
                primary_meta = result.get("metadata", {})
                primary_meta["paired_with"] = paired_idx
                self.swap_store.save(idx, content, primary_meta)

        # Display result
        result_msg = f"Evicted {evicted_count} message(s), freed ~{freed_tokens} tokens."
        if errors:
            result_msg += f" Skipped: {'; '.join(errors[:5])}"
        elapsed = time.time() - t0
        display.tool_done("evict", elapsed, detail=f"{evicted_count} evicted, ~{freed_tokens} tokens freed")

        return result_msg



    def handle_recall(self, arguments: dict) -> str:
        """Process a recall tool call — retrieve evicted content from swap store."""
        index = arguments.get("index") if arguments else None
        display.tool_start("recall", f"index={index}")
        t0 = time.time()

        if not arguments:
            display.tool_done("recall", time.time() - t0, detail="missing index", error=True)
            return "ERROR: 'index' parameter is required."

        if index is None:
            display.tool_done("recall", time.time() - t0, detail="missing index", error=True)
            return "ERROR: 'index' parameter is required."
        if isinstance(index, float) and index == int(index):
            index = int(index)
        if not isinstance(index, int):
            display.tool_done("recall", time.time() - t0, detail="invalid type", error=True)
            return "ERROR: 'index' must be an integer."

        content = self.swap_store.load(index)
        if content is None:
            # Fallback: try recall from full_log
            content = self.history.recall_from_full_log(index)
            if content is None:
                display.tool_done("recall", time.time() - t0, detail=f"index {index} not found", error=True)
                return f"ERROR: No evicted content found at index {index}."
            elapsed = time.time() - t0
            display.tool_done("recall", elapsed, detail=f"index={index} from full_log")
            return content

        # Restore the original content back into history
        restored = self.history.recall_message(index, content)

        # Also restore paired message to maintain tool_use/tool_result pairing
        metadata = self.swap_store.load_metadata(index)
        paired_idx = metadata.get("paired_with") if metadata else None
        if paired_idx is not None:
            paired_content = self.swap_store.load(paired_idx)
            if paired_content is not None:
                self.history.recall_message(paired_idx, paired_content)

        restore_status = "restored" if restored else "returned"
        elapsed = time.time() - t0
        display.tool_done("recall", elapsed, detail=f"index={index} {restore_status}")

        return content




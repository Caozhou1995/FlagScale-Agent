"""Context management: evict, recall, evict_list operations.

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
    from flagscale_agent.react.evict_summary import EvictSummaryStore


class ContextManager:
    """Handles evict/recall/evict_list tool operations.

    Dependencies are injected at construction so this class is testable
    without a full Agent instance.
    """

    def __init__(
        self,
        history: "HistoryManager",
        swap_store: "SwapStore",
        evict_summary: "EvictSummaryStore",
        provider=None,
    ):
        self.history = history
        self.swap_store = swap_store
        self.evict_summary = evict_summary
        self.provider = provider

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
        evicted_count = 0
        freed_tokens = 0
        errors = []
        messages_for_summary = []  # [(index, role, tool_name, content)]

        for idx in indexes:
            if isinstance(idx, float) and idx == int(idx):
                idx = int(idx)
            if not isinstance(idx, int):
                errors.append(f"index {idx}: not an integer")
                continue

            # Skip if already summarized (idempotent)
            if self.evict_summary.has(idx):
                result = self.history.evict_message(idx)
                if result:
                    content = result["content"]
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    self.swap_store.save(idx, content, result.get("metadata"))
                    evicted_count += 1
                    freed_tokens += result.get("metadata", {}).get("tokens", 0)
                continue

            # Get message content BEFORE evicting (for summary generation)
            msg = self.history.get_message_at(idx)
            if msg is None:
                errors.append(f"index {idx}: out of range")
                continue

            # Extract info for summary
            role = msg.get("role", "unknown")
            tool_name = None
            content_for_summary = ""

            if role == "tool":
                tool_name = msg.get("name") or msg.get("tool_name", "unknown")
                content_for_summary = msg.get("content", "")
                if isinstance(content_for_summary, list):
                    content_for_summary = json.dumps(content_for_summary, ensure_ascii=False)
            elif role == "assistant":
                content_for_summary = msg.get("content", "")
                if isinstance(content_for_summary, list):
                    parts = []
                    for block in content_for_summary:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_use":
                                tool_name = block.get("name", "")
                                parts.append(f"[tool_use: {tool_name}({json.dumps(block.get('input', {}), ensure_ascii=False)})]")
                    content_for_summary = "\n".join(parts)
            elif role == "user":
                content_for_summary = msg.get("content", "")
                if isinstance(content_for_summary, list):
                    parts = []
                    for block in content_for_summary:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            elif block.get("type") == "tool_result":
                                parts.append(f"[tool_result: {block.get('content', '')}]")
                    content_for_summary = "\n".join(parts)

            messages_for_summary.append((idx, role, tool_name, content_for_summary))

            # Now evict
            result = self.history.evict_message(idx)
            if result is None:
                errors.append(f"index {idx}: not evictable (already evicted, system prompt, protected tail)")
                messages_for_summary.pop()
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
                paired_summary_content = paired_content_str if paired_content_str else f"[tool_result paired with index {idx}]"
                messages_for_summary.append((paired_idx, "user", None, paired_summary_content))

        # Generate summaries via LLM for newly evicted messages
        if messages_for_summary:
            self._generate_summaries(messages_for_summary)

        # Display result
        result_msg = f"Evicted {evicted_count} message(s), freed ~{freed_tokens} tokens."
        if errors:
            result_msg += f" Skipped: {'; '.join(errors[:5])}"
        elapsed = time.time() - t0
        display.tool_done("evict", elapsed, detail=f"{evicted_count} evicted, ~{freed_tokens} tokens freed")

        return result_msg

    def _generate_summaries(self, messages: list):
        """Generate LLM summaries for evicted messages and store them.

        Each message gets its own LLM call for precise summary.
        Uses ThreadPoolExecutor for concurrent API calls.
        Short messages (<=150 chars) are used directly as their own summary.

        Args:
            messages: List of (index, role, tool_name, content_str) tuples.
        """
        if not messages:
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed

        SHORT_THRESHOLD = 150

        needs_llm = []
        results = {}

        for idx, role, tool_name, content in messages:
            content_clean = (content or "").strip()
            if len(content_clean) <= SHORT_THRESHOLD:
                summary = content_clean.replace("\n", " ") if content_clean else f"[{role}] {tool_name or 'empty message'}"
                results[idx] = summary
            else:
                needs_llm.append((idx, role, tool_name, content))

        def _summarize_one(idx: int, role: str, tool_name: str, content: str) -> tuple:
            """Call LLM to generate one-line summary."""
            prompt = (
                "Summarize this evicted conversation message in ONE concise line (max 120 chars). "
                "State the concrete action or content factually. Do NOT infer intent or add interpretation.\n\n"
                f"[role={role}"
            )
            if tool_name:
                prompt += f", tool={tool_name}"
            prompt += f"]\n{content}"

            try:
                response = self.provider.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=[],
                )
                summary = ""
                if isinstance(response, dict):
                    resp_content = response.get("content", "")
                    if isinstance(resp_content, list):
                        for block in resp_content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                summary += block.get("text", "")
                    elif isinstance(resp_content, str):
                        summary = resp_content
                    if not summary:
                        msg = response.get("message", {})
                        summary = msg.get("content", "") if isinstance(msg, dict) else ""
                return (idx, summary.strip().split("\n")[0][:200])
            except Exception:
                fallback = content.replace("\n", " ") if content else f"[{role}]"
                return (idx, f"[no-llm] {fallback}")

        if needs_llm:
            max_workers = min(len(needs_llm), 10)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_summarize_one, idx, role, tool_name, content): idx
                    for idx, role, tool_name, content in needs_llm
                }
                for future in as_completed(futures):
                    try:
                        idx, summary = future.result(timeout=30)
                        results[idx] = summary
                    except Exception:
                        idx = futures[future]
                        results[idx] = "[timeout] summary generation failed"

        # Store all summaries and update placeholders in history
        for idx, role, tool_name, _ in messages:
            summary = results.get(idx, f"[{role}] {tool_name or 'message'}")
            self.evict_summary.add(idx, role, summary, tool_name)
            self.history.update_evict_placeholder(idx, summary)

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

        summary_entry = self.evict_summary.get(index)
        summary_hint = summary_entry.get("summary", "") if summary_entry else ""
        restore_status = "restored" if restored else "returned"
        elapsed = time.time() - t0
        display.tool_done("recall", elapsed, detail=f"index={index} {restore_status} | {summary_hint}")

        return content

    def handle_evict_list(self, arguments: dict) -> str:
        """List all evicted message summaries for recall navigation."""
        keyword = arguments.get("keyword", "") if arguments else ""
        display.tool_start("evict_list", f"keyword='{keyword}'" if keyword else "")
        t0 = time.time()

        entries = self.evict_summary.list_all()
        if not entries:
            display.tool_done("evict_list", time.time() - t0, detail="0 entries")
            return "No evicted messages with summaries found."

        kw_lower = keyword.lower()

        lines = []
        for e in entries:
            if kw_lower and kw_lower not in e.get("summary", "").lower() and kw_lower not in (e.get("tool_name") or "").lower():
                continue
            tool_part = f" [{e['tool_name']}]" if e.get("tool_name") else ""
            lines.append(f"  {e['index']:>5}: ({e['role']}{tool_part}) {e['summary']}")

        if not lines:
            display.tool_done("evict_list", time.time() - t0, detail=f"0 matching '{keyword}'")
            return f"No evicted messages matching '{keyword}'."

        header = f"Evicted messages ({len(lines)} entries):"
        if keyword:
            header = f"Evicted messages matching '{keyword}' ({len(lines)} entries):"

        result = header + "\n" + "\n".join(lines)
        elapsed = time.time() - t0
        display.tool_done("evict_list", elapsed, detail=f"{len(lines)} entries")
        return result


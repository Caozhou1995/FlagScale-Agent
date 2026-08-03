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

"""Conversation history management — V3 (evict/recall based).

Context management is handled by the model via evict/recall tools.
No automatic aging, truncation, or compaction.
"""

import json
from typing import Any, Dict, List, Optional

# Working window ratio: 60% of max_context_tokens
WORKING_WINDOW_RATIO = 0.60
# Fallback if not dynamically set
WORKING_WINDOW_TOKENS = 120_000


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English, ~1.5 tokens per CJK char."""
    if not text:
        return 1  # Every message has at least structural overhead
    # Count CJK characters (they typically become 2-3 tokens each in BPE)
    cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
                    or '\u3040' <= c <= '\u30ff'
                    or '\uac00' <= c <= '\ud7af')
    ascii_count = len(text) - cjk_count
    # CJK: ~1.5 tokens per char; ASCII: ~0.25 tokens per char (4 chars/token)
    tokens = int(cjk_count * 1.5) + (ascii_count // 4)
    return max(1, tokens)


def _message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate tokens in a single message."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return _estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += _estimate_tokens(json.dumps(block, ensure_ascii=False))
            else:
                total += _estimate_tokens(str(block))
        return total
    return _estimate_tokens(json.dumps(msg, ensure_ascii=False))


def _is_tool_result(msg: Dict[str, Any]) -> bool:
    """Check if a message is a tool result (OpenAI role=tool or Anthropic tool_result block)."""
    if msg.get("role") == "tool":
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    return False


def _has_tool_use(msg: Dict[str, Any]) -> bool:
    """Check if an assistant message contains tool_use blocks."""
    if msg.get("tool_calls"):
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
    return False


def _validate_tool_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure valid conversation structure:
    1. Every tool_result has a preceding tool_use (remove orphaned ones)
    2. Merge consecutive user messages (required by Anthropic API)
    """
    # Step 1: Remove orphaned tool results
    result = []
    for i, msg in enumerate(messages):
        if _is_tool_result(msg):
            # Check if there's a matching tool_call_id in history
            tool_call_id = msg.get("tool_call_id", "")
            tool_use_id = ""
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        break

            has_match = False
            search_id = tool_call_id or tool_use_id
            if search_id:
                for prev in result:
                    if prev.get("role") == "assistant":
                        # OpenAI format
                        for tc in prev.get("tool_calls", []):
                            if tc.get("id") == search_id:
                                has_match = True
                                break
                        # Anthropic format
                        prev_content = prev.get("content")
                        if isinstance(prev_content, list):
                            for block in prev_content:
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    if block.get("id") == search_id:
                                        has_match = True
                                        break
                    if has_match:
                        break
            else:
                # No ID — check if previous message is assistant with tool_use
                if result and result[-1].get("role") == "assistant" and _has_tool_use(result[-1]):
                    has_match = True

            if has_match:
                result.append(msg)
            # else: drop orphaned tool_result
        else:
            result.append(msg)

    # Step 2: Merge consecutive user messages (Anthropic requires alternating roles)
    merged = []
    for msg in result:
        if msg.get("role") == "user" and merged and merged[-1].get("role") == "user":
            merged[-1] = _merge_user_messages(merged[-1], msg)
        else:
            merged.append(msg)

    return merged


def _merge_user_messages(msg1: Dict[str, Any], msg2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two consecutive user messages into one with list content."""
    def _to_blocks(msg):
        content = msg.get("content", "")
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            return content
        return [{"type": "text", "text": str(content)}]

    blocks = _to_blocks(msg1) + _to_blocks(msg2)
    return {"role": "user", "content": blocks}


class HistoryManager:
    """Manages conversation message history.

    V3 design: no automatic compaction/aging/truncation.
    Context management is fully handled by the model via evict/recall tools.
    Guard provides pressure awareness, model decides what to evict.
    """

    def __init__(self, max_context_tokens: int = 200000):
        self.max_context_tokens = max_context_tokens
        self.working_window = int(max_context_tokens * WORKING_WINDOW_RATIO)
        self._messages: List[Dict[str, Any]] = []
        self._full_log: List[Dict[str, Any]] = []
        self._actual_input_tokens: int = 0
        # Legacy properties (kept for compatibility, no longer used for compaction)
        self._compaction_count: int = 0
        self._compaction_happened: bool = False

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return self._messages

    @property
    def compaction_happened(self) -> bool:
        """V3: always False — no automatic compaction."""
        return False

    def append(self, message: Dict[str, Any]):
        import copy
        self._messages.append(message)
        # Store a deep copy so eviction (which modifies in-place) doesn't affect the full log
        self._full_log.append(copy.deepcopy(message))

    def set_system_prompt(self, content: str):
        """Replace or prepend the system message."""
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = content
        else:
            self._messages.insert(0, {"role": "system", "content": content})

    def report_actual_tokens(self, input_tokens: int):
        """Feed back the actual input_tokens from the API response."""
        self._actual_input_tokens = input_tokens

    def get_context_pressure(self) -> float:
        """Return current context usage as ratio against dynamic working window (60% of max_context_tokens)."""
        estimated = sum(_message_tokens(m) for m in self._messages)
        actual = self._actual_input_tokens or 0
        total = max(estimated, actual)
        return total / self.working_window

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return messages list. No aging or compaction — evict/recall handles context."""
        return _validate_tool_pairs(list(self._messages))

    def get_message_at(self, index: int) -> Optional[Dict[str, Any]]:
        """Return message at given index, or None if out of range or already evicted."""
        if index < 0 or index >= len(self._messages):
            return None
        msg = self._messages[index]
        if msg.get("_evicted"):
            return None
        return msg

    # Legacy stubs (no-op, kept for backward compatibility with kernel/commands)
    def clear(self):
        """Clear all messages."""
        self._messages.clear()
        self._full_log.clear()
        self._actual_input_tokens = 0

    # ── Evict/Recall (V3 Context Management) ──────────────────────────────────

    def evict_message(self, index: int) -> Dict[str, Any] | None:
        """Evict a message at the given index.

        Can evict any message except:
        - System prompt (index 0 if role=system)
        - Already evicted messages
        - The last 4 messages (to keep recent context intact)

        Returns the original message data (for storage), or None if invalid.
        """
        if index < 0 or index >= len(self._messages):
            return None
        msg = self._messages[index]
        # Never evict system prompt
        if msg.get("role") == "system":
            return None
        if msg.get("_evicted"):
            return None
        # Protect the last 4 messages (recent context)
        if index >= len(self._messages) - 4:
            return None

        content = msg.get("content", "")
        tokens = _message_tokens(msg)

        # Build placeholder based on message type
        role = msg.get("role", "unknown")
        if _is_tool_result(msg):
            tool_name, tool_input = self._extract_tool_info_for_index(index)
            placeholder = (
                f"[evicted | index={index} | {tool_name}({tool_input}) | {tokens} tokens]"
            )
        else:
            # For assistant/user messages, show a brief summary
            text_preview = ""
            if isinstance(content, str):
                text_preview = content[:50].replace("\n", " ")
            elif isinstance(content, list):
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        text_preview = b.get("text", "")[:50].replace("\n", " ")
                        break
            placeholder = (
                f"[evicted | index={index} | role={role} | {text_preview}... | {tokens} tokens]"
            )

        original_content = content

        # If this assistant message contains tool_use blocks, we must also evict
        # the next user message (which contains the paired tool_result blocks).
        # Otherwise the API will see orphaned tool_result without matching tool_use.
        paired_evict = None
        if role == "assistant" and _has_tool_use(msg):
            next_idx = index + 1
            if next_idx < len(self._messages) - 4:
                next_msg = self._messages[next_idx]
                if next_msg.get("role") == "user" and not next_msg.get("_evicted"):
                    next_content = next_msg.get("content", "")
                    next_tokens = _message_tokens(next_msg)
                    next_placeholder = (
                        f"[evicted | index={next_idx} | role=user | tool_results | {next_tokens} tokens]"
                    )
                    paired_evict = {
                        "index": next_idx,
                        "content": next_content,
                        "tokens": next_tokens,
                        "placeholder": next_placeholder,
                    }
        # Reverse: if evicting a user message with tool_result, also evict preceding assistant
        elif role == "user" and _is_tool_result(msg):
            prev_idx = index - 1
            if prev_idx > 0:
                prev_msg = self._messages[prev_idx]
                if prev_msg.get("role") == "assistant" and _has_tool_use(prev_msg) and not prev_msg.get("_evicted"):
                    prev_content = prev_msg.get("content", "")
                    prev_tokens = _message_tokens(prev_msg)
                    prev_placeholder = (
                        f"[evicted | index={prev_idx} | role=assistant | tool_use | {prev_tokens} tokens]"
                    )
                    paired_evict = {
                        "index": prev_idx,
                        "content": prev_content,
                        "tokens": prev_tokens,
                        "placeholder": prev_placeholder,
                    }

        msg["content"] = placeholder
        msg["_evicted"] = True
        msg["_evicted_tokens"] = tokens

        # Apply paired eviction
        if paired_evict:
            next_msg = self._messages[paired_evict["index"]]
            next_msg["content"] = paired_evict["placeholder"]
            next_msg["_evicted"] = True
            next_msg["_evicted_tokens"] = paired_evict["tokens"]

        metadata = {
            "role": role,
            "tokens": tokens,
        }
        # Include tool info in metadata for tool_result messages
        if _is_tool_result(msg) or (role == "tool"):
            tool_name_meta, tool_input_meta = self._extract_tool_info_for_index(index)
            metadata["tool_name"] = tool_name_meta
            metadata["tool_input"] = tool_input_meta

        return {
            "content": original_content,
            "metadata": metadata,
            "paired_evict": paired_evict,
        }

    def recall_message(self, index: int, content: str) -> bool:
        """Restore evicted content at a given index (in-place).
        
        Clears the _evicted flag to allow re-eviction if needed.
        This enables evict → recall → re-evict cycles for dynamic context management.

        Returns True if restored, False if index invalid or not evicted.
        """
        if index < 0 or index >= len(self._messages):
            return False
        msg = self._messages[index]
        if not msg.get("_evicted"):
            return False
        # Deserialize JSON content back to original structure if possible
        # (swap_store serializes lists/dicts to JSON strings; restore them for
        # proper _has_tool_use / _is_tool_result detection on re-eviction)
        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith(("[", "{")):
                try:
                    import json as _json
                    content = _json.loads(stripped)
                except (ValueError, TypeError):
                    pass
        msg["content"] = content
        # Clear _evicted flag to allow this message to be evicted again
        del msg["_evicted"]
        if "_evicted_tokens" in msg:
            del msg["_evicted_tokens"]
        return True

    def update_evict_placeholder(self, index: int, summary: str) -> bool:
        """Update an evicted message's placeholder with a better summary.

        Called after LLM generates a summary, to replace the raw content
        truncation with a meaningful one-liner.

        Returns True if updated, False if not evicted at that index.
        """
        if index < 0 or index >= len(self._messages):
            return False
        msg = self._messages[index]
        if not msg.get("_evicted"):
            return False
        tokens = msg.get("_evicted_tokens", 0)
        role = msg.get("role", "unknown")
        # Rebuild placeholder with summary
        msg["content"] = (
            f"[evicted | index={index} | role={role} | {summary} | {tokens} tokens]"
        )
        return True

    def get_evictable_indexes(self) -> List[int]:
        """Return indexes of messages that can be evicted, in index order.

        All messages except system prompt, already-evicted, and last 4 are evictable.
        Returns in natural index order — no priority implied.
        LLM should decide what to evict based on its own judgment.
        """
        protected_tail = max(0, len(self._messages) - 4)
        result = []
        for i, msg in enumerate(self._messages):
            if msg.get("_evicted"):
                continue
            if msg.get("role") == "system":
                continue
            if i >= protected_tail:
                continue
            result.append(i)
        return result

    def _extract_tool_info_for_index(self, index: int) -> tuple:
        """Extract tool_name and key input for a tool_result at index."""
        tool_name = "unknown"
        tool_input = ""

        for i in range(index - 1, -1, -1):
            msg = self._messages[i]
            if msg.get("role") == "assistant":
                # Anthropic format
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_result_id = self._messages[index].get("tool_use_id", "")
                            if block.get("id", "") == tool_result_id or not tool_result_id:
                                tool_name = block.get("name", "unknown")
                                tool_input = self._summarize_tool_input(
                                    tool_name, block.get("input", {})
                                )
                                return (tool_name, tool_input)
                # OpenAI format
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tool_result_id = self._messages[index].get("tool_call_id", "")
                    for tc in tool_calls:
                        tc_id = tc.get("id", "")
                        if tc_id == tool_result_id or not tool_result_id:
                            fn = tc.get("function", {})
                            tool_name = fn.get("name", "unknown")
                            try:
                                args = json.loads(fn.get("arguments", "{}"))
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                            tool_input = self._summarize_tool_input(tool_name, args)
                            return (tool_name, tool_input)
                break
        return (tool_name, tool_input)

    @staticmethod
    def _summarize_tool_input(tool_name: str, args: dict) -> str:
        """Create a short summary of tool input for the placeholder."""
        if tool_name == "read_file":
            return args.get("path", "")[:80]
        if tool_name == "shell":
            cmd = args.get("command", "")
            return cmd[:60] + ("..." if len(cmd) > 60 else "")
        if tool_name == "write_file":
            return args.get("path", "")[:80]
        if tool_name == "edit_file":
            return args.get("path", "")[:80]
        if tool_name == "web_fetch":
            return args.get("url", "")[:80]
        if tool_name in ("find_latest_log", "parse_training_metrics"):
            return args.get("experiment", args.get("log_path", ""))[:60]
        if tool_name == "monitor":
            return args.get("file", args.get("output_dir", ""))[:60]
        for v in args.values():
            if isinstance(v, str) and v:
                return v[:60]
        return ""

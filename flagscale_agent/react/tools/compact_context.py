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

"""CompactContext tool — triggers eviction of old messages to free context space.

In V3 context management, compaction = eviction. This tool evicts the oldest
evictable messages to free up working space. The LLM should prefer calling
evict() directly with specific indexes, but this tool provides a quick
"free up space" action when precise selection isn't needed.
"""

from flagscale_agent.react.tools.base import Tool


class CompactContextTool(Tool):
    name = "compact_context"
    description = (
        "Free context space by evicting old messages. "
        "Equivalent to calling evict() on the oldest evictable messages. "
        "Prefer using evict() directly for precise control. "
        "This is a convenience shortcut when you just need space quickly."
    )
    parameters = {
        "type": "object",
        "properties": {
            "percent": {
                "type": "integer",
                "description": "Percentage of evictable messages to evict (10-80). Default: 30",
            },
            "reason": {
                "type": "string",
                "description": "Why you're compacting (logged for debugging).",
            },
        },
        "required": [],
    }

    def __init__(self, history_manager):
        self._history = history_manager

    def execute(self, **kwargs) -> str:
        percent = kwargs.get("percent", 30)
        reason = kwargs.get("reason", "proactive compaction")

        percent = max(10, min(80, percent))

        evictable = self._history.get_evictable_indexes()
        if not evictable:
            return "Nothing to evict — all messages are either protected or already evicted."

        count = max(1, len(evictable) * percent // 100)
        to_evict = evictable[:count]

        evicted_count = 0
        freed_tokens = 0
        for idx in to_evict:
            from flagscale_agent.react.history import _message_tokens
            msg = self._history._messages[idx]
            tokens = _message_tokens(msg)
            result = self._history.evict_message(idx)
            if result is not None:
                evicted_count += 1
                freed_tokens += tokens

        if evicted_count == 0:
            return "Eviction failed — no messages could be evicted."

        return (
            f"Evicted {evicted_count} messages, freed ~{freed_tokens} tokens. "
            f"Reason: {reason}"
        )

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

"""Evict tool — swap out conversation messages to free context space."""

from flagscale_agent.react.tools.base import Tool, ToolEffect


class EvictTool(Tool):
    name = "evict"
    effects = ToolEffect()
    description = (
        "Swap out conversation messages to free context space. "
        "Replaces message content with a lightweight placeholder. "
        "Can evict ANY message (tool_result, assistant, user) except the system prompt and "
        "the last 4 messages. "
        "Use wide index ranges (e.g. indexes=[1,2,3,...,100]) to free large amounts. "
        "Each evicted message shows [evicted | index=N | ...] as placeholder — "
        "use recall(index=N) later if you need the content back."
    )
    parameters = {
        "type": "object",
        "properties": {
            "indexes": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "List of message indexes to evict. Use broad ranges for bulk eviction. "
                    "Already-evicted indexes are silently skipped. "
                    "Tip: start from low indexes (oldest messages) for maximum impact."
                ),
            },
        },
        "required": ["indexes"],
    }

    def execute(self, **kwargs) -> str:
        """Evict is handled specially by the agent loop.

        This method should not be called directly — the agent intercepts
        evict calls and processes them against the message history.
        If called directly (e.g., in testing without agent), return an error.
        """
        return (
            "ERROR: evict must be processed by the agent loop. "
            "Direct execution is not supported."
        )

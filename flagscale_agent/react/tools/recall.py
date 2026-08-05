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

"""Recall tool — retrieve previously evicted message content."""

from flagscale_agent.react.tools.base import Tool


class RecallTool(Tool):
    name = "recall"
    description = (
        "Retrieve a previously evicted message by its index. "
        "Use when you see a placeholder like [evicted | index=N | ...] "
        "and need the full content back. Returns the original message content. "
        "Works for any evicted message type (tool_result, assistant, user)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "description": (
                    "The message index from the evicted placeholder "
                    "(shown as index=N in the placeholder text)."
                ),
            },
        },
        "required": ["index"],
    }

    def execute(self, **kwargs) -> str:
        """Recall is handled specially by the agent loop.

        This method should not be called directly — the agent intercepts
        recall calls and retrieves content from the swap store.
        If called directly (e.g., in testing without agent), return an error.
        """
        return (
            "ERROR: recall must be processed by the agent loop. "
            "Direct execution is not supported."
        )

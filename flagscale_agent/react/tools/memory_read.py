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

"""Memory read tool — retrieve a specific memory entry or list by prefix."""

from flagscale_agent.react.tools.base import Tool


class MemoryReadTool(Tool):
    name = "memory_read"
    description = (
        "Read a specific memory entry by key, or list entries by prefix.\n\n"
        "Exact key: memory_read(key='fact/cluster/ssh_port')\n"
        "Prefix: memory_read(key='fact/cluster/') → all cluster facts\n"
        "Prefix: memory_read(key='pitfall/') → all pitfalls"
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Exact key (e.g. 'fact/cluster/ssh_port') or prefix ending "
                    "with '/' (e.g. 'fact/cluster/', 'pitfall/nccl/')."
                ),
            },
        },
        "required": ["key"],
    }

    def __init__(self, memory):
        self._memory = memory

    def execute(self, **kwargs) -> str:
        key = kwargs["key"]

        # Prefix mode: key ends with /
        if key.endswith("/"):
            entries = self._memory.list_by_prefix(key)
            if not entries:
                return f"No entries found with prefix '{key}'."
            lines = []
            for e in entries:
                content = e.get("content", "")
                if len(content) > 200:
                    content = content[:197] + "..."
                lines.append(
                    f"[{e.get('type', '?')}] {e.get('key', '?')}: {content}"
                )
            return f"Found {len(entries)} entries:\n" + "\n".join(lines)

        # Exact key mode
        entry = self._memory.get(key)
        if entry is None:
            return f"No memory found for '{key}'."
        return (
            f"[{entry.get('type', '?')}] {entry.get('key', '?')}\n"
            f"Content: {entry.get('content', '')}\n"
            f"Created: {entry.get('created_at', '?')} "
            f"(session: {entry.get('created_session', '?')})\n"
            f"Updated: {entry.get('updated_at', '?')} "
            f"(session: {entry.get('updated_session', '?')})"
        )

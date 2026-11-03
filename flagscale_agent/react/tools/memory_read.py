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

    def _similar_keys(self, key: str, limit: int = 3):
        """Up to `limit` existing keys sharing this key's domain (best-effort).

        The miss message is only useful if it points at the nearest real entry:
        a typo'd key is the common cause, and the domain segment
        (type/domain/) survives most typos. Never raises — a suggestion is
        advisory, and a memory backend quirk must not turn a miss into an error.
        """
        try:
            parts = key.strip("/").split("/")
            if len(parts) < 2:
                return []
            prefix = "/".join(parts[:2]) + "/"
            found = [e.get("key", "") for e in self._memory.list_by_prefix(prefix)]
            return [k for k in found if k and k != key][:limit]
        except Exception:
            return []

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
            # this used to be a bare "No memory found". Audits
            # #78/#81 measured the failure shape that invites: the agent read
            # this line and then, in the SAME step, quoted a body for that key in
            # its self-assessment ("Observed from world: memory_read returned …")
            # — a citation of a read that returned nothing. The miss must be
            # stated as a PROHIBITION, not an absence, because "not found" reads
            # as "look elsewhere" while the real rule is "do not cite this key".
            similar = self._similar_keys(key)
            hint = ""
            if similar:
                hint = ("\nNearest existing keys (read one of these before "
                        "reusing or overriding it):\n  "
                        + "\n  ".join(similar))
            return (
                f"No memory found for '{key}'. Do NOT cite this key or quote a "
                "content for it: nothing was returned, so any body you attribute "
                "to it in a step close, a report, or a plan note is invented. If "
                "you need it, either write it first (memory_write) or read a key "
                "that exists." + hint
            )
        return (
            f"[{entry.get('type', '?')}] {entry.get('key', '?')}\n"
            f"Content: {entry.get('content', '')}\n"
            f"Created: {entry.get('created_at', '?')} "
            f"(session: {entry.get('created_session', '?')})\n"
            f"Updated: {entry.get('updated_at', '?')} "
            f"(session: {entry.get('updated_session', '?')})"
        )

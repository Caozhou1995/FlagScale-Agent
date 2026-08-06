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

"""Memory write tool — save facts, pitfalls, and insights."""

from flagscale_agent.react.tools.base import Tool


class MemoryWriteTool(Tool):
    name = "memory_write"
    description = (
        "Save a memory entry for cross-session continuity. "
        "Three types only:\n"
        "- fact: verifiable environment state (a value, path, config)\n"
        "- pitfall: failure experience (symptom → cause → fix)\n"
        "- insight: undigested pattern (discovery + digestion direction + target)\n\n"
        "Key format: type/domain/specific (e.g. fact/cluster/ssh_port, "
        "pitfall/nccl/nic_exclude_syntax, insight/agent/memory_redesign).\n\n"
        "Writing the same key updates the existing entry. "
        "Use 'supersedes' to delete old entries that this new one replaces.\n\n"
        "Write conditions:\n"
        "- fact: info obtained by probing, not obvious, likely needed in future sessions\n"
        "- pitfall: debugging took >2 rounds, cause was non-obvious, likely to recur\n"
        "- insight: reusable pattern found, cannot digest now, has clear target artifact\n\n"
        "Do NOT use memory for: "
        "session temp state (→ plan/context), easily re-read configs (→ read_file), "
        "complete procedures (→ skill), systematic knowledge (→ knowledge)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Three-level key: type/domain/specific. "
                    "Examples: 'fact/cluster/ssh_port', 'pitfall/nccl/nic_hang', "
                    "'insight/skill/nccl_debug_method'. "
                    "All segments lowercase, alphanumeric + underscore only."
                ),
            },
            "type": {
                "type": "string",
                "enum": ["fact", "pitfall", "insight"],
                "description": "Memory type. Must match the first segment of the key.",
            },
            "content": {
                "type": "string",
                "description": (
                    "The memory content. Format by type:\n"
                    "- fact: '值: X\\n适用: Y\\n验证命令: Z'\n"
                    "- pitfall: '现象: X\\n原因: Y\\n解决: Z\\n环境: W'\n"
                    "- insight: '发现: X\\n消化方向: Y\\n目标产物: Z'"
                ),
            },
            "supersedes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of old memory keys to delete (this entry replaces them).",
            },
        },
        "required": ["key", "type", "content"],
    }

    def __init__(self, memory, session_id: str = "", task_plan=None):
        self._memory = memory
        self._session_id = session_id
        self._task_plan = task_plan

    def _get_current_task(self) -> str:
        if self._task_plan:
            active = self._task_plan.get_active()
            if active:
                return active.get("title", "")
        return ""

    def execute(self, **kwargs) -> str:
        key = kwargs["key"]
        mem_type = kwargs["type"]
        content = kwargs["content"]
        supersedes = kwargs.get("supersedes", [])
        task = self._get_current_task()

        from flagscale_agent.react.memory import Memory, VALID_TYPES

        # Validate type
        if mem_type not in VALID_TYPES:
            return (
                f"ERROR: Invalid type '{mem_type}'. "
                f"Must be one of: {sorted(VALID_TYPES)}."
            )

        # Validate key format
        error = Memory.validate_key(key)
        if error:
            return f"ERROR: Invalid key '{key}'. {error}"

        # Validate type consistency: key prefix must match type
        key_type = key.split("/")[0]
        if key_type != mem_type:
            return (
                f"ERROR: Key prefix '{key_type}' does not match "
                f"type '{mem_type}'. They must be the same."
            )

        try:
            # Delete superseded entries
            deleted = []
            for old_key in supersedes:
                if self._memory.delete(old_key):
                    deleted.append(old_key)

            # Write new entry
            self._memory.put(key, mem_type, content, self._session_id, task=task)

            supersede_info = f" Superseded: {', '.join(deleted)}." if deleted else ""
            return (
                f"Memorized [{mem_type}] '{key}' "
                f"({len(content)} chars).{supersede_info}"
            )
        except Exception as e:
            return f"ERROR: Failed to save memory: {e}"

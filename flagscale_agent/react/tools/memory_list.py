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

"""Memory list tool — browse and search memory entries, grouped by type."""

from flagscale_agent.react.tools.base import Tool


class MemoryListTool(Tool):
    name = "memory_list"
    description = (
        "List and search memory entries. Returns entries grouped by type "
        "(fact → pitfall → insight). Supports domain-level filtering and "
        "keyword search.\n\n"
        "Usage patterns:\n"
        "- memory_list() → browse all entries, see domains\n"
        "- memory_list(keyword='nccl') → find nccl-related entries\n"
        "- memory_list(type_filter='pitfall') → all pitfalls\n"
        "- memory_list(domain_filter='cluster') → all cluster-domain entries"
    )
    parameters = {
        "type": "object",
        "properties": {
            "type_filter": {
                "type": "string",
                "enum": ["fact", "pitfall", "insight", ""],
                "description": "Filter by memory type. Empty for all.",
            },
            "domain_filter": {
                "type": "string",
                "description": "Filter by domain segment (e.g. 'cluster', 'nccl', 'env').",
            },
            "keyword": {
                "type": "string",
                "description": "Search keyword (case-insensitive substring match on key and content).",
            },
            "limit": {
                "type": "integer",
                "description": "Max entries to return (default 30).",
            },
        },
        "required": [],
    }

    def __init__(self, memory):
        self._memory = memory

    def execute(self, **kwargs) -> str:
        type_filter = kwargs.get("type_filter", "")
        domain_filter = kwargs.get("domain_filter", "")
        keyword = kwargs.get("keyword", "")
        limit = kwargs.get("limit", 30)

        entries = self._memory.list_entries(
            type_filter=type_filter,
            domain_filter=domain_filter,
            keyword=keyword,
        )

        if not entries:
            parts = []
            if type_filter:
                parts.append(f"type={type_filter}")
            if domain_filter:
                parts.append(f"domain={domain_filter}")
            if keyword:
                parts.append(f"keyword='{keyword}'")
            filter_desc = ", ".join(parts) if parts else "no filters"
            return f"(no memory entries found matching {filter_desc})"

        # Group by type for display
        grouped = {"fact": [], "pitfall": [], "insight": []}
        for e in entries:
            t = e.get("type", "fact")
            if t in grouped:
                grouped[t].append(e)

        lines = []
        shown = 0
        for type_name in ("fact", "pitfall", "insight"):
            group = grouped[type_name]
            if not group:
                continue
            lines.append(f"\n── {type_name} ({len(group)}) ──")
            for e in group:
                if shown >= limit:
                    break
                key = e.get("key", "?")
                content = e.get("content", "")
                task = e.get("task", "")
                # Show content on single line, replacing newlines
                content_oneline = content.replace("\n", " | ")
                task_tag = f" @{task}" if task else ""
                lines.append(f"  {key}{task_tag}: {content_oneline}")
                shown += 1
            if shown >= limit:
                break

        total = len(entries)
        header = f"Showing {min(shown, total)}/{total} entries"
        if type_filter or domain_filter or keyword:
            header += " (filtered)"

        return header + "\n" + "\n".join(lines)

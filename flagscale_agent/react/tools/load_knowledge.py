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

"""Load knowledge tool — retrieves domain knowledge for distributed training."""

from flagscale_agent.react.tools.base import Tool


class LoadKnowledgeTool(Tool):
    name = "load_knowledge"
    description = (
        "Load domain knowledge for distributed training infrastructure. "
        "Returns a structured index (TOC with file paths and line numbers) by default. "
        "Use index_only=false to load full document content. "
        "Pass name='list' to see all available knowledge groups."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": (
                    "Knowledge group name (e.g. 'know-nccl-core', 'know-flash-attn'). "
                    "Pass 'list' to see all available groups."
                ),
            },
            "index_only": {
                "type": "boolean",
                "description": (
                    "If true (default), return only the index/TOC with line numbers. "
                    "If false, return full content of all docs in the group."
                ),
            },
        },
        "required": ["name"],
    }

    def __init__(self, knowledge_manager):
        self._km = knowledge_manager

    def execute(self, **kwargs) -> str:
        name = kwargs.get("name", "")
        index_only = kwargs.get("index_only", True)

        if name == "list":
            groups = self._km.list_groups()
            lines = ["Available knowledge groups:\n"]
            for g in groups:
                lines.append(
                    f"  {g['name']}: {g['description']} ({g['doc_count']} docs)"
                )
            return "\n".join(lines)

        if name not in self._km.available_groups:
            available = ", ".join(self._km.available_groups)
            return f"Unknown group '{name}'. Available: {available}"

        if index_only:
            index = self._km.get_index(name)
            if index:
                return index
            return f"No index found for '{name}'. Try regenerating indexes."
        else:
            content = self._km.load_knowledge(name)
            if content:
                return content
            return f"No content found for '{name}'."

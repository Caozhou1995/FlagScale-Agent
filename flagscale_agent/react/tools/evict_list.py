"""Evict list tool — browse summaries of evicted messages to find what to recall."""

from flagscale_agent.react.tools.base import Tool, ToolEffect


class EvictListTool(Tool):
    name = "evict_list"
    description = (
        "List summaries of all evicted messages. Use this BEFORE recall() to find the right "
        "index to restore. Each entry shows: index, role, tool name, and a one-line summary "
        "of the original content. Optionally filter by keyword."
    )
    parameters = {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "Optional keyword to filter summaries (case-insensitive match on summary text and tool name).",
            },
        },
        "required": [],
    }
    effects = ToolEffect()

    def execute(self, **kwargs) -> str:
        # Execution is intercepted by agent._handle_evict_list
        return "ERROR: evict_list should be intercepted by the agent loop."

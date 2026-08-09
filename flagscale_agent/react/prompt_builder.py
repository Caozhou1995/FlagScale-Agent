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

"""System prompt builder for FlagScale Agent.

V2 redesign: builds a static prompt body (cache-friendly) with a tiny
dashboard appended at the end. Memory and plan are NOT injected into the
prompt — they are accessed on-demand via tools (memory_list, plan_status).
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING, Dict

from flagscale_agent.react.prompt import (
    SYSTEM_PROMPT_STATIC,
    SYSTEM_PROMPT_OPTIONAL,
    DASHBOARD_TEMPLATE,
)

if TYPE_CHECKING:
    from flagscale_agent.react.skills import SkillManager


class PromptBuilder:
    """Assembles the system prompt from static template + optional sections + dashboard."""

    def __init__(self, skill_manager: "SkillManager"):
        self._skill_manager = skill_manager
        self._turn_count = 0

    def refresh(
        self,
        history,
        active_skill_content: dict[str, str],
        shared_storage_paths: list[str],
        tool_names: list[str] | None = None,
        # Legacy params — accepted but ignored (removed from prompt injection)
        memory_context: str = "",
        plan_context: str = "",
        session_dir: str = "",
    ):
        """Build and set the system prompt on the history manager.

        Args:
            history: HistoryManager instance to set prompt on
            active_skill_content: IGNORED (kept for backward compat, skill content no longer injected)
            shared_storage_paths: Detected shared filesystem paths
            tool_names: List of available tool names
            memory_context: IGNORED (kept for backward compat, not injected)
            plan_context: IGNORED for prompt injection (used only for dashboard)
        """
        self._turn_count += 1

        # ── Tool names ──
        tools_str = (
            ", ".join(tool_names)
            if tool_names
            else "read_file, write_file, edit_file, shell, web_fetch, load_skill, "
            "memory_write, memory_read, memory_list, monitor, plan_create, "
            "plan_update, plan_status"
        )

        # ── Skills summary for header ──
        skills_summary = self._build_skills_summary()

        # ── Knowledge summary for header ──
        knowledge_summary = self._build_knowledge_summary()

        # ── Optional sections based on scene constraints ──
        optional_sections = self._build_optional_sections(plan_context)

        # ── Shared storage note ──
        shared_storage_note = self._build_shared_storage_note(shared_storage_paths)

        # ── Assemble static block ──
        core = SYSTEM_PROMPT_STATIC.format(
            cwd=os.getcwd(),
            tools=tools_str,
            skills=skills_summary,
            knowledge=knowledge_summary,
            optional_sections=optional_sections + shared_storage_note,
        )

        # ── Append dashboard at the very end ──
        dashboard = self._build_dashboard(plan_context, session_dir)
        if dashboard:
            core += DASHBOARD_TEMPLATE.format(dashboard_content=dashboard)

        history.set_system_prompt(core)

    def _build_optional_sections(self, plan_context: str) -> str:
        """Select optional sections."""
        optional_parts = []

        # Planning section — always inject (includes "when to create" guidance)
        planning = SYSTEM_PROMPT_OPTIONAL.get("planning", "")
        if planning:
            optional_parts.append(planning)

        # Always include these
        optional_parts.append(SYSTEM_PROMPT_OPTIONAL.get("memory_rules", ""))
        optional_parts.append(SYSTEM_PROMPT_OPTIONAL.get("decision", ""))

        # User commands only on first 3 turns
        if self._turn_count <= 3:
            optional_parts.append(SYSTEM_PROMPT_OPTIONAL.get("user_commands", ""))

        return "\n\n".join(p for p in optional_parts if p)

    def _build_skills_summary(self) -> str:
        """Build compact summary of all available skills for the header line."""
        try:
            available = self._skill_manager.list_skills()
            lines = []
            for s in available:
                name = s.get("name", "")
                desc = s.get("description", "")
                lines.append(f"- {name}: {desc}")
            return "\n".join(lines)
        except Exception:
            return "(skills not available)"

    def _build_knowledge_summary(self) -> str:
        """Build compact summary of available knowledge groups."""
        try:
            from flagscale_agent.knowledge import KnowledgeManager
            km = KnowledgeManager()
            groups = km.list_groups()
            if not groups:
                return "(no knowledge loaded)"
            lines = []
            for g in groups:
                lines.append(f"- {g['name']}: {g['description']}")
            return "\n".join(lines)
        except Exception:
            return "(knowledge not available)"

    def _build_dashboard(self, plan_context: str, session_dir: str = "") -> str:
        """Build the dashboard line for the end of the prompt.

        Extracts plan title/step from plan_context if available.
        Format: "Task: <title> | Step: N/M | Turn: <n>"
        Appends session paths so the agent can access conversation logs directly.
        """
        import re
        parts = []

        if plan_context:
            # Extract title from <active-plan title="...">
            title_match = re.search(r'title="([^"]*)"', plan_context)
            if title_match:
                title = title_match.group(1).strip()
                if title:
                    parts.append(f"Task: {title}")

            # Count total steps and find current step
            step_lines = re.findall(r'\[.\] Step (\d+):', plan_context)
            total = len(step_lines)
            # Current step is the one with 🔄 or the first ⬜
            doing_match = re.search(r'\[🔄\] Step (\d+):', plan_context)
            pending_match = re.search(r'\[⬜\] Step (\d+):', plan_context)
            if doing_match:
                current = int(doing_match.group(1))
                parts.append(f"Step: {current}/{total}")
            elif pending_match:
                current = int(pending_match.group(1))
                parts.append(f"Step: {current}/{total}")

        parts.append(f"Turn: {self._turn_count}")

        # Session paths — injected so agent can read logs without shell(find ...)
        if session_dir:
            parts.append(
                f"Session: {session_dir}"
                f" | conversation.json: {session_dir}/conversation.json"
                f" | conversation_full.json: {session_dir}/conversation_full.json"
            )

        # Memory keys — list known keys so agent can memory_read without memory_list()
        memory_keys = self._build_memory_keys_summary()
        if memory_keys:
            parts.append(f"Memory keys: {memory_keys}")

        return " | ".join(parts)

    def _build_memory_keys_summary(self) -> str:
        """Return comma-separated list of all memory keys (no values)."""
        try:
            from flagscale_agent.react.memory import AgentMemory
            from flagscale_agent.react.paths import get_memory_dir
            mem = AgentMemory(get_memory_dir())
            entries = mem.list_entries()
            if not entries:
                return ""
            return ", ".join(e["key"] for e in entries)
        except Exception:
            return ""

    def _build_shared_storage_note(self, shared_storage_paths: list[str]) -> str:
        """Build a note about shared storage paths for conda environments."""
        if not shared_storage_paths:
            return ""
        return (
            "\n\n## Shared Storage\n\nAvailable shared storage paths:\n"
            + "\n".join(f"- `{p}`" for p in shared_storage_paths)
            + "\n\nWhen creating conda environments, use `--prefix` targeting "
            "one of these paths instead of `-n <name>`.\n"
        )

    # Keep old name for backward compatibility
    build_shared_storage_note = _build_shared_storage_note

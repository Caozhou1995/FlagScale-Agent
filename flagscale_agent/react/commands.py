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

"""Command handlers for FlagScale Agent slash commands.

Extracted from agent.py to reduce file size and improve separation of concerns.
"""

import os
import time

from flagscale_agent.react.session import (
    find_resumable_sessions, load_conversation, mark_completed,
)
from flagscale_agent.react.tools.shell import ShellTool


class CommandHandler:
    """Handles slash commands for WorkerAgent.

    This class encapsulates all CLI command handling logic, keeping agent.py
    focused on core agent orchestration.
    """

    def __init__(self, agent):
        """Initialize with reference to parent agent.

        Args:
            agent: WorkerAgent instance that owns this handler
        """
        self.agent = agent

    def _generate_resume_summary(self, session_info: dict) -> str:
        """Generate session summary via LLM for sessions missing it.

        Loads conversation, calls LLM to produce 3-line summary,
        saves it back to conversation.json for future use.
        """
        session_dir = session_info.get("session_dir", "")
        try:
            data = load_conversation(session_dir)
            if not data:
                return "(无法加载会话)"
            messages = data.get("messages", [])
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if not user_msgs:
                return "(空会话)"

            # Extract first + last few user messages as context
            first_msg = ""
            content = user_msgs[0].get("content", "")
            if isinstance(content, str):
                first_msg = content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        first_msg = block.get("text", "")
                        break

            recent = user_msgs[-5:]
            recent_texts = []
            for m in recent:
                c = m.get("content", "")
                if isinstance(c, str):
                    recent_texts.append(c)
                elif isinstance(c, list):
                    for block in c:
                        if isinstance(block, dict) and block.get("type") == "text":
                            recent_texts.append(block.get("text", ""))
                            break

            context_text = f"首条消息: {first_msg}\n最近消息:\n" + "\n".join(recent_texts)

            prompt_msgs = [
                {"role": "user", "content": (
                    "请用中文为这个会话生成一个简短摘要，严格3行，不要多余格式：\n"
                    "第1行：这个会话主要在做什么（一句话）\n"
                    "第2行：当前进展到哪里了（一句话）\n"
                    "第3行：下一步待做什么（没有则写'无下一步待做'）\n\n"
                    f"{context_text}\n\n"
                    "直接输出3行摘要，不要任何前缀或解释。"
                )}
            ]

            response = self.agent.provider.chat(prompt_msgs, [])
            result_text = ""
            if isinstance(response, dict):
                resp_content = response.get("content", "")
                if isinstance(resp_content, list):
                    for block in resp_content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            result_text += block.get("text", "")
                elif isinstance(resp_content, str):
                    result_text = resp_content

            summary = result_text.strip()
            if summary:
                # Save back to conversation.json for future use
                import json
                conv_path = os.path.join(session_dir, "conversation.json")
                if os.path.isfile(conv_path):
                    with open(conv_path, "r", encoding="utf-8") as f:
                        conv_data = json.load(f)
                    conv_data["session_summary"] = summary
                    with open(conv_path, "w", encoding="utf-8") as f:
                        json.dump(conv_data, f, ensure_ascii=False, indent=2)
            return summary or "(生成摘要失败)"
        except Exception as e:
            return f"(生成摘要失败: {str(e)})"

    def handle_slash_command(self, user_input: str) -> bool:
        """Dispatch slash command to appropriate handler.

        Args:
            user_input: Raw user input starting with / (or bare 'resume')

        Returns:
            True if command was handled, False otherwise
        """
        # Allow bare "resume" or "resume <arg>" without / prefix
        stripped = user_input.strip()
        if stripped == "resume" or stripped.startswith("resume "):
            self._handle_resume("/" + stripped)
            return True

        cmd = user_input.split()[0] if user_input.startswith("/") else None
        if not cmd:
            return False

        if cmd == "/quit":
            self.agent._exit()
            return True
        elif cmd == "/reload":
            self._handle_reload(user_input)
            return True
        elif cmd == "/skill":
            self._handle_skill(user_input)
            return True

        elif cmd == "/save":
            self.agent._save_conversation(completed=False)
            print("Conversation saved.")
            return True

        elif cmd == "/memory":
            self._handle_memory(user_input)
            return True
        elif cmd == "/mode":
            self._handle_mode(user_input)
            return True
        elif cmd == "/plan":
            self._handle_plan(user_input)
            return True
        elif cmd == "/resume":
            self._handle_resume(user_input)
            return True
        elif cmd == "/compact":
            evictable = self.agent.history.get_evictable_indexes()
            if not evictable:
                print("Nothing to evict.")
            else:
                count = max(1, len(evictable) * 30 // 100)
                evicted = 0
                for idx in evictable[:count]:
                    if self.agent.history.evict_message(idx) is not None:
                        evicted += 1
                print(f"Evicted {evicted} messages.")
            return True
        elif cmd == "/reset":
            self._handle_reset()
            return True
        elif cmd == "/session":
            self._handle_session()
            return True
        return False

    def _handle_reset(self):
        """Handle /reset command - manually trigger context hard reset."""
        print("Triggering context hard reset...")
        try:
            self.agent._hard_reset_context()
            print("Hard reset complete.")
        except Exception as e:
            print(f"Hard reset failed: {e}")

    def _handle_skill(self, user_input: str):
        """Handle /skill command - list or load skills."""
        parts = user_input.split()
        if len(parts) < 2:
            skills = self.agent.skill_manager.list_skills()
            print("Available skills:")
            for s in skills:
                print(f"  {s['name']}: {s['description']}")
            return
        name = parts[1]
        try:
            self.agent.skill_manager.load(name)
            print(f"Skill '{name}' loaded.")
        except FileNotFoundError:
            print(f"Skill '{name}' not found.")

    def _handle_memory(self, user_input: str):
        """Handle /memory command - manage session memory."""
        parts = user_input.split()
        if len(parts) < 2:
            print("Usage: /memory list | /memory clear [type] | /memory delete <key>")
            return
        sub = parts[1]
        if sub == "list":
            entries = self.agent.memory.list_entries()
            if not entries:
                print("No memory entries.")
                return
            for e in entries:
                key = e.get("key", "?")
                mem_type = e.get("type", "?")
                content = e.get("content", "")
                print(f"  [{mem_type}] {key}: {content}")
        else:
            print(f"Unknown /memory subcommand: {sub}")

    def _handle_mode(self, user_input: str):
        """Handle /mode command - switch between confirm/auto mode."""
        parts = user_input.split()
        if len(parts) < 2:
            print(f"Current mode: {self.agent.config.mode}")
            print("Available modes: confirm, auto")
            return
        mode = parts[1]
        if mode in ("confirm", "auto"):
            self.agent.config.mode = mode
            if mode == "auto":
                self.agent.config.confirm_commands = False
                self.agent.config.max_iterations = 2**31 - 1
                # Re-register shell tool without confirm
                self.agent.tool_registry._tools.pop("shell", None)
                self.agent.tool_registry.register(
                    ShellTool(
                        remind_interval=self.agent.config.shell_remind_interval,
                        check_dangerous=self.agent.config.dangerous_commands_check,
                        require_confirm=False,
                        env=self.agent.config.shell_env,
                        health_judge_fn=self.agent._health_judge,
                    )
                )
            print(f"Mode set to: {mode}")
        else:
            print(f"Unknown mode: {mode}")

    def _handle_plan(self, user_input: str):
        """Handle /plan command - show active plan."""
        parts = user_input.split()
        if len(parts) < 2:
            active = self.agent.task_plan.get_active()
            if active:
                print(f"Active plan: {active.get('id', '?')}")
                for step in active.get("steps", []):
                    icon = {"pending": " ", "doing": "→", "done": "✓", "skipped": "-", "blocked": "!"}.get(step.get("status", "pending"), " ")
                    title = step.get("title", "") or step.get("description", "")
                    print(f"  [{icon}] {title}")
            else:
                print("No active plan.")
            return
        print(f"Unknown /plan subcommand: {' '.join(parts[1:])}")

    def _handle_session(self):
        """Handle /session command — display current session info."""
        agent = self.agent
        created = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(os.path.getctime(agent._session_dir))
        ) if os.path.exists(agent._session_dir) else "unknown"
        print(f"\n  Session ID:  {agent._session_id}")
        print(f"  Directory:   {agent._session_dir}")
        print(f"  Created:     {created}")
        print(f"  Turns:       {agent.turn_count}")
        print()

    def _handle_resume(self, user_input: str):
        """Handle /resume command - resume previous session.

        Supports:
          /resume         — list resumable sessions
          /resume 1       — resume by numeric index
          /resume f73eb28f — resume by session ID (prefix match)
        """
        sessions = find_resumable_sessions(self.agent._sessions_root)
        if not sessions:
            print("No resumable sessions found.")
            return
        parts = user_input.split()
        if len(parts) >= 2:
            arg = parts[1]
            target = None
            if arg.isdigit():
                # Match by numeric index
                idx = int(arg) - 1
                if 0 <= idx < len(sessions):
                    target = sessions[idx]
            else:
                # Match by session ID prefix
                for s in sessions:
                    sid = s.get("session_id", "")
                    if sid.startswith(arg) or sid[:12].startswith(arg):
                        target = s
                        break
            if target:
                data = load_conversation(target["session_dir"])
                if data:
                    self.agent._restore_session(data, target["session_dir"])
                    sid = target.get("session_id", "?")[:12]
                    print(f"Resumed session {sid} ({target.get('user_turns', 0)} turns)")
                    return
                else:
                    print(f"Failed to load conversation from {target['session_dir']}")
                    return
            print(f"No session matching '{arg}' found.")
        for i, s in enumerate(sessions, 1):
            sid = s.get("session_id", "?")[:8]
            ts = time.strftime("%m-%d %H:%M", time.localtime(s['timestamp']))
            turns = s.get("user_turns", 0)
            summary = s.get("session_summary", "")
            if not summary:
                # No summary (forced exit) — generate from conversation on the fly
                summary = self._generate_resume_summary(s)
            print(f"  {i}. {sid}  {ts} ({turns} turns):")
            # Print summary indented
            for line in summary.strip().split("\n"):
                print(f"     {line}")
        print("\nUsage: /resume <number|session_id>")

    def _handle_reload(self, user_input: str):
        """Hot reload: save state, exec new process, auto-resume.

        /reload        — full code reload (restart process)
        /reload config — config-only reload (no restart)
        """
        parts = user_input.split()
        if len(parts) > 1 and parts[1] == "config":
            # Lightweight: just reload config and skills, no process restart
            self.agent.config.reload()
            self.agent.skill_manager.invalidate_cache()
            self.agent._refresh_system_prompt()
            print("Config and skills reloaded (no code reload).")
            return

        # Full code reload via process restart
        print("Saving session state...")
        self.agent._save_conversation(completed=False)

        session_id = self.agent._session_id
        print(f"Restarting process (session: {session_id})...")
        print("All code changes will take effect.\n")

        # Build the command to restart with auto-resume
        import sys
        import os

        # Determine how the agent was launched
        argv = sys.argv[:]

        # Inject --auto-resume flag with session_id
        # Remove any existing --auto-resume to avoid duplication
        clean_argv = [a for a in argv if not a.startswith("--auto-resume")]
        clean_argv.append(f"--auto-resume={session_id}")

        # Use os.execv to replace current process — no orphan processes
        os.execv(sys.executable, [sys.executable] + clean_argv)

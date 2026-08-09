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

"""Shell command tool — pure executor with long-command monitoring."""

import os
import re
import subprocess
import sys
import threading
import time

from flagscale_agent.react.tools.base import Tool


# --- Self-kill protection ---

_SELF_KILL_RE = re.compile(
    r"\bkill\b.*\b(flagscale|agent\.py|react/agent)\b"
    r"|\bgrep\b.*\b(flagscale|agent\.py)\b.*\bkill\b"
    r"|\bpkill\b.*\b(flagscale|agent)\b"
    r"|\bkillall\b.*\b(flagscale|agent)\b",
)


def _get_agent_pids():
    """Get PIDs of the agent process tree that must not be killed."""
    agent_pid = os.getpid()
    ppid = os.getppid()
    exclude = {agent_pid, ppid}
    try:
        with open(f"/proc/{ppid}/stat") as f:
            pppid = int(f.read().split()[3])
            exclude.add(pppid)
    except (OSError, ValueError, IndexError):
        pass
    try:
        result = subprocess.run(
            f"pgrep -P {agent_pid}", shell=True,
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            try:
                exclude.add(int(line.strip()))
            except ValueError:
                pass
    except Exception:
        pass
    return exclude


def _protect_self_kill(command: str) -> str:
    """Rewrite kill pipelines to exclude the agent's own process tree."""
    exclude = _get_agent_pids()
    pids_str = "|".join(str(p) for p in sorted(exclude))

    pkill_re = re.compile(r"\b(pkill|killall)\s+(-\S+\s+)*(flagscale\S*|agent\S*)")
    m = pkill_re.search(command)
    if m:
        signal_flag = m.group(2) or ""
        pattern = m.group(3)
        kill_sig = "-9" if "-9" in signal_flag else ""
        replacement = (
            f"ps aux | grep '{pattern}' | grep -v grep"
            f" | awk '{{print $2}}'"
            f" | grep -Ev '\\b({pids_str})\\b'"
            f" | xargs -r kill {kill_sig}"
        )
        command = command[:m.start()] + replacement + command[m.end():]
        return command

    if "xargs" in command and "kill" in command:
        pid_filter = f"grep -Ev '\\b({pids_str})\\b' | "
        command = re.sub(
            r'\|\s*xargs\s+(-r\s+)?kill',
            lambda m: f"| {pid_filter}xargs {m.group(1) or ''}kill",
            command,
        )

    return command


# --- Trailing pipe optimization ---

_TRAILING_PIPE_RE = re.compile(
    r"\|\s*(tail|head)\s+-n\s*(\d+)\s*$"
    r"|\|\s*(tail|head)\s+-(\d+)\s*$"
    r"|\|\s*(tail|head)\s*$"
)


def _strip_trailing_pipe(command: str):
    """Strip trailing | tail -N / | head -N and return (new_cmd, post_fn).

    Applies the equivalent truncation in Python so we get real-time output
    from the main command instead of buffering in tail/head.
    """
    m = _TRAILING_PIPE_RE.search(command)
    if not m:
        return command, None

    cmd_name = m.group(1) or m.group(3) or m.group(5)
    count_str = m.group(2) or m.group(4)
    count = int(count_str) if count_str else 10

    stripped = command[:m.start()].rstrip()
    stripped = re.sub(r'\s*2>&1\s*$', '', stripped)

    if cmd_name == "tail":
        def post_fn(output):
            lines = output.splitlines(True)
            return "".join(lines[-count:]) if len(lines) > count else output
    else:
        def post_fn(output):
            lines = output.splitlines(True)
            return "".join(lines[:count]) if len(lines) > count else output

    return stripped, post_fn


# --- ShellTool ---

class ShellTool(Tool):
    name = "shell"
    description = "Execute a shell command and return its output (stdout + stderr)."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
        },
        "required": ["command"],
    }

    def __init__(self, remind_interval: int = 120, env: dict = None,
                 health_judge_fn=None):
        self._remind_interval = remind_interval
        self._env = env or {}
        self._health_judge_fn = health_judge_fn

    def execute(self, **kwargs) -> str:
        command = kwargs.get("command", "")
        if not command:
            return "ERROR: 'command' parameter is required but was empty or missing (possible output truncation)."
        if not isinstance(command, str):
            return f"ERROR: shell command must be a string, got {type(command).__name__}: {repr(command)[:200]}"

        quiet = kwargs.pop("_quiet", False)

        # Self-kill protection
        if _SELF_KILL_RE.search(command):
            command = _protect_self_kill(command)

        # Trailing pipe optimization
        command, post_fn = _strip_trailing_pipe(command)

        try:
            run_env = {**os.environ, **self._env} if self._env else None
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=run_env,
            )

            stdout_chunks: list = []
            stderr_chunks: list = []

            def _read_stream(stream, buf):
                for line in stream:
                    buf.append(line)

            t_out = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_chunks), daemon=True)
            t_err = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_chunks), daemon=True)
            t_out.start()
            t_err.start()

            # --- Long-command monitoring loop ---
            start = time.time()
            next_check = min(15, self._remind_interval)
            last_output_snapshot = ""
            stall_count = 0
            health_reason = ""
            _streams_done_at = None
            _STREAM_EOF_GRACE_SECS = 3

            while proc.poll() is None:
                elapsed = time.time() - start

                if elapsed > next_check:
                    next_check = elapsed + self._remind_interval
                    mins = int(elapsed) // 60
                    secs = int(elapsed) % 60
                    time_str = f"{mins}m{secs}s" if mins > 0 else f"{secs}s"
                    recent_text = "".join(stdout_chunks[-20:] + stderr_chunks[-20:])
                    current_snapshot = "".join(stdout_chunks[-10:] + stderr_chunks[-10:])

                    # Track output changes
                    output_changed = not current_snapshot or current_snapshot != last_output_snapshot
                    if not output_changed:
                        stall_count += 1
                    else:
                        stall_count = 0
                    last_output_snapshot = current_snapshot

                    # Health judge decides kill/continue
                    if self._health_judge_fn:
                        decision = self._health_judge_fn(
                            command, recent_text, time_str,
                            output_changed=output_changed,
                            stall_count=stall_count,
                        )
                        if decision.get("kill"):
                            proc.kill()
                            t_out.join(timeout=2)
                            t_err.join(timeout=2)
                            partial = "".join(stdout_chunks) + "".join(stderr_chunks)
                            reason = decision.get("reason", "Unhealthy command")
                            return (
                                f"TERMINATED: {reason} (after {time_str}).\n"
                                f"Output:\n{partial}"
                            )
                        else:
                            reason = decision.get("reason", "")
                            health_reason = reason
                            if reason and not quiet:
                                from flagscale_agent.react import display
                                if hasattr(display, '_active_spinner') and display._active_spinner:
                                    display._active_spinner.set_hint(f"🩺 {reason}")
                            # LLM decides next check interval
                            ncs = decision.get("next_check_seconds")
                            if isinstance(ncs, (int, float)) and 10 <= ncs <= 300:
                                next_check = elapsed + ncs
                            else:
                                next_check = elapsed + self._remind_interval

                    # Display progress for long-running commands
                    if not quiet:
                        recent = stdout_chunks[-5:] + stderr_chunks[-5:]
                        if recent:
                            from flagscale_agent.react import display
                            if hasattr(display, '_active_spinner') and display._active_spinner:
                                display._active_spinner.stop()
                            health_note = f"\n   🩺 {health_reason}\n" if health_reason else ""
                            lines_out = [f"\033[2m   ⏳ [{time_str}]{health_note}   Recent output:\033[0m"]
                            for line in recent[-5:]:
                                lines_out.append(f"\033[2m   │ {line.rstrip()}\033[0m")
                            if hasattr(display, '_stdout_lock'):
                                with display._stdout_lock:
                                    sys.stdout.write("\n".join(lines_out) + "\n")
                                    sys.stdout.flush()
                            else:
                                sys.stdout.write("\n".join(lines_out) + "\n")
                                sys.stdout.flush()
                            if hasattr(display, '_active_spinner') and display._active_spinner:
                                display._active_spinner.start()

                # Ctrl-C handling
                try:
                    pass  # KeyboardInterrupt is caught in outer try
                except KeyboardInterrupt:
                    proc.kill()
                    proc.wait(timeout=3)
                    return f"TERMINATED by user after {int(elapsed)}s."

                # EOF grace kill — prevent zombie processes
                if not t_out.is_alive() and not t_err.is_alive():
                    if _streams_done_at is None:
                        _streams_done_at = time.time()
                    elif time.time() - _streams_done_at > _STREAM_EOF_GRACE_SECS:
                        proc.kill()
                        proc.wait(timeout=3)
                        break
                else:
                    _streams_done_at = None

                time.sleep(0.2)

            t_out.join(timeout=5)
            t_err.join(timeout=5)

            # --- Post-execution: assemble result ---
            output = ""
            if stdout_chunks:
                output += "".join(stdout_chunks)
            if stderr_chunks:
                output += "".join(stderr_chunks)
            if not output:
                output = "(no output)"
            if post_fn and output != "(no output)":
                output = post_fn(output)
            return output

        except KeyboardInterrupt:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=3)
            partial = "".join(stdout_chunks) + "".join(stderr_chunks) if (
                "stdout_chunks" in locals() and "stderr_chunks" in locals()
            ) else ""
            raise
        except Exception as e:
            return f"ERROR: {e}"

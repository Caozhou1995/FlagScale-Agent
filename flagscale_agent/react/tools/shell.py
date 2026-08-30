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

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time

from flagscale_agent.react.tools.base import Tool


# Max seconds to wait for the health-judge LLM call before giving up on it for
# this check. The judge is a best-effort advisory; if the gateway stalls we must
# NOT let it freeze the monitor loop (which would suppress the heartbeat and the
# hard-indicator checks). On timeout we treat the judge as "no opinion" and let
# the loop continue emitting its heartbeat on schedule.
_HEALTH_JUDGE_TIMEOUT_SECS = 20

# get_process_health walks the entire child-process tree with per-process /proc
# reads. Under a huge parallel build the tree explodes and a single sample can
# block for minutes on slow /proc IO. Cap it well under the ~30s check interval so
# the heartbeat never starves; on timeout we fall back to a neutral reading.
_PROCESS_HEALTH_TIMEOUT_SECS = 5


def _run_health_judge_bounded(fn, args, kwargs, timeout=_HEALTH_JUDGE_TIMEOUT_SECS):
    """Call the health-judge fn in a worker thread with a hard timeout.

    Returns the judge's decision dict, or None if it timed out / raised. The
    worker thread is daemonized, so a genuinely wedged LLM call cannot block
    process exit — it is abandoned, not joined indefinitely.
    """
    result = {}

    def _worker():
        try:
            result["value"] = fn(*args, **kwargs)
        except Exception:
            result["value"] = None

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None  # judge stalled — abandon it, keep the loop alive
    return result.get("value")


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
        # Per-prefix command history for the LLM health judge. Maps a command
        # prefix (first 2 tokens) to a list of (outcome, duration_secs) tuples.
        # outcome is "completed", "killed", or "error".
        self._command_history: dict[str, list[tuple[str, float]]] = {}
        # Container resources cached once (memory, cpu count).
        self._cached_resources: str | None = None

    @staticmethod
    def _cmd_prefix(command: str) -> str:
        """Extract a 2-token prefix from a command for history grouping."""
        toks = command.strip().split()[:2]
        return " ".join(toks)

    def _record_history(self, command: str, outcome: str, duration: float):
        """Record the outcome of a command execution for the LLM judge."""
        prefix = self._cmd_prefix(command)
        if prefix not in self._command_history:
            self._command_history[prefix] = []
        self._command_history[prefix].append((outcome, duration))
        # Keep only the last 10 entries per prefix to bound memory.
        if len(self._command_history[prefix]) > 10:
            self._command_history[prefix] = self._command_history[prefix][-10:]

    def _build_history_str(self, command: str) -> str:
        """Build a human-readable history summary for the LLM judge."""
        prefix = self._cmd_prefix(command)
        entries = self._command_history.get(prefix, [])
        if not entries:
            return ""
        parts = []
        for outcome, dur in entries:
            parts.append(f"{outcome} in {dur:.0f}s")
        return f"{prefix}: {len(entries)} prior run(s) — {', '.join(parts)}"

    def _detect_resources(self) -> str:
        """Detect container resources (memory, CPU count) for the LLM judge."""
        if self._cached_resources is not None:
            return self._cached_resources
        try:
            import multiprocessing
            nproc = multiprocessing.cpu_count()
            # Read memory from /proc/meminfo (Linux containers)
            mem_kb = None
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            mem_kb = int(line.split()[1])
                            break
            except Exception:
                pass
            if mem_kb:
                mem_gb = mem_kb / (1024 * 1024)
                self._cached_resources = f"Memory: {mem_gb:.0f}GB, CPU: {nproc} cores"
            else:
                self._cached_resources = f"CPU: {nproc} cores (memory unknown)"
        except Exception:
            self._cached_resources = ""
        return self._cached_resources

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
            run_env = {**os.environ, **self._env} if self._env else dict(os.environ)
            # Encourage line-buffered output from children so long-running
            # commands (make/gcc/etc.) stream progress incrementally instead of
            # full-buffering into the pipe and dumping everything only at exit.
            run_env.setdefault("PYTHONUNBUFFERED", "1")

            # Prefer `stdbuf -oL -eL` when available: it sets _STDBUF_* / LD_PRELOAD
            # env vars that propagate to ALL descendants, forcing line-buffered
            # stdout/stderr transitively (make, gcc, ...). Fall back to a plain
            # shell=True invocation when stdbuf is missing.
            stdbuf = shutil.which("stdbuf")
            if stdbuf:
                popen_args = [stdbuf, "-oL", "-eL", "/bin/sh", "-c", command]
                popen_kwargs = {"shell": False}
            else:
                popen_args = command
                popen_kwargs = {"shell": True}

            proc = subprocess.Popen(
                popen_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=run_env,
                **popen_kwargs,
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
            # Default cadence is 30s; honor a smaller remind_interval so callers
            # (and tests) can request tighter checks. With the default
            # remind_interval (120) this is 30s.
            check_interval = min(30, self._remind_interval)
            next_check = check_interval
            last_output_snapshot = ""
            stall_count = 0
            health_reason = ""
            _streams_done_at = None
            _STREAM_EOF_GRACE_SECS = 3
            _had_children = False  # Track if process ever had children
            _prev_num_children = 0  # Track child count drops (OOM signal)
            _prev_cpu_time = None   # Cumulative CPU secs (liveness signal A)
            _prev_io_bytes = None   # Cumulative IO bytes (liveness signal A)
            _prev_rss_mb = None     # RSS MB (liveness signal A)

            while proc.poll() is None:
                elapsed = time.time() - start

                if elapsed > next_check:
                    next_check = elapsed + check_interval
                    mins = int(elapsed) // 60
                    secs = int(elapsed) % 60
                    time_str = f"{mins}m{secs}s" if mins > 0 else f"{secs}s"
                    recent_text = "".join(stdout_chunks[-20:] + stderr_chunks[-20:])
                    current_snapshot = "".join(stdout_chunks[-10:] + stderr_chunks[-10:])

                    # ── Heartbeat FIRST (before any health checks) ──
                    # The heartbeat must fire unconditionally every check_interval,
                    # independent of whether the health checks below block or stall.
                    # Previously the heartbeat was at the END of the health block;
                    # if get_process_health or the LLM judge blocked (even with
                    # timeouts, GIL contention from psutil C calls or API latency
                    # could delay the main thread), the heartbeat was suppressed
                    # for minutes — making a healthy long build look hung AND
                    # hiding the fact that the monitor itself was alive.
                    if not quiet:
                        from flagscale_agent.react import display
                        recent_hb = stdout_chunks[-5:] + stderr_chunks[-5:]
                        if hasattr(display, '_active_spinner') and display._active_spinner:
                            display._active_spinner.stop()
                        health_note_hb = f"\n   🩺 {health_reason}\n" if health_reason else ""
                        if recent_hb:
                            header_hb = f"\033[2m   ⏳ [{time_str}]{health_note_hb}   Recent output:\033[0m"
                            lines_out_hb = [header_hb]
                            for line_hb in recent_hb[-5:]:
                                lines_out_hb.append(f"\033[2m   │ {line_hb.rstrip()}\033[0m")
                        else:
                            lines_out_hb = [
                                f"\033[2m   ⏳ [{time_str}]{health_note_hb}   "
                                f"(running, no output yet)\033[0m"
                            ]
                        if hasattr(display, '_stdout_lock'):
                            with display._stdout_lock:
                                sys.stdout.write("\n".join(lines_out_hb) + "\n")
                                sys.stdout.flush()
                        else:
                            sys.stdout.write("\n".join(lines_out_hb) + "\n")
                            sys.stdout.flush()
                        if hasattr(display, '_active_spinner') and display._active_spinner:
                            display._active_spinner.start()

                    # Track output changes. Empty output counts as a STALL,
                    # not as "changed" — otherwise commands that never stream
                    # output (piped through tail/grep, silent network waits)
                    # look permanently healthy and stall indicators never fire.
                    output_changed = bool(current_snapshot) and current_snapshot != last_output_snapshot
                    if not output_changed:
                        stall_count += 1
                    else:
                        stall_count = 0
                    last_output_snapshot = current_snapshot

                    # Hard-indicator check (priority)
                    from flagscale_agent.react.tools.process_health import (
                        get_process_health,
                        detect_output_anomalies,
                        should_kill_process,
                    )

                    # Bound the health sample. get_process_health walks the whole
                    # child-process tree (children(recursive=True) + per-proc
                    # cpu_times()/io_counters() /proc reads). Under a make -j / opam
                    # coq build the tree explodes to 100s-1000s of short procs and a
                    # single call can block MINUTES on /proc reads while the disk/CPU
                    # is saturated — starving the heartbeat (only emitted between
                    # samples). Cap it like the judge: on timeout, fall back to a
                    # neutral "alive, working" reading so the loop keeps beating and
                    # never false-kills on a slow sample.
                    proc_health = _run_health_judge_bounded(
                        get_process_health, (proc.pid,), {},
                        timeout=_PROCESS_HEALTH_TIMEOUT_SECS,
                    )
                    if proc_health is None:
                        # Neutral "alive & working" reading. Bump cpu_time by a
                        # nonzero amount so the liveness veto (support A) treats a
                        # slow sample as progress, never as a stall → no false kill.
                        _base_cpu = _prev_cpu_time if _prev_cpu_time is not None else 0.0
                        _base_io = _prev_io_bytes if _prev_io_bytes is not None else 0.0
                        proc_health = {
                            'alive': True, 'zombie': False,
                            'cpu_percent': 1.0, 'memory_mb': _prev_rss_mb or 0.0,
                            'num_children': _prev_num_children,
                            'children_alive': max(1, _prev_num_children),
                            'cpu_time': _base_cpu + 1.0, 'io_bytes': _base_io,
                        }
                    output_anomalies = detect_output_anomalies(recent_text)
                    
                    # Track if we ever had children
                    if proc_health['num_children'] > 0:
                        _had_children = True

                    # Liveness deltas across sampling points (support A):
                    # cumulative CPU time / IO bytes are monotonic counters, so
                    # a positive delta proves the process is actually working —
                    # this vetoes heuristic kills. None on the first sample.
                    if _prev_cpu_time is None:
                        progress_signals = None  # no baseline yet
                    else:
                        progress_signals = {
                            'cpu_time_delta': proc_health.get('cpu_time', 0.0) - _prev_cpu_time,
                            'io_bytes_delta': proc_health.get('io_bytes', 0.0) - _prev_io_bytes,
                            'rss_delta': (proc_health.get('memory_mb', 0.0) - _prev_rss_mb) * 1024 * 1024,
                        }
                    _prev_cpu_time = proc_health.get('cpu_time', 0.0)
                    _prev_io_bytes = proc_health.get('io_bytes', 0.0)
                    _prev_rss_mb = proc_health.get('memory_mb', 0.0)

                    should_kill, kill_reason = should_kill_process(
                        elapsed, output_changed, stall_count,
                        proc_health, output_anomalies, _had_children,
                        _prev_num_children,
                        progress_signals=progress_signals,
                    )
                    
                    # Update child count for next iteration
                    _prev_num_children = proc_health['num_children']

                    if should_kill:
                        proc.kill()
                        t_out.join(timeout=2)
                        t_err.join(timeout=2)
                        partial = "".join(stdout_chunks) + "".join(stderr_chunks)
                        self._record_history(command, "killed", time.time() - start)
                        return (
                            f"TERMINATED: {kill_reason} (after {time_str}).\n"
                            f"This was stopped by the health monitor, not by "
                            f"the command itself. Do not relaunch a variant of "
                            f"the same approach — treat the reason above as a "
                            f"redirection and change your method class before "
                            f"running anything similar again.\n"
                            f"Output:\n{partial}"
                        )

                    # Advisory from hard indicators (e.g. silent stall) —
                    # pass as context to the LLM judge, which has richer
                    # information (command history, container resources,
                    # output pattern) to make a better-informed decision.
                    if kill_reason:
                        health_reason = kill_reason

                    # Fallback: LLM judge (if hard indicators passed).
                    # Pass the live resource signals so the judge can tell a
                    # genuinely hung process (no output + no CPU/child work)
                    # apart from a healthy silent one (no output but actively
                    # computing) instead of guessing from silence alone.
                    if self._health_judge_fn:
                        activity = (
                            f"CPU {proc_health['cpu_percent']:.0f}%, "
                            f"memory {proc_health['memory_mb']:.0f} MB, "
                            f"live child processes {proc_health['children_alive']}"
                            f" of {proc_health['num_children']}"
                        )
                        # Bounded call: a stalled LLM gateway must never freeze
                        # the monitor loop. On timeout we get None and skip the
                        # judge this round, so the heartbeat below still fires on
                        # schedule and hard-indicator checks keep running.
                        decision = _run_health_judge_bounded(
                            self._health_judge_fn,
                            (command, recent_text, time_str),
                            {
                                "output_changed": output_changed,
                                "stall_count": stall_count,
                                "activity": activity,
                                "command_history": self._build_history_str(command),
                                "container_resources": self._detect_resources(),
                            },
                        ) or {"kill": False}
                        if decision.get("kill"):
                            proc.kill()
                            t_out.join(timeout=2)
                            t_err.join(timeout=2)
                            partial = "".join(stdout_chunks) + "".join(stderr_chunks)
                            reason = decision.get("reason", "Unhealthy command")
                            self._record_history(command, "killed", time.time() - start)
                            return (
                                f"TERMINATED: {reason} (after {time_str}).\n"
                                f"This was stopped by the health monitor, not by "
                                f"the command itself. Do not relaunch a variant of "
                                f"the same approach — treat the reason above as a "
                                f"redirection and change your method class before "
                                f"running anything similar again.\n"
                                f"Output:\n{partial}"
                            )
                        else:
                            reason = decision.get("reason", "")
                            health_reason = reason
                            if reason and not quiet:
                                from flagscale_agent.react import display
                                if hasattr(display, '_active_spinner') and display._active_spinner:
                                    display._active_spinner.set_hint(f"🩺 {reason}")

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
            elapsed_total = time.time() - start
            output = ""
            if stdout_chunks:
                output += "".join(stdout_chunks)
            if stderr_chunks:
                output += "".join(stderr_chunks)
            if not output:
                output = "(no output)"
            if post_fn and output != "(no output)":
                output = post_fn(output)

            # Record command outcome for future health-judge context.
            outcome = "completed"
            if output.startswith("TERMINATED"):
                outcome = "killed"
            elif output.startswith("ERROR"):
                outcome = "error"
            self._record_history(command, outcome, elapsed_total)

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

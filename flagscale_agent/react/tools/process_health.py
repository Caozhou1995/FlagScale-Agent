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

"""Process health detection for shell command monitoring."""

import psutil


def get_process_health(proc_pid: int) -> dict:
    """
    Get process health status using psutil.
    
    Returns:
        {
            'alive': bool,
            'zombie': bool,
            'cpu_percent': float,
            'memory_mb': float,
            'num_children': int,
            'children_alive': int,
        }
    """
    try:
        p = psutil.Process(proc_pid)
        
        # Check if zombie
        status = p.status()
        is_zombie = status == psutil.STATUS_ZOMBIE
        
        # CPU and memory (skip if zombie)
        cpu = 0.0
        mem = 0.0
        if not is_zombie:
            try:
                cpu = p.cpu_percent(interval=0.1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            try:
                mem = p.memory_info().rss / 1024 / 1024
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Child processes
        try:
            children = p.children(recursive=True)
            num_children = len(children)
            alive_children = sum(1 for c in children if c.is_running())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            num_children = 0
            alive_children = 0
        
        return {
            'alive': p.is_running(),
            'zombie': is_zombie,
            'cpu_percent': cpu,
            'memory_mb': mem,
            'num_children': num_children,
            'children_alive': alive_children,
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {
            'alive': False,
            'zombie': False,
            'cpu_percent': 0.0,
            'memory_mb': 0.0,
            'num_children': 0,
            'children_alive': 0,
        }


def detect_output_anomalies(recent_output: str) -> dict:
    """
    Detect anomalies in command output.
    
    Returns:
        {
            'oom_killed': bool,
            'killed_count': int,
            'repeated_errors': bool,
        }
    """
    lines = recent_output.split('\n')
    
    # Count "Killed" messages
    killed_count = sum(1 for line in lines if 'Killed' in line)
    
    # OOM indicators
    oom_indicators = [
        'Error 137',
        'Out of memory',
        'MemoryError',
        'cannot allocate memory',
        'OOM killer',
    ]
    oom_killed = any(indicator in recent_output for indicator in oom_indicators)
    
    # Repeated errors (same error 3+ times)
    error_lines = [l for l in lines if 'error' in l.lower() or 'Error' in l]
    repeated_errors = len(error_lines) >= 3 and len(set(error_lines)) <= 2
    
    return {
        'oom_killed': oom_killed,
        'killed_count': killed_count,
        'repeated_errors': repeated_errors,
    }


def should_kill_process(
    elapsed_seconds: float,
    output_changed: bool,
    stall_count: int,
    proc_health: dict,
    output_anomalies: dict,
    had_children: bool,
    prev_num_children: int = 0,
) -> tuple[bool, str]:
    """
    Unified hard-indicator check. Returns (should_kill, reason).
    
    6 hard indicators:
    1. Zombie process
    2. OOM killed (2+ times in output OR >3 children disappeared)
    3. All children dead + no output + >2min
    4. CPU=0% + no output + >3min
    5. Timeout >20min
    6. Child processes suddenly disappeared (OOM killer silent signal)
    """
    
    # 1. Zombie process
    if proc_health['zombie']:
        return True, (
            "Process became zombie (all children died, parent stuck). "
            "This usually means child processes crashed but parent didn't detect it."
        )
    
    # 2. OOM killed repeatedly (output-based or child-disappearance-based)
    child_drop = max(0, prev_num_children - proc_health['num_children'])
    oom_from_output = output_anomalies['oom_killed'] and output_anomalies['killed_count'] >= 2
    oom_from_children = prev_num_children >= 3 and child_drop >= 3
    
    if oom_from_output or oom_from_children:
        reason_detail = (
            f"OOM killer terminated processes (killed_count={output_anomalies['killed_count']}, "
            f"child_drop={child_drop} from {prev_num_children}→{proc_health['num_children']}). "
            if oom_from_children else 
            f"OOM killer terminated processes {output_anomalies['killed_count']} times. "
        )
        return True, (
            reason_detail +
            "Reduce parallelism (e.g., make -j1 instead of make -j) or increase memory."
        )
    
    # 3. All children dead + no output + >2min
    if (had_children and 
        proc_health['children_alive'] == 0 and 
        not output_changed and 
        elapsed_seconds > 120):
        return True, (
            f"All {proc_health['num_children']} child processes exited "
            f"but parent still running with no output for {int(elapsed_seconds)}s. "
            "Build likely failed silently."
        )
    
    # 4. CPU=0% + no output + stall_count>=6 (3min at 30s interval)
    if (proc_health['cpu_percent'] < 0.5 and 
        not output_changed and 
        stall_count >= 6 and
        elapsed_seconds > 180):
        return True, (
            f"Process using 0% CPU with no output for {stall_count * 30}s. "
            "Command appears stuck or waiting indefinitely."
        )
    
    # 5. Timeout >20min
    if elapsed_seconds > 1200:  # 20 minutes
        return True, (
            f"Command exceeded 20-minute timeout (ran {int(elapsed_seconds)}s). "
            "No single command should take this long without results."
        )
    
    return False, ""

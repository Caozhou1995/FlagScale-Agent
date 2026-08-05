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

"""Shared utilities for guards.

Includes:
- Two-phase detection helpers (classify_fn)
- Launch command detection (_is_flagscale_launch_command)
- Read-only tool classification
"""

from __future__ import annotations

import re


# Tools that only read state and never modify anything.
# Used by guards to distinguish exploratory actions from mutations.
READ_ONLY_TOOLS = frozenset({
    "read_file", "memory_read", "memory_list", "evict_list", "recall",
    "load_skill", "load_knowledge", "plan_status",
    "inspect_checkpoint", "validate_config", "web_fetch", "flagscale_train_monitor",
})

# Source constants — must match judge.py
SOURCE_LLM = "llm"
SOURCE_FAST = "fast"
SOURCE_CACHE = "cache"
SOURCE_DEFAULT = "default"
SOURCE_UNAVAILABLE = "unavailable"

_TRUSTED_SOURCES = frozenset({SOURCE_LLM, SOURCE_FAST, SOURCE_CACHE})


def get_judge_result(classify_fn, category: str, context: dict, default=None):
    """Call classify_fn and return (value, source) tuple.

    Shared utility for all guards that use two-phase detection.
    Returns (default, SOURCE_UNAVAILABLE) on any failure.
    """
    try:
        judge = getattr(classify_fn, "__self__", None)
        if judge and hasattr(judge, "classify_traced"):
            return judge.classify_traced(category, context, default)

        result = classify_fn(category, context, default=default)
        if (isinstance(result, tuple) and len(result) == 2
                and isinstance(result[1], str)
                and result[1] in (SOURCE_FAST, SOURCE_LLM, SOURCE_CACHE,
                                  SOURCE_DEFAULT, SOURCE_UNAVAILABLE)):
            return result
        return (result, SOURCE_LLM)
    except Exception:
        return (default, SOURCE_UNAVAILABLE)


def is_trusted(source: str) -> bool:
    """Check if a classify source is trustworthy (not a fallback default)."""
    return source in _TRUSTED_SOURCES


# ---------------------------------------------------------------------------
# Launch command detection
# ---------------------------------------------------------------------------

def _is_flagscale_launch_command(cmd: str) -> bool:
    """Detect FlagScale training launch commands.

    Supports compound commands (cd xxx && flagscale train ...).
    Strips quoted content to avoid grep "flagscale train" false positives.
    """
    if not isinstance(cmd, str):
        return False

    cmd_lower = cmd.lower()

    # Remove quoted content to avoid false positives like grep "flagscale train"
    cleaned = re.sub(r'''["'][^"']*["']''', '', cmd_lower)

    # Pattern 1: flagscale train <model>
    if "flagscale train " in cleaned:
        non_run_flags = ("--stop", "--dryrun", "--test", "--query", "--tune")
        if any(flag in cleaned for flag in non_run_flags):
            return False
        return True

    # Pattern 2: flagscale run ...
    if "flagscale run " in cleaned:
        non_run_actions = ("--action dryrun", "--action stop", "--action test",
                          "--action query", "--action auto_tune",
                          "-a dryrun", "-a stop", "-a test")
        if any(a in cleaned for a in non_run_actions):
            return False
        return True

    # Pattern 3: python[3] run.py ... action=run
    if ("python" in cleaned and "run.py" in cleaned
            and ("--config-name" in cleaned or "--config-path" in cleaned)
            and "action=run" in cleaned):
        return True

    return False

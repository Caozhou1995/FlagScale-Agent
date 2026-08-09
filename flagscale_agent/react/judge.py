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

"""LLM Judge — cached LLM classification for shell safety and health monitoring."""

from __future__ import annotations

import hashlib
import json
from typing import Any


# ── Prompts ───────────────────────────────────────────────────────────────────

_CLASSIFY_PROMPTS: dict[str, str] = {
    "is_fatal": """\
Is this shell command FATALLY DESTRUCTIVE — causing IRREVERSIBLE catastrophic damage?

Command: {command}

Answer YES ONLY for commands that would:
- Destroy entire filesystems (rm -rf /, rm -rf /*, rm -rf ~)
- Format/overwrite disks (mkfs, dd if=/dev/zero of=/dev/sd*)
- Fork bombs (:(){ :|:& };:)
- Wipe all data on the system
- Brick the operating system

Answer NO for: targeted file deletion, package removal, process killing, git operations,
permission changes — these are risky but NOT catastrophic.
Reply ONLY: {{"real": true/false}}""",

    "is_dangerous": """\
Is this shell command DANGEROUS and should be BLOCKED?

Command: {command}

Answer YES for: rm -rf on system paths (/ or ~), chmod 777 on system dirs,
fork bombs, mkfs, dd without clear target, redirects to /dev/sd*.
Answer NO for: normal file operations, package management, regular shell commands.
Reply ONLY: {{"real": true/false}}""",
}


_HEALTH_PROMPT = """\
You are monitoring a running shell command. Analyze its status and decide
whether it should continue or be terminated.

Command: {command}
Total elapsed: {elapsed}
Output changed since last check: {output_changed}
Consecutive checks with no output change: {stall_count}
Recent output:
{output}

## Phase-aware monitoring

Identify the command's current lifecycle phase and adapt your judgment:

- STARTUP (no output yet, imports loading, initializing): check frequently (10-30s).
- INSTALLING (pip/conda installing packages): moderate (60-120s). Large packages can take 3-10 minutes with zero output.
- COMPILING (gcc/nvcc/ninja building C++/CUDA extensions): very patient (120-300s).
- DOWNLOADING (wget, curl, git clone, pip downloading): moderate (30-60s). MUST show progress indicators.
- LOADING (model weights, data loading): moderate (30-60s).
- STABLE (training iterations running, loss printing regularly): relaxed (120-300s).
- ANOMALY (errors in output, repeated failures): check soon (10-15s) or kill.

## Kill criteria

Kill immediately if:
- Repeated error messages or crash signatures in output
- Network failures with no retry mechanism
- Deadlock indicators (process stuck after error, infinite retry loops)

Phase-specific patience (kill if exceeded with no progress):
- pip install "Installing collected packages": up to 10 minutes
- Source builds: up to 30 minutes
- git clone/fetch: up to 5 minutes IF progress shown, else 2 minutes
- wget/curl: up to 5 minutes IF progress shown, else 1 minute
- conda: up to 10 minutes

When uncertain about install/compile: increase next_check_seconds.
When uncertain about network: KILL — network hangs don't self-resolve.

Reply ONLY: {{"kill": true/false, "reason": "...", "next_check_seconds": <int 10-300>}}"""


# ── Judge ─────────────────────────────────────────────────────────────────────

class Judge:
    """Cached LLM classification for shell safety and health monitoring."""

    def __init__(self, provider):
        self.provider = provider
        self._cache: dict[str, Any] = {}

    def classify(self, category: str, context: dict, default: Any = False) -> Any:
        """Classify via LLM. Returns bool for safety categories."""
        cache_key = self._key(category, context)
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt_template = _CLASSIFY_PROMPTS.get(category)
        if not prompt_template:
            return default

        prompt = prompt_template
        for key, val in context.items():
            placeholder = "{" + key + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(val))

        data = self._call_and_parse(prompt)
        result = self._extract_bool(data, default)
        self._cache[cache_key] = result
        return result

    def health(
        self, command: str, recent_output: str, elapsed: str,
        output_changed: bool = True, stall_count: int = 0,
    ) -> dict:
        """Evaluate whether a long-running command is healthy."""
        prompt = _HEALTH_PROMPT.format(
            command=command, elapsed=elapsed,
            output=recent_output[-2000:],
            output_changed="yes" if output_changed else "no",
            stall_count=stall_count,
        )
        return self._call_and_parse(prompt) or {"kill": False}

    def reset_turn(self):
        """Clear per-turn cache."""
        self._cache.clear()

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_bool(data: Any, default: Any) -> Any:
        """Extract boolean from LLM response dict."""
        if not isinstance(data, dict):
            return default
        real = data.get("real")
        if isinstance(real, bool):
            return real
        decision = data.get("decision")
        if isinstance(decision, bool):
            return decision
        if isinstance(decision, str):
            return decision.lower() in ("yes", "true", "y")
        return default

    def _call_and_parse(self, prompt: str) -> dict | list:
        """Make LLM call and parse JSON response."""
        text = self._call(prompt)
        if not text:
            return {}
        return self._parse_json(text)

    def _call(self, prompt: str) -> str:
        """Dispatch LLM call through provider."""
        try:
            response = self.provider.chat(
                [{"role": "user", "content": prompt}], tools=[]
            )
            return (response.get("content") or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _parse_json(text: str) -> dict | list:
        """Extract JSON from LLM response text."""
        text = text.strip()
        if text.startswith("{"):
            end = text.rfind("}")
            if end > 0:
                try:
                    return json.loads(text[:end + 1])
                except json.JSONDecodeError:
                    pass
        if text.startswith("["):
            end = text.rfind("]")
            if end > 0:
                try:
                    return json.loads(text[:end + 1])
                except json.JSONDecodeError:
                    pass
        # Fallback: find JSON bounds
        for first, last in [("{", "}"), ("[", "]")]:
            start = text.find(first)
            end = text.rfind(last)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return {}

    @staticmethod
    def _key(category: str, context: dict) -> str:
        raw = category + json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()[:16]

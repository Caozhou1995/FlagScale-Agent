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

"""Constants and configuration for FlagScale Agent.

Extracted from agent.py to reduce file size and improve maintainability.
"""

# ── Tool Sets ─────────────────────────────────────────────────────────────────

READ_ONLY_TOOLS = {
    "read_file", "grep", "find", "ls", "list_files",
    "memory_read", "memory_list", "plan_status", "web_fetch",
}

# ── Tool Behavior Configuration ───────────────────────────────────────────────

READ_FILE_SUMMARY_THRESHOLD = 8000

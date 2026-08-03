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

"""FileToolGuard — prevents write_file truncation and helps with large file reading.

Two responsibilities:
1. Write Length Check: Before write_file with large content, inject a reminder
   to use append mode for content > 3000 chars. Also detects when path is missing
   (sign of output truncation) and provides recovery guidance.
2. Read Efficiency: After read_file on a large file (500 line limit hit),
   suggest using summarize patterns or targeted reads.

NOTE: The truncation issue is at the LLM output token level — the tool CALL
itself gets truncated because the content parameter is too long for a single
LLM response. The fix is behavioral: split writes proactively.
"""

import os

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class FileToolGuard(Guard):
    """Guard against file tool misuse patterns."""

    name = "file_tool"
    priority = 40  # Low priority — informational

    def __init__(self):
        self._large_file_warned: set[str] = set()
        self._truncation_count = 0

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Check write_file content length and warn about splitting."""
        if ctx.tool_name == "write_file":
            content = ctx.tool_args.get("content", "")
            path = ctx.tool_args.get("path", "")
            mode = ctx.tool_args.get("mode", "write")

            # Detect missing path — strong sign of output truncation
            if not path:
                self._truncation_count += 1
                return GuardVerdict.inject(
                    f"[FileTool] CRITICAL: write_file called with empty path — "
                    f"this means the LLM output was truncated before the tool call "
                    f"could be fully formed. The content was too long for a single response. "
                    f"RECOVERY: Split the document into chunks of ≤2500 chars each. "
                    f"Write chunk 1 with mode='write', then append remaining chunks. "
                    f"Plan the split BEFORE generating content — do NOT attempt to write "
                    f"the full document in one shot. "
                    f"(Truncation count this session: {self._truncation_count})",
                    reason="truncation_detected_no_path",
                )

            # Only warn for initial writes (not appends) with large content
            if mode == "write" and len(content) > 4000:
                # Check if content looks truncated (ends mid-line or mid-string)
                if self._looks_truncated(content):
                    self._truncation_count += 1
                    return GuardVerdict.inject(
                        f"[FileTool] WARNING: Content appears truncated "
                        f"({len(content)} chars). This file write may be "
                        f"incomplete. Split large files: write first 2500 chars "
                        f"with mode='write', then append remaining with mode='append'. "
                        f"Split at natural boundaries (## headers, function defs). "
                        f"(Truncation count this session: {self._truncation_count})",
                        reason="possible_truncation",
                    )

        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """After read_file hits limit, suggest efficient alternatives."""
        if ctx.tool_name == "read_file" and ctx.tool_result:
            path = ctx.tool_args.get("path", "")
            # Detect "truncated at N lines" message in result
            if "truncated" in ctx.tool_result.lower() or "Use start_line=" in ctx.tool_result:
                if path not in self._large_file_warned:
                    self._large_file_warned.add(path)
                    return GuardVerdict.inject(
                        f"[FileTool] Large file hit read limit. For {path}:\n"
                        f"  - To find specific functions: grep -n 'def function_name' {path}\n"
                        f"  - To read a specific range: read_file(path, start_line=X, end_line=Y)",
                        reason="large_file_efficiency",
                    )

        # Check if write_file had a path-missing error (output truncation)
        if ctx.tool_name == "write_file" and ctx.tool_result:
            if "'path' parameter is required but was empty or missing" in ctx.tool_result:
                self._truncation_count += 1
                return GuardVerdict.inject(
                    f"[FileTool] TRUNCATION FAILURE: Your write_file call was truncated — "
                    f"the content was too long and the output hit the token limit. "
                    f"DO NOT retry with the same approach. Instead:\n"
                    f"  1. Plan the document structure (list sections/headers)\n"
                    f"  2. Write section 1 (≤2500 chars) with mode='write'\n"
                    f"  3. Append each subsequent section (≤2500 chars) with mode='append'\n"
                    f"  4. Never put more than ~2500 chars of content in a single write_file call\n"
                    f"(Total truncation failures this session: {self._truncation_count})",
                    reason="truncation_recovery_guidance",
                )

            # Check if write_file produced a file smaller than expected
            if "total file size:" in ctx.tool_result:
                # Extract reported size
                import re
                m = re.search(r"total file size:\s*(\d+)", ctx.tool_result)
                if m:
                    content = ctx.tool_args.get("content", "")
                    actual_size = int(m.group(1))
                    expected_size = len(content.encode("utf-8"))
                    if actual_size < expected_size * 0.9:
                        return GuardVerdict.inject(
                            f"[FileTool] Write may be incomplete: expected ~{expected_size} "
                            f"bytes but file is {actual_size} bytes. Use read_file to verify "
                            f"the file end, then append missing content with mode='append'.",
                            reason="write_possibly_incomplete",
                        )

        return None

    @staticmethod
    def _looks_truncated(content: str) -> bool:
        """Heuristic: does content look like it was cut off?"""
        if not content:
            return False
        # Ends with incomplete string (no closing quote/paren/bracket)
        last_line = content.rstrip().split("\n")[-1] if content.strip() else ""
        # Obvious truncation markers
        if content.rstrip().endswith(("...", "\\", ",")):
            return True
        # Unbalanced brackets suggest truncation
        opens = content.count("{") + content.count("[") + content.count("(")
        closes = content.count("}") + content.count("]") + content.count(")")
        if opens - closes > 3:
            return True
        # Unmatched triple-quotes
        triple_dq = content.count('"""')
        triple_sq = content.count("'''")
        if triple_dq % 2 != 0 or triple_sq % 2 != 0:
            return True
        return False

    def reset_turn(self):
        pass

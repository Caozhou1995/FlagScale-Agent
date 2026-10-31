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

"""Static-prompt invariants that must survive EVERY edit.

Two properties are load-bearing regardless of how the prompt is worded:

1. NO GRADER LEAK — the prompt must never name a specific task's engine or
   construct (a library, a parser, a browser, a hex address). The prompt is
   general guidance; leaking one task's grader overfits every run to it.
2. FORMAT CONTRACT — the builder ``.format()``s the string with exactly
   {cwd, tools, skills, knowledge}; no other placeholder may appear.

Both are asserted against the WHOLE prompt because they are global invariants
(not section-scoped). Concept/section presence is covered by
``test_prompt_semantics.py``.
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_prompt_leaks_no_grader_or_specifics():
    """GENERAL posture: no task's grader, engine, or construct may be hardcoded."""
    p = SYSTEM_PROMPT_STATIC
    for bad in ("BeautifulSoup", "bs4", "html.parser", "html5lib", "lxml",
                "str(soup)", "selenium", "chrome"):
        assert bad not in p, f"prompt leaked {bad!r}"
    assert not re.search(r"0x[0-9a-fA-F]{4,}", p), "prompt leaked a hex address"


def test_prompt_placeholders_are_exactly_the_format_contract():
    """Only cwd/knowledge/skills/tools may be .format placeholders."""
    assert set(re.findall(r"{([a-z_]+)}", SYSTEM_PROMPT_STATIC)) == {
        "cwd", "knowledge", "skills", "tools"}

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

"""HARD CONSTRAINT: no authored harness content may encode a task's GRADER.

The prompt already had this rule (``test_no_grader_oracle_leak_in_prompt``),
but that guard scanned ONLY ``SYSTEM_PROMPT_STATIC``.  A guard module that
injects text into the agent's context is just as much "authored harness
content" as the prompt is -- and a leak there hands the agent the checker just
the same, inviting it to satisfy the sampled check instead of the real need.

This is a POSITIVE test over a NEGATIVE property: it asserts the absence of a
grader oracle (a parser library named as the comparison oracle, a concrete
normalization chain, or a verbatim task-text grant) from every harness module
that shapes agent behaviour.  It is generic -- it hardcodes no task, no corpus,
no expected output; it forbids them.
"""
import re
from pathlib import Path

import pytest

# Modules that shape agent behaviour: any text they emit reaches the agent and
# can therefore leak a checker.  ``knowledge/`` docs are reference material (not
# injected per-task) and are excluded; ``web_fetch.py`` LEGITIMATELY parses HTML
# to render a page and is the single sanctioned parser consumer.
_HARNESS_ROOT = Path(__file__).resolve().parent.parent / "flagscale_agent" / "react"

# Grader-oracle tokens: naming one of these as THE comparison oracle tells the
# agent which parser/serializer the checker uses.  A general engine name is fine
# in prose ("a parser library may differ"); binding the ORACLE to a concrete
# third-party symbol is the leak.
_ORACLE_TOKENS = (
    "BeautifulSoup",
    "bs4",
    "html.parser",
    "html5lib",
    "lxml",
    "str(soup)",
)

# A concrete grader normalization chain: whitespace/entity stripping composed
# to reproduce a checker's comparison.  Matched structurally, not by task.
_NORMALIZE_CHAIN = re.compile(
    r"replace\(\s*['\"] .*?replace\(",
)

# A verbatim task-text grant quoted into harness content: the task's OWN words
# used as an example is a leak (the agent reads the requirement, not the spec).
_TASK_TEXT_GRANTS = (
    "except for normalization that may occur",
    "normalization that may occur during",
)

# Only this module is allowed to import a parser as a real function it performs.
_SANCTIONED = {"tools/web_fetch.py"}


def _harness_files():
    for path in sorted(_HARNESS_ROOT.rglob("*.py")):
        rel = path.relative_to(_HARNESS_ROOT).as_posix()
        if rel in _SANCTIONED:
            continue
        yield rel, path


def test_no_grader_oracle_token_in_harness_modules():
    """No behaviour-shaping harness module may name a parser as its oracle."""
    offenders = []
    for rel, path in _harness_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for tok in _ORACLE_TOKENS:
            if tok in text:
                offenders.append(f"{rel}: {tok!r}")
    assert not offenders, (
        "harness module leaked a parser-oracle token (hands the agent the "
        "checker):\n  " + "\n  ".join(offenders)
    )


def test_no_grader_normalization_chain_in_harness_modules():
    """No harness module may reproduce a checker's comparison chain."""
    offenders = []
    for rel, path in _harness_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if _NORMALIZE_CHAIN.search(text):
            offenders.append(rel)
    assert not offenders, (
        "harness module reproduced a grader normalization chain:\n  "
        + "\n  ".join(offenders)
    )


def test_no_verbatim_task_grant_in_harness_modules():
    """No harness module may quote a task's own requirement text."""
    offenders = []
    for rel, path in _harness_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for grant in _TASK_TEXT_GRANTS:
            if grant in text:
                offenders.append(f"{rel}: {grant!r}")
    assert not offenders, (
        "harness module quoted a verbatim task-text grant:\n  "
        + "\n  ".join(offenders)
    )


def test_sanctioned_parser_consumer_is_only_web_fetch():
    """The test's own allowlist must not silently swallow the whole surface.

    A positive control: if every file were excluded, the three negative tests
    above would pass vacuously.  Assert the scan actually covers the guard and
    tools trees (i.e. the allowlist is a single module, not a directory).
    """
    scanned = {rel for rel, _ in _harness_files()}
    assert any(rel.startswith("guard/") for rel in scanned)
    assert any(rel.startswith("tools/") for rel in scanned)
    assert "tools/web_fetch.py" not in scanned
    assert len(_SANCTIONED) == 1

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

"""Regression tests for the PRESERVE-BY-DEFAULT delivery clause.

Empirical basis (657 settled losing trials, 50.8% clean-wrong): the largest
single failure class is DELIVERY FIDELITY — a correct solution that rewrites
the input artifact (edits in place, re-encodes untouched regions, drops
sibling content) when the task wanted the original preserved alongside a new
output. The clause teaches: default to a NEW output path, keep the input
untouched, and PROVE preservation with a byte-compare before delivering.

This is prompt-only guidance, so the test pins the clause in
SYSTEM_PROMPT_STATIC (a future edit must not silently drop it) and asserts
it leaks no task-specific grader (same hard rule as the ambiguity tests).
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_preserve_by_default_clause_present():
    """The three load-bearing behaviors must each be stated."""
    p = SYSTEM_PROMPT_STATIC
    low = p.lower()
    # 1. default: NEW output path, not in-place rewrite
    assert "new output path" in low or "new path" in low, \
        "preserve-by-default: new-path guidance missing"
    # 2. untouched regions must stay untouched
    assert "untouched" in low, "preserve-by-default: untouched-region guidance missing"
    # 3. preservation must be PROVEN, not assumed
    assert re.search(r"cmp\s+|diff\s+|checksum|byte[- ]compare", low), \
        "preserve-by-default: byte-compare proof step missing"


def test_preserve_clause_is_generic_no_task_leak():
    """The clause is general posture: no task's file names, formats, or
    grader expressions may appear."""
    p = SYSTEM_PROMPT_STATIC
    for bad in ("filter.py", "weights.pt", "model_ref.xml", "input_data",
                "clean.html", "BeautifulSoup", "html.parser"):
        assert bad not in p, f"preserve clause leaked {bad!r}"


def test_preserve_clause_sits_in_before_declaring_done_block():
    """Placement contract: the clause must live with the delivery-discipline
    clauses (the 'Before declaring done' block), not buried elsewhere."""
    p = SYSTEM_PROMPT_STATIC
    anchor = "Canonical round-trip over a hand-rolled transform"
    assert anchor in p
    idx_anchor = p.index(anchor)
    idx_next_block = p.index("When delivering:", idx_anchor)
    clause = p[idx_anchor:idx_next_block]
    # the preserve-by-default bullet must be between the round-trip bullet
    # and the delivering block
    assert "untouched" in clause, \
        "preserve-by-default bullet not found in the Before-declaring-done block"

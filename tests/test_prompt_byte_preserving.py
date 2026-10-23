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

"""Regression tests for the BYTE-PRESERVING TRANSFORM architecture clause.

Empirical basis (filter-js-from-html, losing in every cohort through
2026-10-23, four cohorts): the agent's filter delivered byte-EXACT output
for inputs without URL-bearing attributes but routed structured inputs
(forms, links, images) through a full parse-and-reserialize path, and the
serializer's canonical form rewrote untouched bytes: a blank line after the
doctype, attribute order, optional void-tag syntax, entity re-encoding and
indentation — five rewrite axes on 5 of 12 clean samples. The agent's own
58-case suite passed because a finite corpus cannot certify an unbounded
serializer output space. The clause teaches the DISCRIMINATOR the agent
lacked: when the checker compares preserved regions BYTE-WISE, the
architecture is surgical span editing on raw bytes, and parse-then-reserialize
is disqualified as a route — not merely "tested more".

Prompt-only guidance: pin the clause in SYSTEM_PROMPT_STATIC (a future
edit must not silently drop it) and assert it leaks no task-specific
domain detail (same hard rule as the other delivery-discipline tests).
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_byte_preserving_transform_clause_present():
    """The load-bearing behaviors must each be stated."""
    p = SYSTEM_PROMPT_STATIC
    anchor = "PRESERVE BY DEFAULT"
    idx_anchor = p.index(anchor)
    idx_next_block = p.index("Your own examples are not a held-out test set", idx_anchor)
    low = p[idx_anchor:idx_next_block].lower()
    # 1. the discriminator: byte-level comparison of preserved regions
    assert re.search(r"byte[- ]wise|byte.?(ident|level)|cmp|md5", low), \
        "byte-preserving: byte-comparison discriminator missing"
    # 2. the disqualified architecture: parse-then-reserialize when
    #    preservation is byte-level (serializer canonical form differs)
    assert re.search(r"(parse.{0,40}(re)?serial|reseriali[sz]e)", low), \
        "byte-preserving: parse-and-reserialize disqualification missing"
    # 3. the required architecture: surgical/span edit on the raw bytes
    assert re.search(r"(surgical|span)", low), \
        "byte-preserving: surgical span-edit architecture missing"
    # 4. the unbounded-output-space argument: no finite corpus certifies it
    assert re.search(r"(unbounded|no finite (test )?corpus|finite corpus)", low), \
        "byte-preserving: unbounded-serializer-output rationale missing"


def test_byte_preserving_transform_clause_is_generic_no_task_leak():
    """The clause is general posture: no task's domain terms, file names,
    element names or grader expressions may appear."""
    p = SYSTEM_PROMPT_STATIC
    for bad in ("xss", "filter.py", "clean_html", "javascript:", "onerror",
                "textarea", "doctype", "copyright"):
        assert bad not in p.lower(), f"byte-preserving clause leaked {bad!r}"


def test_byte_preserving_transform_clause_sits_in_before_declaring_done_block():
    """Placement contract: the clause must live with the delivery-discipline
    clauses (the 'Before declaring done' block), adjacent to the clause it
    arbitrates (Canonical round-trip / PRESERVE BY DEFAULT)."""
    p = SYSTEM_PROMPT_STATIC
    anchor = "PRESERVE BY DEFAULT"
    assert anchor in p
    idx_anchor = p.index(anchor)
    idx_next_block = p.index("Your own examples are not a held-out test set", idx_anchor)
    clause = p[idx_anchor:idx_next_block]
    assert "surgical" in clause.lower(), \
        "byte-preserving bullet not found in the Before-declaring-done block"

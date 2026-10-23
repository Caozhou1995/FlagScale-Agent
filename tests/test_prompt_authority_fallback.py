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

"""Regression tests for the UNVERIFIED-AUTHORITY FALLBACK delivery clause.

Empirical basis (dna-assembly, losing in every cohort through 2026-10-23):
the agent correctly identified that a vendor requirement governs the
deliverable, attempted to fetch the vendor page, hit a network failure,
tried one mirror, then delivered from RECALLED memory — and the memory
version was missing one structural element the requirement demanded
(a safety element BEFORE the operative site, not after). A zero-cost
conservative variant that satisfies every plausible reading of the
requirement existed and was skipped. The clause teaches: when the
authority cannot be reached and a zero-cost safer superset exists, take
the superset and say so; never let a recall be the sole delivery basis.

Prompt-only guidance: pin the clause in SYSTEM_PROMPT_STATIC (a future
edit must not silently drop it) and assert it leaks no task-specific
source or domain detail (same hard rule as the consumer-driven tests).
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_authority_fallback_clause_present():
    """The load-bearing behaviors must each be stated."""
    p = SYSTEM_PROMPT_STATIC
    low = p.lower()
    # 1. the unreachable-authority situation is named
    assert re.search(r"(unreachab|fetch fail|mirror down|no network)", low), \
        "authority-fallback: unreachable-source guidance missing"
    # 2. the zero-cost conservative superset is REQUIRED, not suggested
    assert re.search(r"(conservative (superset|variant|union|default|margin)|safer (variant|superset))", low), \
        "authority-fallback: conservative-superset requirement missing"
    # 3. recall-as-sole-basis is prohibited when a safer superset is free
    assert re.search(r"recal[^.]{0,80}(hypothesis|not.{0,20}(delivery|sole))", low), \
        "authority-fallback: recall-is-not-a-delivery-basis guidance missing"


def test_authority_fallback_clause_is_generic_no_task_leak():
    """The clause is general posture: no task's vendor, domain terms,
    sequence motifs, or grader expressions may appear."""
    p = SYSTEM_PROMPT_STATIC
    for bad in ("neb.com", "bsai", "primer", "clamp", "ggtctc",
                "golden gate", "overhang"):
        assert bad not in p.lower(), f"authority-fallback clause leaked {bad!r}"


def test_authority_fallback_clause_sits_in_before_declaring_done_block():
    """Placement contract: the clause must live with the delivery-discipline
    clauses (the 'Before declaring done' block)."""
    p = SYSTEM_PROMPT_STATIC
    anchor = "PRESERVE BY DEFAULT"
    assert anchor in p
    idx_anchor = p.index(anchor)
    idx_next_block = p.index("Your own examples are not a held-out test set", idx_anchor)
    clause = p[idx_anchor:idx_next_block]
    assert "authoritative source" in clause.lower(), \
        "authority-fallback bullet not found in the Before-declaring-done block"

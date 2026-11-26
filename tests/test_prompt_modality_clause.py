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

"""Anti-decay pins for the MODALITY clause (soft wording is still an assertion).

Why this file exists (measured, two consecutive single-trial runs): the agent
reached the correct form in its REASONING and then discarded it, because the
authority it consulted phrased the requirement softly. In the second run the
log holds both halves of the failure one screen apart -- the correct form
written out with the leading bases BEFORE the recognition site, and, forty
lines earlier, the sentence that released it ("not strictly required followed
by my call that it is optional"). The delivered bytes therefore omitted a
property the consumer asserts on, and every other assertion passed.

The capability this pins: a soft wording (recommended / preferred / usually /
often) is not permission. The tell is the agent's OWN sentence calling a
requirement optional -- that sentence is where a deliverable property is about
to be dropped, and it is where the consumer tends to test. Task-agnostic on
purpose: no domain nouns, so this file adds no instance leak.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
FLAT = re.sub(r"\s+", " ", SRC.read_text(encoding="utf-8"))


# ── Retired 2026-11-26: tests retargeted to the live carrier ──────────────
# The "WEIGHT IS NOT PERMISSION" block was deleted from prompt.py by an
# earlier compression pass. The six tests that stood here pinned that
# block's sentences verbatim and had been red ever since. The rule itself
# survives in §5 in a stronger, positive form, so the coverage is retargeted
# rather than dropped: a required element is never optional, whatever
# modality the source sentence used.
def test_soft_requirement_is_never_optional():
    """A soft wording releases nothing: what is required stays required."""
    assert "An element a named standard requires is never optional" in FLAT


def test_clause_stays_task_agnostic():
    """No domain nouns: a clause that leaks the instance hands over the answer.

    Same widening lesson as the DEFINITION-AXIS gate: the ban list covers the
    measured instance family, so the clause cannot smuggle it back in.
    """
    for banned in ("oligotm", "primer3", "bsai", "overhang", "anneal",
                   "dna-assembly", "plasmid", "circular",
                   "clamp", "cut site", "restriction site", "recognition site",
                   "ggtctc", "golden gate", "not strictly required",
                   "lz77", "deflate", "metacirc"):
        assert banned not in FLAT.lower(), banned

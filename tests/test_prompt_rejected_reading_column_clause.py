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

"""Anti-decay pins for the REJECTED-READING-IS-A-COLUMN requirement (Section 3 (b2)).

Why this file exists (measured terminal-bench run, reward 0): the agent swept the
design parameter exhaustively and called the real measuring tool on every
candidate. It also EXPLICITLY considered the second construction of the measured
input -- and dismissed it in one written line ("the leading tail is not part of
the region that base-pairs with the template"), then optimised under the
remaining single definition. The consumer's construction included bases the agent
had excluded, so the consumer's own per-pair spread measured 6.65 where the
agent's definition was well inside the bound. The design axis was swept; the
rejected construction was never computed as a column, and every row of the
agent's table agreed with itself.

That is a different failure from "never thought of the second reading". The
sibling file (`test_prompt_definition_axis_clause.py`) pins the
enumerate-the-axis half. This file pins the half that was still missing after the
agent DID see the axis: the rejected reading has to be COMPUTED side by side with
the accepted one, over the same candidate rows, before a row is chosen.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _flat() -> str:
    """Whitespace-normalised source so line wrapping cannot hide a clause."""
    return re.sub(r"\s+", " ", SRC.read_text(encoding="utf-8"))


FLAT = _flat()

CLAUSE_HEAD = "(b2) A REJECTED READING IS A COLUMN YOU STILL HAVE TO FILL."


def _clause() -> str:
    """The clause body, from its head to the start of the next sub-section.

    Scoped slice so the assertions below cannot be satisfied by a sentence
    that happens to live somewhere else in the prompt.
    """
    start = FLAT.find(CLAUSE_HEAD)
    assert start != -1, "the (b2) clause head is missing from the prompt"
    end = FLAT.find("(c) Could this check have gone red", start)
    assert end != -1, "the (b2) clause has no following (c) section -- prompt shape changed"
    return FLAT[start:end]


def test_clause_exists_and_precedes_the_next_subsection():
    """The clause is present and sits inside Section 3, between (b) and (c)."""
    body = _clause()
    assert len(body) > 400, f"clause body unexpectedly short: {len(body)} chars"


def test_definition_must_not_be_settled_before_optimising():
    """The failure mechanism: choosing the definition, then optimising under it."""
    body = _clause()
    assert "do not settle the definition and then optimize under it" in body
    assert "the best member of the wrong family" in body


def test_rejected_construction_must_be_a_computed_column():
    """The repaired behaviour: one table, one column per construction."""
    body = _clause()
    assert "a column per construction" in body
    assert "each cell a computed value" in body


def test_choice_rule_is_select_a_row_clearing_every_column():
    """The selection rule is intersection-first, not preference-first."""
    body = _clause()
    assert "take only rows clearing the bound in EVERY column" in body


def test_one_line_dismissal_is_named_as_the_drop_point():
    """The behavioural tell this clause targets: the one-line argument-away.

    Without this sentence the clause is generic advice; with it, it names the
    exact moment the column gets dropped, which is what the measured run did.
    """
    body = _clause()
    assert "dismissed the rejected construction in one line" in body
    assert "that line is where its column gets dropped" in body


def test_bet_only_when_no_row_clears_every_column():
    """The escape hatch is preserved: when the intersection is empty, bet."""
    body = _clause()
    assert "Bet only when no row clears every column" in body


def test_clause_stays_task_agnostic():
    """Scientific pattern only -- no task nouns that would leak the live instance.

    Applied to THIS clause body rather than the whole prompt: a clause that
    ships with any concrete instance of a live failing task teaches away the
    very step being tested.
    """
    body = _clause().lower()
    for banned in ("oligotm", "primer3", "bsai", "overhang", "anneal",
                   "dna-assembly", "plasmid", "circular", "clamp",
                   "cut site", "restriction site", "recognition site",
                   "ggtctc", "golden gate", "melting", "temperature",
                   "lz77", "deflate", "metacirc"):
        assert banned not in body, banned

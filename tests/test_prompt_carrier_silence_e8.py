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

"""E8 — a carrier's silence is not a negative finding.

Why this file exists (failure family measured 2026-11-20): a task can require a
property that the authoritative source the agent reads was never written to
carry (a design recommendation, a threshold, a naming convention). The agent
reads that source, does not find the property, and concludes "not required" --
so the requirement never enters its plan. The E5 copy was 4/4 in context during
that run and the agent's own reasoning still contained zero mentions of the
missing dimension: the gap is not obedience, it is a mis-read of what an
absence means.

E8 states the relation that was missing: absence in a carrier that does not
carry that dimension is not evidence, so the agent must change carrier or mark
the clause UNCONSULTED.

Every assertion here pins a RELATION of the shipped copy verbatim. Softening or
deleting the copy turns the corresponding test red -- that ablation is the only
thing that makes this file evidence rather than decoration.
"""

import pathlib
import re

from flagscale_agent.react import prompt as prompt_mod
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC, prompt_len, PROMPT_BUDGET

# The static prompt is wrapped across source lines; normalise whitespace before
# matching so an assertion is about the SENTENCE, not the line-wrapping.
FLAT = " ".join(SYSTEM_PROMPT_STATIC.split())
LOW = FLAT.lower()


def _e8() -> str:
    """The E8 paragraph, sliced out of the shipped copy by its own markers."""
    start = FLAT.index("E8 A CARRIER'S SILENCE")
    end = FLAT.index("Each rank exists because ignoring it lost a run.")
    return FLAT[start:end]


def test_e8_exists_and_sits_at_the_evidence_ranking():
    """E8 must be an E-ranked entry, placed with the others (not a stray §5 note).

    Placement is part of the mechanism: the ranking list is what the agent reads
    when two sources disagree, which is exactly the moment this clause has to
    fire. A clause parked elsewhere is a clause read at the wrong time.
    """
    assert "E8 A CARRIER'S SILENCE IS NOT A NEGATIVE FINDING." in FLAT
    # it must come AFTER E7 and BEFORE the list's closing line
    assert FLAT.index("E7 A named authority's call") < FLAT.index("E8 A CARRIER'S")
    assert FLAT.index("E8 A CARRIER'S") < FLAT.index(
        "Each rank exists because ignoring it lost a run.")


def test_e8_names_the_dimension_relation_not_just_diligence():
    """The load-bearing relation: silence is about the CARRIER, not the NEED.

    This is the sentence whose absence let the failure happen. A rewrite that
    keeps the heading but drops the relation ("go read the source carefully")
    must redden here.
    """
    e8 = _e8()
    assert ("only if that source was written to carry X's dimension" in e8), (
        "E8 must condition the negative finding on what the carrier was written "
        "to carry -- that condition is the whole mechanism"
    )
    assert "silence is not evidence" in e8.lower()


def test_e8_gives_the_two_exits():
    """Both escapes must be named: fetch the right carrier, or mark UNCONSULTED.

    Naming only one leaves the agent with a rule it cannot follow on a carrier
    it cannot reach -- the state that produced a fabricated answer elsewhere.
    """
    e8 = _e8()
    assert "go to the carrier that holds it" in e8
    assert "UNCONSULTED" in e8


def test_e8_forbids_the_exact_inference_that_failed():
    """The forbidden move must be stated, in the failure's own shape."""
    assert '"The source does not say X" becomes "X is not required" only if' in FLAT


def test_e8_carries_no_benchmark_literal():
    """A clause illustrated with a real task's own constants teaches nothing.

    The example IS the answer: an agent that can pattern-match the literal skips
    the reasoning the clause exists to force. Guard the abstract-only form.
    """
    e8 = _e8().lower()
    for leaked in ("clamp", "ggtctc", "bsai", "neb", "golden gate", "primer",
                   "raman", "fasttext", "sam-cell-seg", "gcode", "doom"):
        assert leaked not in e8, f"benchmark literal {leaked!r} leaked into E8"


def test_e8_does_not_swallow_the_ranking_it_lives_in():
    """Adding E8 must not delete a ranked entry (the budget's usual casualty)."""
    for head in ("E1 The task text's own named authority", "E2 The task text's own wording",
                 "E3 A recorded death", "E4 Your own constructed checker",
                 "E5 Where an authority names a variant", "E6 When a requirement is only REMEMBERED",
                 "E7 A named authority's call that errors"):
        assert head in FLAT, f"ranked entry lost while seating E8: {head!r}"


def test_budget_rise_is_a_recorded_user_ruling():
    """The envelope only moves on a RECORDED user ruling.

    This used to pin a literal (50000, then 51000), which turned every legitimate
    raise into a red test and — worse — made a genuine deficit look like a stale
    anchor.  What the ratchet actually protects is the RULE, not the number: a
    raise must leave a quoted user ruling beside the constant.  So assert the
    rule and, separately, that the constant is not below the highest floor any
    earlier ruling established (45000), then cross-check that the ruling text is
    really there.
    """
    assert PROMPT_BUDGET >= 45_000, (
        "the envelope was silently lowered below a previously recorded ruling"
    )
    src = pathlib.Path(prompt_mod.__file__).read_text(encoding="utf-8")
    head = src[: src.index("PROMPT_BUDGET =")]
    assert "USER RULING" in head, (
        "raising the budget needs a USER RULING recorded beside the constant "
        "(see prompt.py); a clause must be seated by compressing existing "
        "prose, not by growing the envelope"
    )
    # Every raise names its own date and quote: an unrecorded raise and a typo
    # are indistinguishable once the number lives alone on one line.
    assert head.count("USER RULING") >= 3, (
        "each raise must be individually recorded with its own quote"
    )
    assert prompt_len() <= PROMPT_BUDGET

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

"""Regression tests for the DELIVERED-BYTES-ARE-THE-AUDITED-BYTES discipline.

Evidence class this locks (TB dna-insert, four_m18 trial r6yhjan, reward 0):
  * The agent's recorded hedge evidence reads
    "fwd 18 nt tm=66.274 | rev 38 nt tm=62.435 ... ALL PASS, delta 3.839".
  * The delivered primers.fasta is fwd 55 nt / rev 36 nt. Recomputed with the
    grader's own construction, the delivered fragments are 16 nt and 36 nt, so
    the audited pair was never shipped. Arithmetic tell: 18 + 39-nt insert = 57
    != 55. The grader failed only `abs(fwd_tm - rev_tm) <= 5` (5.1727 > 5).

This class is invisible to every guard: the check output still says PASS, it
merely describes a revision that has been replaced. Only the agent's own
discipline can catch it, so the load-bearing fix is prompt text.
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _prompt_low() -> str:
    return " ".join(SYSTEM_PROMPT_STATIC.split()).lower()


def test_prompt_requires_checking_the_delivered_bytes():
    """The self-check must be bound to the FINAL artifact bytes, not an earlier candidate."""
    low = _prompt_low()
    assert "does my recorded check refer to the bytes i am actually shipping" in low


def test_prompt_says_a_late_edit_voids_earlier_evidence():
    """The mechanism, not just the slogan: re-editing after a check invalidates it."""
    low = _prompt_low()
    # must name the invalidation event explicitly
    assert "voids every earlier check" in low
    # and the residual symptom that makes it invisible: the old output still says PASS
    assert "still reads pass" in low


def test_prompt_requires_recomputing_numbers_from_the_delivered_file():
    """The falsifier: every recorded number must be recomputable from the shipped file."""
    low = _prompt_low()
    assert "recompute from the delivered file alone" in low
    # and the diagnostic tell when a stale audit survived
    assert "different lengths, offsets, or values" in low


def test_delivered_bytes_clause_stays_generic():
    """No originating-task specifics may leak into the generic prompt."""
    low = _prompt_low()
    for leak in ("primers.fasta", "r6yhjan", "5.1727", "oligotm", "dna-insert", "delta 3.839"):
        assert leak not in low, f"prompt leaked task-specific identifier: {leak}"
    assert not re.search(r"0x[0-9a-f]{4,}", low)

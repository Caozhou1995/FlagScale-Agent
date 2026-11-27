"""Pin the TEMPORAL dimension of the last gate.

Third round of three-task failures (dna / gcode / gpt2) shared a shape the
prompt did NOT cover: the graded number was measured BEFORE the last edit to
the delivered file.  The three axes, read verbatim from the graders:

  gpt2  assert 5075 < 5000   -- it had measured 4970 on an earlier rev
  dna   assert 0 >= 1        -- clamp never landed in the delivered primers
  gcode 'ch4LLenGng' != ...  -- correct string read early, abridged at bank

The prompt already covered the SPATIAL dimension ("enumerate the 2-4 ways a
consumer could MEASURE your artifact") and the OBJECT dimension ("self-check
reads the delivered bytes against the decisions you wrote EARLIER").  What it
lacked was WHEN: a reading taken before the last edit does not survive it.

These tests bind that one sentence so its removal turns them red.  Each test
asserts something a stranger could re-check against the delivered bytes alone,
never against a paraphrase of the rule.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flagscale_agent.react import prompt as P  # noqa: E402

S = P.SYSTEM_PROMPT_STATIC
LOW = S.lower()

GATE = "THE LAST GATE RUNS ON THE DELIVERED BYTES."
TIMING = "AFTER your last edit, not before"
HOST = "iterate only on a parallel file."


def test_timing_clause_is_present_verbatim():
    """The sentence must exist character-for-character, not paraphrased."""
    assert TIMING in S, "temporal clause missing from SYSTEM_PROMPT_STATIC"


def test_timing_clause_says_the_measurement_is_on_the_delivered_path():
    """Object AND time must be in the SAME sentence, not two clauses apart."""
    i = S.index(TIMING)
    sentence = S[S.rindex(".", 0, i) + 1 : S.index(".", i) + 1]
    assert "delivered" in sentence.lower(), sentence
    assert "measure" in sentence.lower(), sentence


def test_timing_clause_sits_inside_the_last_gate_paragraph():
    """Placement matters: a timing rule parked elsewhere is not at the gate."""
    gate = S.index(GATE)
    timing = S.index(TIMING)
    nxt = S.index("UNDERDETERMINED CHOICES.")
    assert gate < timing < nxt, (gate, timing, nxt)


def test_timing_clause_follows_the_ownership_sentence_not_replaces_it():
    """The pre-existing 'keep holding those bytes' rule must survive."""
    assert HOST in S, "the ownership half of the gate was dropped"
    assert S.index(HOST) < S.index(TIMING)


def test_timing_clause_is_not_duplicated():
    assert S.count(TIMING) == 1


def test_timing_clause_states_the_failure_mode_not_only_the_duty():
    """'not before' names the wrong reading; a bare duty can be nodded past."""
    assert "not before" in S[S.index(TIMING) - 10 : S.index(TIMING) + 80]


def test_prompt_stays_within_budget():
    assert len(S) <= P.PROMPT_BUDGET, (len(S), P.PROMPT_BUDGET)

"""The step note is a WORKING MEDIUM, not an archive (anti-decay pins).

Carrier history (both measured on dna-assembly, standard budget):

2026-11-26 13:03 trial — carrier = a plain scratch work file.
  Behaviour: the clause WAS adopted at the counter level (write_file 0 -> 4,
  plan_update 11..13 -> 0), but the write happened only at "design is settled,
  time to emit the script": of 9 bouts with >=60s of thinking, only 3/9 were
  followed by a write as the first action; the other 6 were followed by a shell
  probe. The plan channel was killed outright (plan_update = 0), so the plan
  lifecycle never advanced past pending. 2026-11-26 12:28 trial — carrier = the
  plan step note, reward = 1: the plan completed and the deliverable shipped.

So the carrier is the plan step note (the agent does use it, and it does not
cost the plan lifecycle), and the clause keeps the long-thinking case explicit
because that is where the measured loss happened: a 240s reasoning stream was
aborted (`[thinking stream ran 240s ... aborting this stream]`) with the
load-bearing results ("common prefix 214, common suffix 2040, inserted 1337")
living only in that stream.

The gap the clause fills: the prompt already said a step note persists across
eviction (SS8) and that delivered bytes must carry the transformed value (SS5).
Neither said to use the note as the MEDIUM the later code is written against,
nor that a long reasoning bout must be persisted before it can be lost.

Honest limit: on the plan-note trial the authority read (plans/plan_*.yaml,
all 8 steps `notes: ''`) shows the note was NOT used as a working medium
either. These tests pin the CLAUSE TEXT, not its behavioural uptake; uptake is
measured by reading an agent log, and by call counts that distinguish
`plan_update(notes=...)` from `step_done`/`complete`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as PROMPT  # noqa: E402

FLAT = " ".join(PROMPT.split())


def test_step_note_is_named_a_working_medium() -> None:
    assert "THE STEP NOTE IS A WORKING MEDIUM" in FLAT


def test_the_work_file_carrier_is_gone() -> None:
    """The discarded carrier must not linger as a rival instruction."""
    assert "THE WORK FILE IS YOUR WORKING MEDIUM" not in FLAT


def test_conclusions_are_written_as_they_settle() -> None:
    assert "When reasoning settles a load-bearing result" in FLAT
    assert "write it to that step's note" in FLAT


def test_code_is_written_against_the_note_not_memory() -> None:
    assert "read the step back" in FLAT
    assert "code against the note, not your memory of it" in FLAT


def test_long_thinking_bout_must_be_persisted_first() -> None:
    """The measured failure shape: a long stream truncated before any tool call.

    The clause must (a) name the long-thinking case, (b) say the stream can be
    cut off before a tool call, and (c) require the FIRST call after it to be the
    note — in that order, since (c) is the action and (a)(b) are its trigger.
    """
    i = FLAT.find("ESPECIALLY AFTER A LONG THINKING BOUT")
    j = FLAT.find("cut off before any tool call")
    k = FLAT.find("the FIRST call after one must be the note")
    assert 0 <= i < j < k < i + 400


def test_clause_hangs_off_the_persistence_rule() -> None:
    """It must attach to SS8's existing 'persists on disk across evictions'
    sentence, not float in an unrelated section."""
    i = FLAT.find("it persists on disk across evictions")
    j = FLAT.find("THE STEP NOTE IS A WORKING MEDIUM")
    assert 0 <= i < j < i + 700


def test_no_task_literal_leaked() -> None:
    low = FLAT.lower()
    for banned in ("ggtctc", "bsai", "neb", "dna-assembly", "overhang"):
        assert banned not in low, f"task literal leaked: {banned}"

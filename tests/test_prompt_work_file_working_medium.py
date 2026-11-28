"""The reasoning channel is a SEARCHABLE SOURCE, not a note to transcribe.

Carrier history (all measured, not assumed):

2026-11-26 13:03 trial — carrier = a plain scratch work file.
  The clause WAS adopted at the counter level (write_file 0 -> 4, plan_update
  11..13 -> 0), but only at "design is settled, time to emit the script": of 9
  bouts with >=60s of thinking, only 3/9 were followed by a write as the first
  action. The plan channel was killed outright, so the lifecycle never advanced.
2026-11-26 12:28 trial — carrier = the plan step note, reward = 1.

That history justified a clause telling the agent to write its settled
conclusions into a step note. 2026-11-28, the clause was measured on its own
terms and did not survive: across the three trajectories of one job
(dna-assembly, gcode-to-text, gpt2-codegolf),
`plan_update(notes=...)` was called 0 times and the phrase "step note" was
mentioned 0 times. The clause asked the model to re-transcribe its own
reasoning, and models generally do not do that — the user's words:
"不要强调step note时写thinking了，这个条款llm一般不遵从".

Meanwhile the premise under the clause changed. The harness now writes EVERY
reasoning block of the session to <session_dir>/thinking_log.md in FULL (no
per-entry slice, no dropped entries, no byte ceiling), and exposes
thinking_search(query=..., mode='entry') to locate a block and read it whole.
So "where did I decide X" is no longer an archaeology problem that a hand-note
must pre-solve: it is a search over the actual reasoning, at full fidelity.

The clause is therefore REWRITTEN, not extended:
  · the reasoning channel is named as searchable, with the tool and its two
    modes, so the instruction points at a capability that exists;
  · the step note keeps only its still-valid half — carrying the VALUES later
    code must re-read (a value, an offset, an invariant), which is exactly what
    a search of prose is a poor substitute for;
  · the measured long-bout risk is kept, because a stream can still be cut off
    before any tool call, and values living only there are lost.

Honest limit: these tests pin the CLAUSE TEXT and the budget, not behavioural
uptake. Uptake is measured by reading an agent log — the same way the old clause
was falsified.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flagscale_agent.react.prompt import (  # noqa: E402
    PROMPT_BUDGET,
    SYSTEM_PROMPT_STATIC as PROMPT,
    prompt_len,
)

FLAT = " ".join(PROMPT.split())


def test_reasoning_is_named_searchable_not_a_note() -> None:
    assert "YOUR REASONING IS SEARCHABLE, NOT A NOTE TO TRANSCRIBE" in FLAT


def test_the_transcribe_clause_is_gone() -> None:
    """The discarded instruction must not linger as a rival instruction.

    'write it to that step's note' is precisely the ask the model declined; if
    it stayed, the prompt would carry both the old ask and its replacement.
    """
    assert "THE STEP NOTE IS A WORKING MEDIUM" not in FLAT
    assert "write it to that step's note" not in FLAT
    assert "the FIRST call after one must be the note" not in FLAT


def test_the_clause_names_the_tool_and_both_modes() -> None:
    """It must point at a real capability: the tool NAME plus the way to get
    one block's whole text. A paraphrase without the tool name is unactionable."""
    i = FLAT.find("YOUR REASONING IS SEARCHABLE")
    window = FLAT[i:i + 400]
    assert "thinking_search(query=...)" in window
    assert "mode='entry'" in window


def test_it_says_the_blocks_are_on_disk_in_full() -> None:
    """The justification for searching rather than transcribing is that the
    full text is already persisted — not a head slice the reader must guess at."""
    i = FLAT.find("YOUR REASONING IS SEARCHABLE")
    assert "on disk in full" in FLAT[i:i + 200]


def test_the_note_keeps_its_still_valid_half() -> None:
    """The note is not abolished: it carries the VALUES later code re-reads,
    which a search over prose is a poor substitute for."""
    i = FLAT.find("The step note carries only what later code needs")
    assert i != -1
    window = FLAT[i:i + 120]
    assert "a value" in window and "an offset" in window and "an invariant" in window


def test_the_long_bout_risk_is_kept() -> None:
    """Measured loss shape, still true: a reasoning stream can be cut off before
    any tool call, so values living only in it are lost. The clause must state
    the risk AND the action, in that order."""
    i = FLAT.find("A long bout can be cut off before a tool call")
    j = FLAT.find("so land those first", i)
    assert 0 <= i < j < i + 120


def test_clause_hangs_off_the_persistence_rule() -> None:
    """It must attach to SS8's existing 'persists on disk across evictions'
    sentence, not float in an unrelated section."""
    i = FLAT.find("it persists on disk across evictions")
    j = FLAT.find("YOUR REASONING IS SEARCHABLE")
    assert 0 <= i < j < i + 700


def test_prompt_stays_within_budget() -> None:
    """The rewrite was sized to fit the EXISTING ceiling, not to raise it.

    User ruling: a new clause must merge with what is there, not lift the
    budget ("不会轻易抬高prompt预算").
    """
    assert prompt_len() <= PROMPT_BUDGET


def test_no_task_literal_leaked() -> None:
    low = FLAT.lower()
    for banned in ("ggtctc", "bsai", "neb", "dna-assembly", "overhang"):
        assert banned not in low, f"task literal leaked: {banned}"

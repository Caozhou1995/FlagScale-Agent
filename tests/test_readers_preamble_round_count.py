"""The preamble must describe the passes that ACTUALLY ran.

Regression guard for a defect found 2026-11-15: `_READERS_PREAMBLE` hard-coded
"three independent passes" while `_READERS_ROUNDS` had been lowered to 3 -> 2.
The preamble is the text that tells the agent how to read the blocks, so a
stale count misdescribes a 2-pass run as a 3-ballot one — the exact voting
reading the same paragraph exists to forbid.
"""
import os
import pytest

os.environ["FLAGSCALE_TASK_READERS"] = "1"

from flagscale_agent.react.agent import WorkerAgent


class _FakeReaders:
    def __init__(self, body="reading body"):
        self._body = body
        self._task_text = ""

    def execute(self):
        return self._body


def _agent(rounds, bodies):
    a = WorkerAgent.__new__(WorkerAgent)
    a._READERS_ROUNDS = rounds
    produced = list(bodies)
    a._task_statement_readers = _FakeReaders()
    # make .execute() return each body in turn, then repeat the last
    seq = iter(produced)

    def _exec():
        try:
            return next(seq)
        except StopIteration:
            return produced[-1]

    a._task_statement_readers.execute = _exec
    return a


@pytest.mark.parametrize("rounds,expected_word,expected_digits", [
    (2, "two", "2"),
    (3, "three", "3"),
])
def test_preamble_matches_actual_round_count(rounds, expected_word, expected_digits):
    a = _agent(rounds, ["body"] * rounds)
    out = a._auto_read_task_statement("do the thing")
    assert f"{expected_word} independent passes" in out
    assert f"Below it are {expected_digits} SEPARATE PASSES" in out
    assert f"These are NOT {expected_word} votes" in out
    # the hard-coded literal must never survive into the rendered text
    assert "three independent passes" not in out or rounds == 3
    assert (
        "{n}" not in out and "{word}" not in out and "{votes}" not in out
        and "{nv}" not in out and "{pass_word}" not in out
    )


def test_preamble_counts_PASSES_THAT_RAN_not_rounds_configured():
    """One pass failing must render as a 1-pass run, not as configured rounds."""
    a = _agent(2, ["only body"])
    calls = {"n": 0}

    def _exec():
        calls["n"] += 1
        if calls["n"] == 1:
            return "only body"
        raise RuntimeError("second pass died")

    a._task_statement_readers.execute = _exec
    out = a._auto_read_task_statement("do the thing")
    assert "one independent pass over" in out
    assert "two independent passes" not in out
    assert "only 1 of 2 produced a usable reading" in out


def test_readers_rounds_is_two():
    """The value the batch was lowered to; catches an accidental revert to 3."""
    assert WorkerAgent._READERS_ROUNDS == 2


def test_preamble_counts_the_viewpoints_that_SHIPPED_not_a_remembered_ten():
    """The preamble must name the real number of viewpoints.

    Same defect class as the pass count above, one layer down: the sentence
    "asking the same ten questions" was a literal while VIEWPOINTS grew from
    10 to 11. It is the ONLY place the agent is told how many questions it was
    asked, so a stale literal misdescribes the harness that produced the block.
    """
    from flagscale_agent.react.tools.read_task_statement import VIEWPOINTS

    n = len(VIEWPOINTS)
    a = _agent(2, ["body"] * 2)
    out = a._auto_read_task_statement("do the thing")
    assert f"asking the same {n} questions" in out, (
        f"preamble must state {n} questions, got: "
        + next((l for l in out.splitlines() if "questions" in l), "<none>")
    )
    # the old literal must never come back
    assert "same ten questions" not in out

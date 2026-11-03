"""Regression tests: G1c (WAITING IS NOT WORK) must stay pinned in the prompt.

Motivation (measured full batch, settled=49):
  - poll% (wait-class tool calls / total calls) mean: winners 8.4% vs losers
    16.0%; poll%>20 bucket went 5W/4L in the same batch.
  - rstan-to-pystan loss: 68 tool calls, 23 were sleep/shell_jobs-wait
    polls (34%), 3 guard overrides burned on wait-sleeps.
  - The clause exists in prompt.py as §7 "Waiting is not work" (current wording,
    L153) but NO test anchored it -- the same unanchored-clause shape that let
    the distillation run (9632fcf6) silently delete a discipline rule.

The clause must stay domain-neutral (no task names, no harness internals).
"""

from pathlib import Path

PROMPT = (
    Path(__file__).resolve().parents[1]
    / "flagscale_agent"
    / "react"
    / "prompt.py"
)
MARKER = "WAITING IS NOT WORK"
# The current marker (title-case, em-dash): "Waiting is not work:"
MARKER_LOW = "waiting is not work"


def _text() -> str:
    return PROMPT.read_text()


def _clause() -> str:
    """The §7 waiting clause: the single prompt line holding the marker."""
    for ln in _text().splitlines():
        if MARKER_LOW in ln.lower():
            return ln
    return ""


def test_marker_present():
    assert MARKER_LOW in _text().lower()


def test_two_rules_present():
    clause = _clause().lower()
    # rule 1: bank a crude deliverable BEFORE waiting (secure deliverable first)
    assert "secure the deliverable first" in clause
    # rule 2: bound the wait + every poll productive (cap with timeout, poll it,
    # do real work meanwhile -- no idle re-check loop)
    assert "cap a long job with a timeout" in clause
    assert "poll it" in clause
    assert "do real work meanwhile" in clause


def test_no_task_names_in_clause():
    # extract the G1c block (the §7 'Waiting is not work' clause line)
    clause = _clause()
    assert clause, "marker missing"
    banned = [
        "filter-js", "ocaml", "mteb", "rstan", "pytorch", "tbench",
        "harbor", "v61", "reward", "verifier", "subset3",
    ]
    hits = [b for b in banned if b in clause.lower()]
    assert hits == [], f"task/harness identifiers leaked into G1c clause: {hits}"

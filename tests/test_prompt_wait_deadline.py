"""G13: A WAIT ARGUMENT IS A DEADLINE, NOT A HINT.

If an API accepts a wait/throttle/deadline argument, the caller uses it to
guarantee a side effect has COMPLETED (port listening, file flushed, process
spawned). The wait must be honoured up to the deadline even if there is NO
output during the wait; an idle-detection early-exit heuristic ("return after
N ms of silence") is forbidden as the sole settle mechanism, because it loses
the race against a one-shot consumer that does not retry.

Derived from a quantitative race measurement (jobs/2026-10-21__08-10-31,
headless-terminal): the grader issues ONE retry-less request right after the
wait call returns; the delivered implementation's idle-based early exit
returned at ~100 ms while the side effect (server port) became ready at
+186 ms => ECONNREFUSED. The wait argument was treated as a hint, not a
deadline. Also requires the pre-delivery self-check: replay the CONSUMER'S
literal invocation shape (one-shot, no retry) against a wait set to N.
"""
import re
import subprocess
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text(encoding="utf-8")

# Generic harness/engineering terms only -- no task-identifying words.
FORBIDDEN = [
    r"\bterminal-bench\b",
    r"\bheadless-terminal\b",
    r"\bhttp\.server\b",
    r"\bECONNREFUSED\b",
    r"\brequests\.get\b",
    r"\bsend_keystrokes\b",
]


def test_g13_deadline_clause_present():
    assert "A WAIT ARGUMENT IS A DEADLINE, NOT A HINT" in TEXT


def test_g13_forbids_idle_heuristic_early_exit():
    # The clause must explicitly forbid settling via a silence/idle heuristic
    # when a wait deadline is given, and require honouring the full deadline.
    assert re.search(r"even if (there is )?(absolutely )?no output", TEXT, re.I)
    assert re.search(r"not a hint|not a suggestion", TEXT, re.I)


def test_g13_requires_one_shot_consumer_shape_selfcheck():
    # Pre-delivery self-check: replay the consumer's literal invocation shape
    # (a single retry-less probe) after honouring the wait.
    assert re.search(r"one[- ]shot", TEXT, re.I)
    assert re.search(r"no retry|without retry|not retry", TEXT, re.I)


def test_g13_side_effect_examples_are_generic():
    # Generic side-effect classes, no task-specific vocabulary.
    for kw in ["port", "flush", "spawn"]:
        assert re.search(kw, TEXT, re.I)


def test_no_task_identifying_words_added():
    body = TEXT.lower()
    for pat in FORBIDDEN:
        assert not re.search(pat, body), f"leak pattern in prompt: {pat}"

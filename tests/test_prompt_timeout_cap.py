"""Anti-decay tests for the per-command timeout-cap clause (prompt.py TIME BUDGET).

Locks:
1. The clause exists with its load-bearing phrases (judge liveness limitation, measured
   evidence, cap mechanism, cap-sizing rule, dead-cap response).
2. No task-specific literals leak in (make-doom/gcode/gpt2/torch-pipeline are the
   motivating losses; the clause must stay generic).
"""
import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _text() -> str:
    return PROMPT.read_text(encoding="utf-8")


def test_timeout_cap_clause_present():
    t = _text()
    assert "EVERY LONG COMMAND CARRIES ITS OWN TIMEOUT CAP" in t
    for phrase in [
        "can only kill a command that is BROKEN",
        "never one that is merely unnecessary",
        "one unbounded tree-wide grep ran 14 minutes",
        "timeout <N>s <cmd>",
        "Size N to what the step can legitimately need",
        "do not re-run the same command with a bigger cap",
    ]:
        assert phrase in t, phrase


def test_timeout_cap_clause_no_task_literals():
    t = _text()
    for lit in ["make-doom", "gcode", "gpt2", "torch-pipeline", "codegolf"]:
        assert lit not in t, lit

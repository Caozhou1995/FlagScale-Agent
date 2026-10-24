"""Anti-decay tests: event-selection dominance + distractor variant (2026-10-24).

The v2 video batch lost to a first-match event selector (takeoff=38: a 100x
smaller clutter pulse at f34-39 was reported instead of the dominant jump at
f219-223). The clause "SELECT BY MEASURED DOMINANCE, not by first match" plus
the distractor-variant requirement in the INVISIBLE-INPUT CONTRACT must not
soften. Mutation: replace 'not by first match' -> 'or by first match' and
'not optional' -> 'optional'; expect exactly these tests to fail.
"""
from pathlib import Path

import pytest

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _body() -> str:
    return PROMPT.read_text(encoding="utf-8")


def test_dominance_selection_clause_present():
    b = _body()
    assert "SELECT BY MEASURED DOMINANCE" in b
    assert "not by first match" in b
    # the ranking key must be the event's own measured scale, task-agnostic
    assert "excursion size" in b


def test_dominance_clause_is_general_no_task_literals():
    b = _body()
    for leak in ("takeoff", "landing", "video", "jump", "frame 38", "MNIST"):
        assert leak not in b.split("SELECT BY MEASURED DOMINANCE")[1].split("not optional")[0]


def test_distractor_variant_required_not_optional():
    b = _body()
    assert "a distractor variant" in b
    assert "is not optional" in b
    # the variant list must still be answered by RUNNING, not reading
    assert "RUNNING each synthesized variant in that list" in b

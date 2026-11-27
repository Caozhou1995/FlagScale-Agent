"""Pin the SHAPE of the (c) worked-example detector.

The detector must fire on a real value replay and must NOT fire on a numbered
procedure. Both directions are asserted here because a rule that stops firing
altogether also passes "the prompt is clean", so the clean-prompt test alone
cannot distinguish a narrowed detector from a dead one.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GATE_PY = REPO / "analysis" / "leak_gate.py"


def _gate():
    if not GATE_PY.is_file():
        pytest.skip(f"leak_gate.py absent at {GATE_PY}")
    spec = importlib.util.spec_from_file_location("_leak_gate_ord", GATE_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_leak_gate_ord"] = mod
    spec.loader.exec_module(mod)
    return mod


def _fires(g, line: str) -> bool:
    nums = {int(m.group(1)) for m in g.BARE_NUMBER_PAIR_PAT.finditer(line)}
    return (len(nums) >= 2
            and bool(g.EXAMPLE_MARKER_PAT.search(line))
            and bool(g.BOUND_WORD_PAT.search(line)))


def test_detector_still_fires_on_a_real_value_pair():
    """POSITIVE control: a measured value beside the stated bound is the leak."""
    g = _gate()
    leak = ("for example the stated bound is 5000 and your value came out 5695")
    assert _fires(g, leak), "detector went blind to an actual value replay"


def test_detector_ignores_parenthesised_ordinals():
    """NEGATIVE control: a ranked procedure is not a value replay.

    This is the exact shape that produced the false positive on the prompt's
    inline-confidence ladder. The line MUST carry a bound word ("value") and an
    example marker ("worked example") — otherwise the conjunction fails for a
    trivial reason and the test cannot tell the narrowed regex from the old one.
    """
    g = _gate()
    ladder = ("stopping at the first rung that answers: (1) an implementation "
              "already in the container, never your own formula beside the "
              "value (2) A library default or the standard's worked example, "
              "with its source (3) What the consumer's material fixes")
    assert g.EXAMPLE_MARKER_PAT.search(ladder), "fixture lost its marker"
    assert g.BOUND_WORD_PAT.search(ladder), "fixture lost its bound word"
    assert not _fires(g, ladder), "detector fired on enumeration labels"


def test_detector_ignores_a_bare_date():
    """NEGATIVE control: a date line carries no worked-example marker, so the
    conjunction (marker AND bound word AND pair) never closes.

    Deliberately does NOT pair the date with 'worked example' — doing so really
    does read like a value replay, and firing there is correct.
    """
    g = _gate()
    assert not _fires(g, "measured on 2026-10-27 against the graded instance")


def test_ordinal_exclusion_does_not_hide_a_bare_pair_beside_an_ordinal():
    """The narrowing must not become a blanket amnesty on the same line."""
    g = _gate()
    mixed = ("(2) A library default or the standard's worked example: "
             "a bound of 5000 and your value 5695")
    assert _fires(g, mixed), "ordinal exclusion swallowed a genuine pair"


def test_the_shipped_prompt_is_clean():
    """End-to-end: the real agent-facing tree reports no (c) violations."""
    gate = _gate()
    assert not gate.check_agent_facing(REPO)

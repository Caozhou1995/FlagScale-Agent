"""The provenance clause: an all-green self-built list can coexist with a rejection.

Measured (dna-insert, three independent trials): the agent closed each run with an
"N/N clause checks PASS" list it had authored itself, while the grader rejected the
deliverable on a property that had never entered that list. The list could not fail
because the item was absent -- so tightening the verdict changed nothing. The lever
is provenance: every item must name where it came from; an item that cannot is an
open choice to be covered, not a face to drop.
"""
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


def _flat() -> str:
    spec = importlib.util.spec_from_file_location("prompt_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return " ".join(mod.SYSTEM_PROMPT_STATIC.lower().split())


FLAT = _flat()


def test_all_green_list_is_not_completeness():
    """Names the failure shape itself: green and rejected at once."""
    assert "can coexist with a rejection" in FLAT


def test_uncarried_item_cannot_fail():
    """The mechanism: absence of an item is why the list stays green."""
    assert "an item you never carried cannot fail" in FLAT


def test_provenance_is_mandatory_per_item():
    """The actionable demand, not a virtue."""
    assert "make provenance mandatory per item" in FLAT


def test_the_three_admissible_sources_are_named():
    """Task sentence / executed authority read / consumer tool run."""
    assert "the task sentence it reads" in FLAT
    assert "the authority read you executed" in FLAT
    assert "the consumer tool run you observed" in FLAT


def test_no_source_means_open_choice_and_coverable():
    """An item owing none is a guess -> list the choice open, carry every reading."""
    assert "an item owing none is your own guess" in FLAT
    assert "list the choice open and carry every reading you can name" in FLAT


def test_clause_is_task_agnostic():
    """No domain nouns from any measured task may appear in the clause."""
    i = FLAT.find("make provenance mandatory per item")
    assert i > 0
    window = FLAT[max(0, i - 700): i + 700]
    for banned in ("clamp", "primer", "tm ", "overhang", "ggtctc", "bsai", "restriction"):
        assert banned not in window, banned

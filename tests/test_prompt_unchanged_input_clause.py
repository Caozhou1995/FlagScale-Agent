"""Regression: unchanged-input byte-level self-check clause stays intact.

Guards against softening of the clause added 2026-10-25 (lever v8) after the
filter-js failure class: agent self-checks round-trip both sides through the
same parser/serializer (normalize both sides) and structurally cannot see
their own rewrites, so "clean files unchanged" was never actually verified.
"""
import pathlib
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"


@pytest.fixture(scope="module")
def prompt_text():
    return SRC.read_text(encoding="utf-8")


def test_clause_present(prompt_text):
    assert "UNCHANGED-INPUT SELF-CHECK IS BYTE-LEVEL OR IT IS BLIND" in prompt_text


def test_clause_requires_byte_comparison(prompt_text):
    seg = prompt_text.split("UNCHANGED-INPUT SELF-CHECK IS BYTE-LEVEL OR IT IS BLIND", 1)[1].split("- ", 1)[0]
    for needle in ("cmp", "md5sum"):
        assert needle in seg, f"clause lost byte-comparison anchor {needle!r}"


def test_clause_names_the_co_source_trap(prompt_text):
    seg = prompt_text.split("UNCHANGED-INPUT SELF-CHECK IS BYTE-LEVEL OR IT IS BLIND", 1)[1].split("- ", 1)[0]
    assert "normalizes BOTH sides" in seg
    assert "structurally incapable" in seg


def test_clause_no_task_literals(prompt_text):
    """Red line: no benchmark task names/specifics in the clause."""
    seg_start = prompt_text.find("UNCHANGED-INPUT SELF-CHECK IS BYTE-LEVEL OR IT IS BLIND")
    seg = prompt_text[seg_start:seg_start + 2500]
    for bad in ("filter-js", "html", "HTML", "bs4", "BeautifulSoup"):
        assert bad not in seg, f"task literal {bad!r} leaked into generic clause"

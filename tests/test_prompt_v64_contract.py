"""v64 system-prompt contract (semantic, not verbatim).

Replaces the verbatim-assertion test family retired in docs/prompt_invariants.md.
Each test asserts an INVARIANT of the v64 prompt -- the presence of a discipline,
not the exact sentence that states it -- so the prompt can be edited without
either locking it to stale wording or silently losing a discipline.

C11 (no legacy section names) is the anti-backflow ratchet: it fails if an old
section heading creeps back, which is how the previous prompt grew to 101k chars.
"""

from __future__ import annotations

import re

import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC, prompt_len


@pytest.fixture(scope="module")
def text() -> str:
    return SYSTEM_PROMPT_STATIC


@pytest.fixture(scope="module")
def rendered(text: str) -> str:
    return text.format(cwd="/x", tools="t", skills="s", knowledge="k")


def test_c1_name_unchanged(text: str) -> None:
    """C1: the agent's name is unchanged (the user relies on it)."""
    assert "FlagScale Agent" in text.splitlines()[0]


def test_c2_all_placeholders_present_and_renderable(text: str, rendered: str) -> None:
    """C2: exactly the four runtime placeholders, and .format() works."""
    for ph in ("{cwd}", "{tools}", "{skills}", "{knowledge}"):
        assert ph in text, f"missing placeholder {ph}"
    assert "{cwd}" not in rendered
    assert "/x" in rendered


def test_c3_total_size_bounded(text: str) -> None:
    """C3: the prompt stays a high-signal set, not a clause accumulation.

    15000 chars is the budget this rewrite commits to (was 101657). The bound is
    deliberately loose enough to allow refinement and tight enough that a relapse
    into per-loss clause appending fails here rather than silently.

    Measured through prompt.prompt_len() — the SINGLE canonical reading
    (prop_9f81ebea). Asserting len(SYSTEM_PROMPT_STATIC) here while the agent
    measured a rendered assembly produced two numbers 134 chars apart for the
    same tree, so a "within budget" claim could be green in one and red in the
    other.
    """
    n = prompt_len()
    assert n <= 20_000, f"prompt is {n} chars, budget 20000"


def test_c3b_the_two_readings_of_prompt_size_agree(text: str) -> None:
    """C3b (prop_9f81ebea): static-vs-rendered drift cannot hide again.

    The budget is enforced on the RENDERED text. If the raw constant ever stops
    being a faithful proxy for it (placeholders gaining long defaults, or the
    render growing an extra block), the two readings diverge and every
    agent-reported size silently stops meaning what the gate checks. Pin them
    together so the divergence itself is the failure.
    """
    rendered = prompt_len()
    static = len(text)
    # The ONLY permitted difference is the placeholder substitution itself
    # ({cwd}/{tools}/{skills}/{knowledge} replaced by short sentinels).
    assert abs(static - rendered) <= 200, (
        f"static={static} vs rendered={rendered}: the prompt budget is measured "
        "on one object and reported on another")


def test_c4_ten_sections(text: str) -> None:
    """C4: the ten-section skeleton is intact."""
    heads = re.findall(r"^════ (.+?) ════$", text, re.M)
    assert len(heads) == 10, f"expected 10 sections, got {len(heads)}: {heads}"


def test_c5_four_disciplines_present(text: str) -> None:
    """C5: the four cross-field disciplines (U1/U3/U5/D5) each have a clause."""
    low = text.lower()
    assert "falsifiab" in low, "1.1 falsifiability missing"
    assert "systematicity" in low or "more systematic" in low, "1.2 systematicity missing"
    assert "error cost" in low, "1.3 error cost missing"
    assert "verification" in low and "validation" in low, "1.4 verification≠validation missing"


def test_c6_field_table_covers_six_fields(text: str) -> None:
    """C6: the field-mode table exists and names the six fields."""
    assert "| field |" in text, "field-mode table header missing"
    low = text.lower()
    for field in ("math", "physics", "biology", "simulation", "statistics", "infra"):
        assert field in low, f"field row missing: {field}"
    # the transferable-move sentence that makes the table useful
    assert "borrow" in low or "borrowing" in low, "cross-field transfer warning missing"


def test_c7_measurement_three_questions(text: str) -> None:
    """C7: who produced it / a second path / would the check have gone red."""
    low = text.lower()
    assert "who produced it" in low
    assert "second way" in low or "second path" in low or "does not share source" in low
    assert "have failed if i were wrong" in low or "would this run still have produced" in low


def test_c8_closeout_four_steps(text: str) -> None:
    """C8: the closing-out four steps survive (the old G1-G5 was folded in here)."""
    for step in ("9.1", "9.2", "9.3", "9.4"):
        assert step in text, f"close-out step {step} missing"
    low = text.lower()
    assert "clause by clause" in low
    assert "validation" in low
    assert "not verified" in low


def test_c9_response_three_channels(text: str) -> None:
    """C9: the RSI wrap-up channels are named (this is RSI's new home)."""
    assert "Language" in text
    assert "Task result" in text
    assert "RSI result" in text
    low = text.lower()
    assert "mechanism changes landed" in low
    assert "proposals" in low
    assert "memory updated" in low or "memory" in low


def test_c10_markers_present(text: str) -> None:
    """C10: both stop markers are defined."""
    assert "[TASK_COMPLETE]" in text
    assert "[NEED_USER_INPUT]" in text


def test_c11_no_legacy_section_names(text: str) -> None:
    """C11 (anti-backflow ratchet): retired v63 headings must not reappear.

    Behind this assertion sits the whole observed failure: every past loss added a
    clause, and those clauses became the 45-section 101k-char prompt. A reappearing
    heading means the additive reflex came back.
    """
    retired = [
        "PRINCIPLE 1 —",
        "PRINCIPLE 2 —",
        "PRINCIPLE 4 —",
        "G1 BYTES, NOT MODEL",
        "2.1a A CHECK IS ONLY AS GOOD",
        "2.1b SEMANTIC CO-SOURCING",
        "STAGE 2 — OPERATE",
        "STAGE 3 — DELIVER",
        "BUDGET ORDER",
    ]
    found = [h for h in retired if h in text]
    assert not found, f"retired v63 headings reappeared: {found}"


def test_c12_no_chinese_prose(text: str) -> None:
    """C12: English prose only; short quoted examples are allowed.

    Matches the repo's existing contract (tests/test_context_v3.py::
    test_static_prompt_is_english): deliberate teaching snippets inside double
    quotes are data, not prose. Only Chinese OUTSIDE a quoted example fails.
    """
    ok_spans = [
        m.span() for m in re.finditer(r'"[^"]*[\u4e00-\u9fff][^"]*"', text)
    ]
    for m in re.finditer(r"[\u4e00-\u9fff]+", text):
        assert any(s <= m.start() and m.end() <= e for s, e in ok_spans), (
            f"Chinese prose outside a quoted example at {m.start()}: "
            f"{text[max(0, m.start() - 40):m.end() + 20]!r}"
        )


def test_c13_delivery_selfcheck_not_ritualized(text: str) -> None:
    """C13: close-out must demand running the task's own acceptance path.

    This is the discipline the deleted G1-G5 gate block encoded. It must survive
    as a demand, not as a 23k-char checklist.
    """
    low = text.lower()
    assert "task's own acceptance path" in low or "task names a test" in low
    assert "raw output" in low
    assert "delivered" in low and "bytes" in low


def test_c14_budget_walls_sum_not_whole(text: str) -> None:
    """C14 (anti-backflow for the #90-F1 regression): the fan-out tail clause
    is a verbatim load-bearing phrase. #90-F1 proved a 13-passed gate can hide
    its deletion (the deleted half was outside every gate's evidence set).
    """
    assert "longest chain plus its join, not the sum" in text

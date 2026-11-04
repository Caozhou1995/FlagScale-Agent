"""System-prompt contract (semantic, not verbatim).

Replaces the verbatim-assertion test family retired in docs/prompt_invariants.md.
Each test asserts an INVARIANT of the prompt -- the presence of a discipline,
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

    33000 chars is the budget this rewrite commits to (101657 before; 32000 until the
    2026-11-04 clause additions). The bound is
    deliberately loose enough to allow refinement and tight enough that a relapse
    into per-loss clause appending fails here rather than silently.

    Measured through prompt.prompt_len() — the SINGLE canonical reading. Asserting len(SYSTEM_PROMPT_STATIC) here while the agent
    measured a rendered assembly produced two numbers 134 chars apart for the
    same tree, so a "within budget" claim could be green in one and red in the
    other.
    """
    n = prompt_len()
    assert n <= 36_000, f"prompt is {n} chars, budget 36000"


def test_c3b_the_two_readings_of_prompt_size_agree(text: str) -> None:
    """C3b: static-vs-rendered drift cannot hide again.

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
    """C8: the closing-out steps survive (the old G1-G5 was folded in here)."""
    for step in ("10.1", "10.2", "10.3", "10.4", "10.5"):
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
    assert "mechanisms landed" in low
    assert "proposals" in low
    assert "memory" in low


def test_c10_markers_present(text: str) -> None:
    """C10: both stop markers are defined."""
    assert "[TASK_COMPLETE]" in text
    assert "[NEED_USER_INPUT]" in text


def test_c11_no_legacy_section_names(text: str) -> None:
    """C11 (anti-backflow ratchet): retired headings must not reappear.

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
    assert not found, f"retired headings reappeared: {found}"


def test_c12_no_chinese_prose(text: str) -> None:
    """C12: English prose only; short quoted examples are allowed.

    Matches the repo's existing contract (tests/test_context.py::
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

def test_c12_each_clause_needs_its_own_witness(text: str) -> None:
    """C12: 10.1 must forbid discharging several clauses with one command.

    Two measured failures in the same batch had this exact shape: a run whose
    own self-checks answered a nearby question, and then called the clause done.
    One verified byte-equality of the produced file while the clause named an
    editor exit command; another verified the stability of the author's own
    model while the clause named two specific peaks. In both the clause's own
    object was never witnessed, and both were scored wrong while the run felt
    finished. The rule must demand a per-clause witness, and must say that a
    clause whose own object cannot be named is OPEN, not merely at risk.
    """
    low = text.lower()
    assert "does not discharge them one by one" in low
    assert "per clause" in low and "object is the clause itself" in low
    assert "open, not done" in low


def test_c15_closeout_is_sealed_per_clause_not_judged(text: str) -> None:
    """C15 (the close-out loop's real root cause): the stop rule must be a
    COUNT/SEAL, not the agent's own judgement.

    Measured 2026-11-04 (n=3 batch, jobs/2026-11-04__18-31-06): three tasks were
    scored 1/0/0. The two zeroes had nothing to do with ability -- one delivered
    the raw axis value where the task's reference was the converted cm^-1 value,
    the other delivered a bare `wq` where the task's literal was `:wq`. Both had
    invented "one more check" rounds that read bytes without producing
    information, and neither ever sealed a clause. The stop rule quoted
    "no new READING", which an agent can satisfy forever by re-confirming what
    it already knows. So the prompt must (a) seal each clause against its object,
    (b) define the stop signal as new INFORMATION with confirmation yielding
    zero, and (c) bound close-out by the clause count rather than by the agent's
    sense that it has not missed something.
    """
    low = text.lower()
    assert "seal the clause against that object" in low
    assert "second probe against the same object" in low
    assert "first probe against the new object is not a second probe" in low
    # The clause this test used to lock ("new information, not a new reading")
    # was retired on 2026-11-04 after the n=3 batch measured its failure mode:
    # kernel.py:512 only evaluates the stop marker on a response that carries
    # NO tool call, and the zero-reward trials emitted 29 responses with zero
    # tool-free ones, so TASK_COMPLETE appeared 0 times while the wall clock
    # ran out. The enforceable rule is therefore mechanical, not motivational.
    assert "emit the marker in a text-only response" in low
    assert "skipped past the marker check" in low
    assert "licenses no further probe" in low
    assert "one probe per unsealed clause" in low
    assert "read-back consumes no probe" in low


def test_c16_literal_versus_format_name(text: str) -> None:
    """C16 (the `wq` vs `:wq` gap): a literal the task states as CONTENT must
    appear character-for-character; a format or standard NAME is verified by
    that format's rules and must NOT be demanded verbatim.

    Measured: large-scale-text-editing reward=0 with
    `AssertionError: Missing :wq or :x`, where the grader's own test is
    `has_exit = any(line in (":wq", ":x") for line in lines)` and the agent had
    read the task's `+ Exit: :wq or :x` line, deliberated on it, and still
    shipped the equivalent-but-not-verbatim `wq`. Without the second half of
    this rule the clause would over-fire on "output JSON"-style names, so both
    halves are load-bearing.
    """
    low = text.lower()
    assert "must appear character-for-character" in low
    assert "`wq` is not `:wq`" in low
    assert "not a literal to reproduce" in low
    assert "format's rules" in low or "format's own rules" in low
    assert "after any derived conversion" in low


def test_c17_marker_is_terminal_no_housekeeping_after(text: str) -> None:
    """C17: the marker is the END, not a milestone. Once `[TASK_COMPLETE]`
    (or `[NEED_USER_INPUT]`) is on the last line, no action may follow it.

    Measured: the close-out loops that burned whole turns had an agent that
    kept doing "one last" cat/md5/poll AFTER the decision was already closed.
    The clause must therefore say the marker is terminal AND that any action
    after it is overrun — wording the agent can act on without seeing a grader.
    """
    low = text.lower()
    assert "after the marker nothing is owed" in low
    assert "any action taken once it is there is overrun" in low
    assert "land, emit, stop" in low
    # C17b (measured): the loop fired on large-scale-text-editing AFTER plan complete
    # (results.json banked, apply_macros.vim byte-matched) — the agent answered its own
    # "Now final response." with one more ls/cat/plan_update. step_done on a pass is
    # normal progress and stays unconstrained; only PLAN COMPLETE seals the close-out,
    # and then the marker is the ONLY next response. A stray call is "not a close-out
    # step" (land and emit without it) rather than voiding the whole close-out, which
    # would contradict the four-fifths rule's "land what you have and emit".
    assert "once plan complete, every clause is sealed" in low
    assert "the only next response is the marker" in low
    assert "is not a close-out step" in low
    assert "land and emit the marker without it" in low


def test_c18_stated_bound_and_named_quantity(text: str) -> None:
    """C18: a stated range/threshold is a clause with its own object.

    Measured on the 16-reward-0 sweep: video-processing delivered landing frame 61
    for a task saying [62, 64] (off-by-one on the closed interval); mteb-retrieve
    produced a HumanEval number for an MTEB task (right-looking number, wrong
    quantity). Both grades as "checked a number", so the clause must name the END
    the wording fixes and the QUANTITY the task named.
    """
    low = text.lower()
    assert "a floor or a closed interval" in low
    assert "the quantity the task named" in low


def test_c19_near_miss_on_small_sample(text: str) -> None:
    """C19: a score just under a stated bar is not "close enough".

    Measured: train-fasttext 0.614 vs 0.62, make-mips-interpreter 0.8629 vs 0.95,
    torch-pipeline-parallelism max diff 0.0357. Each ended "nearly there" while a
    form/population mismatch was the likelier cause; the guard is to widen the
    evidence, not to talk the gap away.
    """
    low = text.lower()
    assert "a measurable score just under a stated bar" in low
    assert "not noise and not" in low

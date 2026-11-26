"""The blind-spot reader must be able to report a REQUIRED ELEMENT'S PRESENCE as
its own blind spot, not only a choice among its possible values.

Why this file exists (measured, not imagined): a real graded run was failed by a
checker asserting that a required structural element was present at all. The
region that was injected into that run enumerated six blind spots, every one a
value/geometry choice, and did not raise the presence question once. The old
prompt's output contract carried a single slot ("the TWO settlements"), which
fits a value choice and silently excludes a presence point; it also capped output
at 8 lines, so an external standard's several required elements got collapsed
into one line. Both defects are asserted away here.
"""

import re

from flagscale_agent.react.tools.read_task_statement import _BLINDSPOT_PROMPT


def _flat(text):
    return " ".join(text.split())


def test_presence_form_is_named_as_a_settlement_shape():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "PRESENCE" in flat
    assert "carries it vs omits it" in flat


def test_value_form_and_presence_form_are_both_admitted():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "VALUE" in flat and "GEOMETRY" in flat
    assert flat.index("VALUE") < flat.index("PRESENCE")


def test_external_standard_requires_one_line_per_required_element():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "ONE PER LINE" in flat
    assert "never collapse the set into one" in flat


def test_enumeration_outranks_the_line_cap():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "outranks that cap" in flat
    # the old flat cap of 8 is what collapsed a multi-element standard
    assert "At most 8 lines" not in flat


def test_cap_is_still_bounded_so_the_region_cannot_run_away():
    flat = _flat(_BLINDSPOT_PROMPT)
    m = re.search(r"At most (\d+) lines", flat)
    assert m, flat[:400]
    assert 8 <= int(m.group(1)) <= 14


def test_presence_point_must_not_be_restated_as_a_value_choice():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "never restate it as which value to give the element" in flat


def test_placeholders_are_intact_and_the_template_renders():
    assert _BLINDSPOT_PROMPT.count("{sentinel}") == 1
    assert _BLINDSPOT_PROMPT.count("{task_text}") == 1
    rendered = _BLINDSPOT_PROMPT.format(sentinel="SENTINEL_X", task_text="TASK_Y")
    assert "SENTINEL_X" in rendered and "TASK_Y" in rendered


def test_prompt_stays_within_its_size_budget():
    """The reader call is already the slowest part of startup; a big prompt
    growth costs wall clock on every run. 3048 = the previous 2848 + 200, which
    was sized for the twelve-viewpoint reader; the slim blind-spot reader
    replaced it and measures 3533, so 3600 = measured + slack. Keep it equal to
    the sibling test's bar."""
    assert len(_BLINDSPOT_PROMPT) <= 3600

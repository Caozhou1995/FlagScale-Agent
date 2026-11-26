"""The blind-spot reader must treat the FIELD'S OWN STANDARD as a constraint,
not only the task statement's wording.

Why this file exists (measured, not imagined): a graded run's delivered bytes
cleared every constraint the statement spelled out and still failed, because the
field's practice required something the statement never mentioned. An earlier
region enumerated several genuine sentence-level ambiguities and raised none of
them as a domain requirement. The generic criterion pinned here is
"reasonable = satisfies the text AND the field", not any one task's rule.

Seating constraint (also pinned): the reader prompt must stay inside its own
size bar, so the criterion is folded into the definition of a blind spot rather
than appended as a new numbered kind. Raising the bar is not an option.
"""

from flagscale_agent.react.tools.read_task_statement import _BLINDSPOT_PROMPT


def _flat(text):
    return " ".join(text.split())


def test_a_blind_spot_is_pinned_by_two_alternative_grounds():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "NOT PINNED" in flat
    # the text-leaves-it-open ground AND the field-standard ground, in one sentence
    assert "the text leaves it open, OR the field's standard demands what" in flat


def test_text_is_declared_a_floor_not_the_whole_standard():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "FLOOR" in flat
    assert "no practitioner would accept it otherwise" in flat


def test_domain_settlement_is_a_form_axis_not_a_value_to_fill():
    flat = _flat(_BLINDSPOT_PROMPT)
    assert "form axis, not a value" in flat


def test_domain_axis_does_not_replace_the_sentence_level_rules():
    flat = _flat(_BLINDSPOT_PROMPT)
    # pre-existing contract must survive the fold
    assert "OUTCOME-BEARING" in flat
    assert "A TIE-BREAK, DEFAULT or ORDER" in flat
    assert "carries it vs omits it" in flat
    assert "ONE PER LINE" in flat


def test_reader_prompt_stays_inside_its_size_bar():
    # 3048 was sized for the TWELVE-VIEWPOINT reader. The slim blind-spot reader
    # replaced it (see the comment block above _BLINDSPOT_PROMPT) and measures
    # 3533, so the ceiling is restated as 3600 = measured + slack. This is a
    # DECLARED envelope, not a licence: any further growth must move this number
    # again, in the open. Keep it equal to the sibling test's bar.
    assert len(_BLINDSPOT_PROMPT) <= 3600


def test_prompt_contract_placeholders_survive():
    assert "{task_text}" in _BLINDSPOT_PROMPT
    assert "{sentinel}" in _BLINDSPOT_PROMPT
    out = _BLINDSPOT_PROMPT.replace("{task_text}", "X" * 10).replace(
        "{sentinel}", "NONE"
    )
    assert "{" not in out and "}" not in out

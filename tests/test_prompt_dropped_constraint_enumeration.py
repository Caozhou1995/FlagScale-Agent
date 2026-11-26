"""§10.4 must make the DROPPED-constraint class visible before delivery.

Evidence (2026-11-23, dna-assembly trial iU2rY9m, reward=0):
  - The agent identified the extra-5'-base requirement FIVE times (log lines
    132/134/326/331/957), tried FOUR authorities (all refused), then decided at
    log lines 4623-4627: "this is a minor efficiency concern, not a correctness
    requirement ... I'll keep the site at the 5' end (standard)."
  - It then shipped clamp=0 on all 8 primers -> grader `assert 0 >= 1` at
    tests/test_outputs.py:49, a failure that stops before any downstream check.
  - Its own 30/30 checklist was green, because a constraint judged non-binding
    never enters the checklist that the checklist is built from.

So the gap is not "compute the value a second way" (E1-E8 already say that); it
is that a BINDING-CLASS decision is invisible to every existing gate.
"""

import flagscale_agent.react.prompt as prompt

NEW_CLAUSE = (
    "Then list every constraint you CONSIDERED AND DROPPED as not binding, each with the "
    "reading that would make it binding."
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _rendered() -> str:
    return _flat(prompt.render_prompt())


def _sweep_block() -> str:
    flat = _rendered()
    start = flat.find("10.4 SWEEP FOR WHAT YOU DID NOT REACH")
    assert start != -1, "§10.4 heading not found"
    end = flat.find("10.5 THE LOOP ON A RESPONSE", start)
    assert end != -1, "§10.5 heading not found"
    return flat[start:end]


def test_sweep_requires_enumerating_dropped_constraints():
    r = _rendered()
    assert NEW_CLAUSE in r


def test_dropped_enumeration_is_inside_sweep_not_stray():
    """It must land in 10.4 (the sweep), not float somewhere unrelated."""
    assert NEW_CLAUSE in _sweep_block()


def test_states_why_a_green_self_check_cannot_see_it():
    """The causal half: omission is invisible to the checklist it never entered."""
    r = _rendered()
    assert "never enters your own checklist" in r
    assert "invisible by construction" in r


def test_dropping_is_named_a_bet():
    r = _rendered()
    assert "Dropping it is a bet" in r


def test_unreachable_authority_is_not_evidence_for_dropping():
    """E7 says a failed call indicts your call; it must not license omission."""
    r = _rendered()
    assert "an unreachable authority is not its evidence" in r


def test_binding_class_is_named_not_just_value_readings():
    """The clause must speak of binding/advisory, the class E1-E8 never cover."""
    blk = _sweep_block()
    assert "not binding" in blk
    assert "advisory" in blk


def test_clause_stays_task_agnostic():
    """No benchmark vocabulary may leak in (example == answer)."""
    banned = [
        "oligotm", "primer3", "bsai", "ggtctc", "primer", "anneal",
        "overhang", "dna", "clamp", "tm ", "restriction",
    ]
    blk = _sweep_block().lower()
    hits = [b for b in banned if b in blk]
    assert not hits, f"task vocabulary leaked into §10.4: {hits}"


def test_sweep_still_demands_truncation_be_stated():
    """The pre-existing pin must survive -- this clause is additive only."""
    assert "a stated truncation is an honest result, a silent one is a false completion" in _rendered()


def test_prompt_within_budget():
    assert prompt.prompt_len() <= prompt.PROMPT_BUDGET, (
        f"{prompt.prompt_len()} > {prompt.PROMPT_BUDGET}"
    )

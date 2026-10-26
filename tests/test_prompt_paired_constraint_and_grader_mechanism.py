"""Anti-attenuation tests for the two TB2.1 verifier-reverse-engineering clauses:
(1) A PAIRED CONSTRAINT LIVES IN THE GENERATOR, NOT IN YOUR HEAD
(2) WHEN NO RUNNABLE VERIFIER EXISTS, BUILD ONE WITH THE GRADER'S MECHANISM

Root causes (v12.2 batch trials): dna-assembly emitted the motif 128 times while
violating the stated minimum-index rule on every one (constraint existed only as
a remembered intention, never as a generator assert); filter-js delivered an HTML
sanitizer after manual inspection only — the grader ran it in headless Chrome with
28 batches of attack vectors and 2 batches slipped through.
"""
import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _paired_text():
    for line in SYSTEM_PROMPT_STATIC.splitlines():
        if "A PAIRED CONSTRAINT LIVES IN THE GENERATOR" in line:
            return line
    return ""


def _grader_text():
    for line in SYSTEM_PROMPT_STATIC.splitlines():
        if "WHEN NO RUNNABLE VERIFIER EXISTS, BUILD ONE WITH THE GRADER" in line:
            return line
    return ""


def test_paired_constraint_clause_present():
    assert "GENERATOR" in _paired_text()


def test_paired_clause_pins_generator_side_assert():
    text = _paired_text()
    assert "hard `assert`" in text
    # the measured loss it prevents: 128 emitted motifs, zero generator checks
    assert "128 times" in text


def test_paired_clause_names_the_anchor_point():
    text = _paired_text()
    # assert sits where the related values are produced, not in review
    assert "at the exact point" in text


def test_paired_clause_covers_named_standard_derived_rules():
    text = _paired_text()
    assert "source C" in text
    assert "generator invariant" in text


def test_grader_mechanism_clause_present():
    assert "headless Chrome" in _grader_text()


def test_grader_mechanism_clause_demands_real_engine():
    text = _grader_text()
    # not a parser approximation: the mechanism that RUNS the artifact
    assert "the real engine, not a parser approximation" in text


def test_grader_mechanism_clause_demands_adversarial_inputs():
    text = _grader_text()
    assert "attack vectors" in text
    assert "boundary values" in text


def test_both_clauses_survive_static_prompt_assembly():
    static = SYSTEM_PROMPT_STATIC
    assert "A PAIRED CONSTRAINT LIVES IN THE GENERATOR" in static
    assert "BUILD ONE WITH THE GRADER" in static

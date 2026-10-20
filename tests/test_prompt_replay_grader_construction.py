"""Falsifiable checks for the 'replay the implied derivation' discipline."""
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_replay_clause_present():
    assert "Have I replayed the derivation the task implies, against my delivered bytes?" in SYSTEM_PROMPT_STATIC


def test_replay_names_the_final_bytes_requirement():
    s = SYSTEM_PROMPT_STATIC
    assert "FINAL bytes at the delivery path" in s


def test_replay_requires_the_grader_relation_not_a_bare_pass():
    s = SYSTEM_PROMPT_STATIC
    assert "the same relation the task defines" in s
    assert "rather than a bare PASS" in s


def test_replay_requires_printing_numbers():
    assert "print the comparison's own numbers" in SYSTEM_PROMPT_STATIC


def test_replay_rejects_inspect_in_isolation():
    assert "the offset it finds is the first match" in SYSTEM_PROMPT_STATIC

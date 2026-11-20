"""E5 second half: a self-built proxy is never evidence about the consumer's definition."""
import pytest
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_e5_names_the_consumer_side_definition():
    p = SYSTEM_PROMPT_STATIC.lower()
    assert "no substitute for the consumer's definition" in p


def test_e5_lists_the_four_proxy_kinds_as_abstract_forms():
    p = SYSTEM_PROMPT_STATIC.lower()
    for probe in ("your own call", "question", "script", "wording"):
        assert probe in p, probe


def test_e5_orders_compute_both_then_ship_consumer_side():
    p = SYSTEM_PROMPT_STATIC.lower()
    i_both = p.find("compute both")
    i_ship = p.find("ship the consumer's")
    i_bet = p.find("record yours as a bet")
    assert i_both > 0 and i_ship > i_both and i_bet > i_ship


def test_e5_keeps_the_zero_hit_signature():
    assert "zero-hit count" in SYSTEM_PROMPT_STATIC.lower()


@pytest.mark.parametrize("literal", ["clamp", "mteb", "primer", "oligotm", "ggtctc", "raman"])
def test_e5_carries_no_task_literal(literal):
    assert literal not in SYSTEM_PROMPT_STATIC.lower()

"""Background/time-budget/multi-agent guidance — semantic checks.

Kept lean: these assert that the load-bearing CONCEPTS remain, not the exact
sentences. See tests/test_prompt_semantics.py for the concept-presence core.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_no_new_placeholders_introduced():
    assert set(re.findall(r"{([a-z_]+)}", SYSTEM_PROMPT_STATIC)) == {
        "cwd", "tools", "skills", "knowledge",
    }


def test_tool_guide_covers_long_running_classes():
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "background" in low
    assert "shell_jobs" in SYSTEM_PROMPT_STATIC
    assert "training" in low and ("build" in low or "download" in low)


def test_background_must_be_paired_with_other_work():
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "wait-spin" in low or "blocking in costume" in low or "bare wait" in low


def test_budget_order_preserved():
    assert "BUDGET ORDER" in SYSTEM_PROMPT_STATIC
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "crude" in low and ("scored" in low or "scores" in low)


def test_conveys_urgency_from_the_start():
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "scarce" in low and "time" in low


def test_overlaps_independent_expensive_steps():
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "longest branch" in low or "overlap" in low


def test_mentions_timebudget_advisory():
    assert "[TimeBudget]" in SYSTEM_PROMPT_STATIC


def test_multi_agent_present_and_tools_named():
    low = SYSTEM_PROMPT_STATIC.lower()
    for tool in ("spawn_worker", "dispatch_many", "poll_tasks"):
        assert tool in SYSTEM_PROMPT_STATIC
    assert "worker" in low and "reviewer" in low


def test_findings_are_claims_not_verdicts():
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "claims" in low and ("reproduce-or-refute" in low or "reproduce" in low)

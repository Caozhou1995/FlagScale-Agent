"""Background / time-budget / multi-agent guidance — semantic checks.

Kept lean: these assert that the load-bearing CONCEPTS remain, not the exact
sentences. See tests/test_prompt_semantics.py for the concept-presence core.

The prompt was rebuilt from scratch around a ten-section loop, so
the sentences these tests once froze no longer exist. Each assert below now
pins the CURRENT wording that carries the same concept, and each one goes red
if that wording is deleted:

  retired sentence                    -> current wording pinned here
  "shell_jobs" (named tool)          -> "run it in the background, poll it"
  "wait-spin" / "bare wait"          -> "Waiting is not work: a sleep loop
                                         produces nothing"
  "secure the deliverable" +         -> "secure the deliverable first" +
   "fastest finishing path"             "Land a crude, complete, scorable
                                         artifact at the delivery path EARLY"
  "scarce"                           -> "The clock is tight"
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_no_new_placeholders_introduced():
    assert set(re.findall(r"{([a-z_]+)}", SYSTEM_PROMPT_STATIC)) == {
        "cwd", "tools", "skills", "knowledge",
    }


def test_tool_guide_covers_long_running_classes():
    """Long-running work must be steered into the background and polled.

    the old prompt named the concrete tool `shell_jobs`; the current prompt inlines the
    mechanic instead ("run it in the background, poll it"), so the concept —
    not the tool name — is what survives and what is asserted.
    """
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "run it in the background, poll it" in low, (
        "prompt must tell the agent to run a long job in the background and poll it"
    )
    assert "do real work meanwhile" in low, (
        "backgrounding must be paired with doing other real work"
    )
    assert "long job" in low and "timeout" in low, (
        "prompt must bound a long job with a timeout"
    )


def test_background_must_be_paired_with_other_work():
    """The wait-spin failure mode, in its current wording."""
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "waiting is not work" in low, (
        "prompt must state that waiting is not work"
    )
    assert "a sleep loop produces nothing" in low, (
        "prompt must name the sleep-loop anti-pattern it forbids"
    )


def test_budget_order_preserved():
    # rebuild: the old heading "BUDGET ORDER" and the phrases "fastest
    # finishing path" / "refine" were retired (contract C11); the failure mode
    # they guarded — land a scorable deliverable before polishing — is stated
    # in §7 without the retired labels.
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "secure the deliverable first" in low, (
        "prompt must put securing the deliverable first under a time budget"
    )
    assert "land a crude, complete, scorable artifact" in low, (
        "prompt must demand an early, complete, scorable deliverable"
    )
    assert "improve it in place" in low, (
        "prompt must say the early artifact is then refined in place, not rewritten"
    )


def test_conveys_urgency_from_the_start():
    # "scarce" was the old word; the current phrase is "The clock is tight",
    # routed from §1's ROUTE BY SITUATION line into §7.
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "the clock is tight" in low, (
        "prompt must convey that time is scarce, in the current wording"
    )
    assert "four fifths of budget" in low, (
        "prompt must set the stop-exploring threshold against the budget"
    )


def test_overlaps_independent_expensive_steps():
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "the wall is the longest chain plus its join, not the sum" in low, (
        "prompt must state that fan-out wall time is the longest chain plus join"
    )
    assert "fan out" in low or "overlap" in low


def test_mentions_timebudget_advisory():
    assert "[TimeBudget]" in SYSTEM_PROMPT_STATIC


def test_multi_agent_present_and_tools_named():
    low = SYSTEM_PROMPT_STATIC.lower()
    for tool in ("spawn_worker", "dispatch_many", "poll_tasks"):
        assert tool in SYSTEM_PROMPT_STATIC
    assert "worker" in low and "reviewer" in low


def test_findings_are_claims_not_verdicts():
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "claims" in low and "reproduce or refute" in low, (
        "prompt must treat a worker's done / a reviewer's findings as claims "
        "to reproduce or refute"
    )


def test_budget_concepts_survive():
    """Cross-file anchor: the §7 budget the retired deliberation tests appealed to.

    tests/test_prompt_deliberation_budget.py retires its per-token tests but
    still relies on this concept being live: a time budget that must be priced
    before committing. This assert goes red if §7's pricing sentence is deleted.
    """
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "price before committing: unit cost times units against the budget left" in low, (
        "prompt must require pricing a route before committing to it"
    )

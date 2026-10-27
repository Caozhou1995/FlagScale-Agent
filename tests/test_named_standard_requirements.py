"""Named-standard requirements-read gate (instrument 1c).

Principle 2.3b: a requirement the task text POINTS AT is still a requirement.
When the task defers to an external standard ("satisfy the vendor's
requirements", "per the protocol"), the clauses live in the standard, not the
task text -- so a checklist built only from the task text looks complete while
being incomplete. The measured loss: the agent read a clause in the standard,
judged the checker unlikely to test it, dropped it, and failed on exactly that
clause while every clause it did satisfy passed.

The gate requires, AT PLAN CREATION, a clause list recording what the standard
said: one entry per clause, each with the clause text, where it came from, and
how the plan satisfies it (or an explicit 'dropped: <reason>').

The refusal message ships a FILLED-IN example (user directive: give the
finished shape, never slots), and the example carries NO task-specific tokens.
"""
import re

from flagscale_agent.react.instruments import (
    check_requirements_read,
    task_names_external_standard,
)
from flagscale_agent.react.tools.plan_create import PlanCreateTool


# A task text that defers to a standard (wording shape, no task tokens).
STANDARD_TASK = (
    "Design the layout. The result must satisfy the manufacturer's "
    "requirements for mounting clearance, and check that the assembly "
    "follows the protocol's documentation."
)
PLAIN_TASK = "Sort the list and write the result to out.txt."


def test_standard_task_detected():
    assert task_names_external_standard(STANDARD_TASK) is True


def test_plain_task_not_detected():
    assert task_names_external_standard(PLAIN_TASK) is False


def test_non_string_text_is_not_detected():
    assert task_names_external_standard(None) is False
    assert task_names_external_standard("") is False


def test_missing_list_refused():
    reason, msg = check_requirements_read(STANDARD_TASK, None)
    assert reason == "requirements_read_missing"
    assert "COPY THIS SHAPE" in msg


def test_empty_list_refused():
    reason, _ = check_requirements_read(STANDARD_TASK, [])
    assert reason == "requirements_read_missing"


def test_bare_string_refused_not_coerced():
    reason, msg = check_requirements_read(STANDARD_TASK, "read the manual")
    assert reason == "requirements_read_not_list"
    assert "list of clause objects" in msg


def test_entry_missing_clause_refused():
    reason, msg = check_requirements_read(STANDARD_TASK, [
        {"source": "s", "covered_by": "c"},
        {"clause": "clearance must be >= one unit", "source": "spec p1",
         "covered_by": "layout uses two units"},
    ])
    assert reason == "requirements_read_incomplete"
    assert "[0].clause" in msg


def test_placeholder_clause_refused():
    reason, msg = check_requirements_read(STANDARD_TASK, [
        {"clause": "n/a", "source": "s", "covered_by": "c"},
    ])
    assert reason == "requirements_read_incomplete"
    assert "placeholder" in msg


def test_missing_covered_by_refused():
    reason, msg = check_requirements_read(STANDARD_TASK, [
        {"clause": "clearance >= one unit", "source": "spec p1"},
    ])
    assert reason == "requirements_read_incomplete"
    assert "covered_by" in msg


def test_valid_list_passes():
    reqs = [
        {"clause": "protective base 5' of the site", "source": "protocol p2",
         "covered_by": "added to every segment"},
        {"clause": "span excludes the tail", "source": "protocol p3",
         "covered_by": "dropped: the bound is stated in the task itself"},
    ]
    assert check_requirements_read(STANDARD_TASK, reqs) is None


def test_dropped_clause_is_legal_with_reason():
    reqs = [{"clause": "c", "source": "s",
             "covered_by": "dropped: superseded by the task's own bound"}]
    assert check_requirements_read(STANDARD_TASK, reqs) is None


def test_refusal_message_example_is_neutral():
    reason, msg = check_requirements_read(STANDARD_TASK, None)
    for token in ("primer", "oligotm", "annealed", "mteb", "dna_insert"):
        assert token not in msg.lower()

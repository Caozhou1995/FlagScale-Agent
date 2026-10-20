"""The independent-side discipline must be stated as three EXECUTABLE conditions.

Motivation (measured on zero-leak runs, no checker reachable): a design task's agent
verified every constraint against the pieces as IT had laid them out, reported all
checks passing, and still failed, because re-parsing the same delivered bytes yielded
a different boundary and therefore a different length / paired difference. The lesson
must therefore be stated as conditions a reader can satisfy mechanically, not as
"be careful about definitions".

Each test below is written so that deleting or weakening its condition turns it red.
"""
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

FLAT = " ".join(SYSTEM_PROMPT_STATIC.split())


def test_requires_independent_side_reads_only_delivered_bytes():
    assert "reads ONLY the delivered bytes" in FLAT
    # and it must forbid substituting in-memory design values
    assert "never your in-memory design values" in FLAT


def test_requires_construction_not_reapplying_own_intent():
    assert "a construction you did NOT choose while designing" in FLAT
    assert "must accept ITS location" in FLAT


def test_requires_reporting_margin_not_just_pass_fail():
    assert "reports the quantity's margin against the limit" in FLAT
    assert "not just pass/fail" in FLAT


def test_names_the_failure_mode_it_closes():
    assert "as YOU laid them out" in FLAT
    assert "a re-parse of the same bytes yields a different" in FLAT


def test_conditions_stay_free_of_task_specifics():
    """No worked instance, no task vocabulary, no concrete numbers."""
    for leaked in ("primers", "oligotm", "Tm", "anneal", "plasmid",
                   "1e-4", "1.8e-2", "15-17", "56 nt"):
        assert leaked not in FLAT, f"task-specific token leaked: {leaked}"

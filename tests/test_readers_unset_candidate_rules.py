"""The preamble's rule 6 and 7 must survive, and must not leak task-specific answers.

Rules 6/7 were added 2026-11-15 after two independent failures where the
statement left something UNSET and the agent took the more expensive reading:

  * a scale/unit the statement never pinned -> the agent noticed the ambiguity
    but never tried the unknown-free candidate form;
  * two readings both satisfying the literal words -> the agent took the one
    requiring an irreversible change that reached beyond the named objects,
    destroying the reference the result was measured against.

These tests pin THREE properties, each of which has failed or could fail:

1. PRESENCE — the two clauses are actually in the rendered preamble (not just
   in a comment or a neighbouring constant).
2. NO LEAK — the clauses must not carry any task-specific value. The whole
   point of the reader block is to widen the agent's view of the statement,
   NOT to hand it the answer; a leaked constant would make the measurement
   meaningless. This test is the mechanical guard for that rule.
3. GENERICITY — the clauses must be stated as principles, and must not name
   any benchmark, dataset or task.
"""
import os
import re

os.environ["FLAGSCALE_TASK_READERS"] = "1"

from flagscale_agent.react.agent import WorkerAgent


def _render(n=2, word="two", pass_word="passes", votes="two votes"):
    # `nv` was added to the template when the preamble's hardcoded "ten" was
    # replaced by a derived count. Derive it here too, from the same source the
    # production call site uses, so this fixture cannot drift from it again.
    return WorkerAgent._READERS_PREAMBLE.format(
        n=n,
        nv=WorkerAgent._n_viewpoints(),
        word=word,
        pass_word=pass_word,
        votes=votes,
    )


def test_rule6_enumerate_candidates_present():
    pre = _render()
    assert "6. When the UNSET is a SCALE" in pre
    assert "ENUMERATE the candidate conventions and TRY them" in pre
    # the operative test: keep the candidate under which the target is visible
    assert "becomes VISIBLE and WELL-FORMED" in pre
    # (a) fewest additional unknowns
    assert "FEWEST additional unknowns" in pre
    # (b) monotone transforms are indistinguishable by shape -> test landing
    assert "monotone transform" in pre
    assert "Test the landing point, not the profile." in pre
    # (c) a remembered typical range must not outvote the material
    assert "weak evidence and can reject the right candidate" in pre


def test_rule7_prefer_smaller_reversible_change_present():
    pre = _render()
    assert "7. When two readings of the statement both satisfy its literal words" in pre
    assert "CHANGES ARE SMALLER AND REVERSIBLE" in pre
    # the reason must be stated, not just the instruction: unnamed things may be
    # the measurement's own reference
    assert "may be exactly what the result is measured against" in pre
    assert "break the tie toward what you can undo" in pre


# Task-specific values that must NEVER appear in the reader preamble. If any of
# these leaks in, the reader is handing over the answer instead of widening the
# view, and any measurement built on it is void.
_LEAK_BLACKLIST = [
    # raman-fitting grader values
    "1580", "2670", "9.06", "8382", "6329", "3745", "1e7", "graphene", "raman",
    # sanitize-git-repo task identity
    "d6987af", "dclm", "filter-repo", "sanitize",
    # generic answer-shaped numbers that would indicate a worked example
    "= 5", "approx 1", "multiply by",
]


def test_preamble_leaks_no_task_specific_value():
    pre = _render().lower()
    leaked = [tok for tok in _LEAK_BLACKLIST if tok.lower() in pre]
    assert not leaked, f"reader preamble leaks task-specific values: {leaked}"


def test_preamble_names_no_benchmark_or_dataset():
    pre = _render().lower()
    for name in ["terminal-bench", "terminalbench", "benchmark", "dataset", "leaderboard"]:
        assert name not in pre, f"reader preamble names {name!r}"


def test_rules_are_numbered_without_gaps():
    """Rule numbering must stay contiguous so an operator can cite 'rule 7'."""
    pre = _render()
    nums = [int(m.group(1)) for m in re.finditer(r"^(\d+)\. ", pre, re.M)]
    assert nums == list(range(1, len(nums) + 1)), f"rule numbering broken: {nums}"
    assert len(nums) >= 7, f"expected at least 7 rules, found {nums}"

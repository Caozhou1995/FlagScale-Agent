"""Semantic tests for SYSTEM_PROMPT_STATIC.

These assert the prompt's STRUCTURE and the PRESENCE OF CONCEPTS, not exact
sentences. A prompt is prose under active revision; freezing its wording in
tests makes every rewrite fail for no behavioral reason, and — worse — lets a
real regression hide behind a thousand irrelevant breakages. What must not
silently disappear is the presence of each load-bearing idea and the format
contract; how it is phrased is free to change.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


# ── Format contract (load-bearing, exact) ────────────────────────────────

def test_formats_without_keyerror():
    """The builder .format()s this string with these placeholders."""
    out = SYSTEM_PROMPT_STATIC.format(
        cwd="/c", tools="t", skills="s", knowledge="k",
        # legacy placeholders the builder may still pass:
        critical_rules="", optional_sections="", skill_context="",
    )
    assert len(out) > 1000


def test_placeholders_are_exactly_the_expected_set():
    found = set(re.findall(r"{([a-z_]+)}", SYSTEM_PROMPT_STATIC))
    assert found == {"cwd", "tools", "skills", "knowledge"}


def test_is_english():
    assert not re.search(r"[\u4e00-\u9fff]", SYSTEM_PROMPT_STATIC)


def test_no_python_format_brace_leak_in_prose():
    # Every literal brace must be doubled so .format() never raises.
    assert "{{" in SYSTEM_PROMPT_STATIC  # the tool-param example is escaped


# ── Concept presence (loose — survives paraphrase) ───────────────────────

CONCEPTS = {
    # (name, [alternative substrings — any one present])
    "research_first": ["problem class", "standard method", "route the task"],
    "cost_arithmetic": ["cost arithmetic", "wall-clock", "sample and extrapolate"],
    "grader_contract": ["grader's contract", "acceptance clause", "testable assertion"],
    "observation_over_argument": ["observation", "argument", "verification"],
    "not_verified_debt": ["not verified", "load-bearing premise", "falsify"],
    "far_end_verification": ["far end", "target's own state", "consumer"],
    "generalization": ["generalizes", "overfit", "stress input"],
    "constraint_loyalty": ["non-negotiable", "given", "range"],
    "anti_fabrication": ["fabricat", "blocked"],
    "irreversibility": ["hard-to-undo", "recovery path", "stricter one"],
    "scope_gate": ["within the task", "widen", "narrower one"],
    "citation_precision": ["re-verify", "same turn", "grep"],
    "logical_undo": ["logical undo", "byte restore", "operate on a copy"],
    "escape_downward_upward": ["downward", "upward", "sideways"],
    "stall_information_gain": ["information gain", "stalled", "did not already know"],
    "invisible_grader": ["invisible", "hedge", "predicate"],
    "minimal_uncertainty": ["minimal-uncertainty", "widening the uncertainty", "no more"],
    "direction_of_loss": ["direction-of-loss", "inclusion", "omission"],
    "direction_of_comparison": ["normalized", "one-sided", "strictest predicate"],
    "plan_os": ["plan", "frame", "step_done"],
    "plan_gated_by_acting": ["gated", "too simple", "disarm"],
    "memory": ["memory", "cross-session", "falsifiable"],
    "tool_guide": ["background", "shell", "write_file"],
    "multi_agent": ["worker", "reviewer", "dispatch"],
    "time_budget": ["time budget", "budget order", "longest branch"],
    "environment_resilience": ["proxy", "mirror", "unreachable"],
    "resource_probing": ["cgroup", "container", "ladder"],
    "code_quality": ["pytest", "unit test", "commit message"],
    "response_format": ["task result", "rsi result", "language"],
}


def test_every_load_bearing_concept_present():
    low = SYSTEM_PROMPT_STATIC.lower()
    missing = []
    for name, alts in CONCEPTS.items():
        if not any(a.lower() in low for a in alts):
            missing.append((name, alts))
    assert not missing, f"prompt lost these concepts: {missing}"


def test_three_principles_survive():
    assert "principle 1" in SYSTEM_PROMPT_STATIC.lower()
    assert "principle 2" in SYSTEM_PROMPT_STATIC.lower()
    assert "principle 3" in SYSTEM_PROMPT_STATIC.lower()


def test_integrity_red_lines_present():
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "never" in low and "fabricat" in low
    assert "never claim done without an observation" in low


def test_reply_marker_contract_present():
    assert "[TASK_COMPLETE]" in SYSTEM_PROMPT_STATIC
    assert "[NEED_USER_INPUT]" in SYSTEM_PROMPT_STATIC


# ── Anti-leak (must never regress) ───────────────────────────────────────

def test_no_grader_oracle_leak():
    p = SYSTEM_PROMPT_STATIC
    for bad in ("BeautifulSoup", "html.parser", "html5lib", "lxml", "str(soup)",
                "normalization that may occur"):
        assert bad not in p, f"prompt leaked grader oracle token: {bad!r}"


def test_no_task_specific_magic_numbers():
    # No tuned constants / task-specific identifiers hardcoded into the prompt.
    assert not re.search(r"0x[0-9a-fA-F]{4,}", SYSTEM_PROMPT_STATIC)
    for w in ("ffmpeg", "pandas", "numpy"):
        assert w not in SYSTEM_PROMPT_STATIC.lower()

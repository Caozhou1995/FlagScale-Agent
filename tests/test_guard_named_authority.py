"""NamedAuthorityPhase: fires on the SHAPE "the statement hands a requirement to
an outside party", and on nothing else.

The load-bearing number in this file is the FALSE-POSITIVE rate. This phase
blocks real work until the agent records a sub-requirement enumeration, so a
trigger that fires on ordinary tasks pays a whole turn on every one of them.
Measured on the harness's own 90-task set, deduplicated by task name: 1/90
fired, and that one is the task whose grader checks a sub-requirement the
statement never wrote out. A trigger that drifts wider than that turns this
phase into a tax.

ALSO: the phase's message is rendered into every context that trips it, so it
may not contain any task's own vocabulary. An illustration built from a live
task's words is answer leakage — the same mistake this session made once in
net_awareness.py and the gate below now catches mechanically.
"""

from __future__ import annotations

import re

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.named_authority import (
    NamedAuthorityPhase,
    names_external_authority,
)


# ── 1. Trigger: the shape it must catch ──────────────────────────────────────
@pytest.mark.parametrize(
    "statement",
    [
        # candidate-named, possessive
        "check that the sites you design satisfy VENDORX's requirements",
        # standard body, acronym
        "the output must comply with IEEE 754 requirements",
        # "per the requirements of <Named>"
        "design per the requirements of the OpenAPISpec specification",
        # possessive on a named org
        "according to ACMECORP's protocol, use the following steps",
        # conformance verb + institution
        "your file must conform to the ISO8601 standard",
        # as-required-by, named party
        "the header must be written as required by RFCX",
    ],
)
def test_fires_on_named_authority(statement: str) -> None:
    assert names_external_authority(statement) is True, statement


# ── 2. Trigger: the shapes it must NOT catch ─────────────────────────────────
@pytest.mark.parametrize(
    "statement",
    [
        # the bound is IN the statement — nothing to go read
        "satisfy the length requirement of at most 5000 bytes",
        "the answer must satisfy the constraints stated above",
        "follow the requirements listed below",
        # bare noun, no outside party
        "write a JSON file",
        "use a standard file format",
        "compute the mean of the values",
        # per- instructions that are in the statement
        "per the instructions below",
        # prose that merely mentions a repo
        "Write a regex matching all valid moves. See the reference in the repo.",
        "",
    ],
)
def test_stays_silent_without_an_outside_party(statement: str) -> None:
    assert names_external_authority(statement) is False, statement


def test_sentence_wide_window_catches_post_nominal_party() -> None:
    """Regression: "<cue> the requirements OF <Party>" puts the party AFTER the
    matched nouns. A span-bounded check missed exactly this ordering."""
    assert names_external_authority(
        "implement it per the requirements of ZephyrWorks"
    )


# ── 3. Phase behaviour ───────────────────────────────────────────────────────
def _ctx(cmd: str = "", messages: list[dict] | None = None,
         tool_result: str | None = None, tool_name: str = "shell"):
    return GuardContext(
        tool_name=tool_name,
        tool_args={"command": cmd} if cmd else {},
        tool_result=tool_result,
        messages=messages or [],
    )


_STATEMENT = [{"role": "user", "content":
               "Do the thing and make sure it satisfies VENDORX's requirements."}]
_PLAIN = [{"role": "user", "content": "Write a file containing the numbers 1 to 10."}]


def test_blocks_real_work_when_armed() -> None:
    ph = NamedAuthorityPhase()
    v = ph.check(_ctx("cat input.txt", _STATEMENT))
    assert v is not None
    assert v.reason == "startup_named_authority"
    assert v.overridable is False, "a self-issued exemption is the measured failure"


def test_disarmed_leaves_ordinary_tasks_alone() -> None:
    ph = NamedAuthorityPhase()
    assert ph.check(_ctx("cat input.txt", _PLAIN)) is None
    assert ph.is_satisfied() is True, "a disarmed phase must report satisfied"


def test_no_statement_found_means_disarmed() -> None:
    """A phase that cannot see the statement must not guess."""
    ph = NamedAuthorityPhase()
    assert ph.check(_ctx("cat input.txt", [])) is None


def test_research_and_sweep_acts_are_never_blocked() -> None:
    ph = NamedAuthorityPhase()
    assert ph.check(_ctx("", _STATEMENT, tool_name="web_fetch")) is None
    assert ph.check(_ctx("", _STATEMENT, tool_name="load_knowledge")) is None
    ph2 = NamedAuthorityPhase()
    assert ph2.check(_ctx("curl -sI https://example.invalid", _STATEMENT)) is None


def test_one_recording_act_satisfies_it() -> None:
    ph = NamedAuthorityPhase()
    ph.check(_ctx("cat input.txt", _STATEMENT))
    ph.observe_post(_ctx("", _STATEMENT, "written", tool_name="plan_update"))
    assert ph.is_satisfied() is True


def test_blocked_payload_does_not_count_as_the_recording_act() -> None:
    ph = NamedAuthorityPhase()
    ph.check(_ctx("cat input.txt", _STATEMENT))
    ph.observe_post(_ctx("cat input.txt", _STATEMENT,
                         "[BLOCKED BY GUARD] ...", tool_name="plan_update"))
    assert ph.is_satisfied() is False, (
        "a guard's own refusal satisfied the phase it refused"
    )


def test_override_is_refused() -> None:
    ph = NamedAuthorityPhase()
    ph.check(_ctx("cat input.txt", _STATEMENT))
    assert ph.accept_override("the requirement looks optional", _ctx()) is False


# ── 4. No task literal in the rendered message ───────────────────────────────
_HOSTNAME = re.compile(
    r"(?:[a-z0-9-]+\.)+(?:com|org|net|cn|io|sh|dev|edu|gov)\b", re.I)
_TASK_VOCAB = (
    "neb", "bsai", "ggtctc", "clamp", "golden gate", "oligotm", "overhang",
    "primer", "plasmid", "dna", "restriction", "anneal",
    "make-doom", "mips", "codegolf", "raman", "tensor-parallel",
)


def test_message_has_no_hostname() -> None:
    from flagscale_agent.react.guard.named_authority import _SWEEP_MESSAGE
    assert not _HOSTNAME.findall(_SWEEP_MESSAGE)


def test_message_has_no_task_vocabulary() -> None:
    from flagscale_agent.react.guard.named_authority import _SWEEP_MESSAGE
    low = _SWEEP_MESSAGE.lower()
    hits = [w for w in _TASK_VOCAB if w in low]
    assert not hits, (
        f"task vocabulary in a message rendered into every tripped context: {hits}"
    )


def test_vocabulary_gate_can_go_red() -> None:
    probe = "the clamp must precede the site"
    assert [w for w in _TASK_VOCAB if w in probe.lower()]


# ── 5. The message must carry the same route axes as the guard ───────────────
def test_message_carries_all_three_route_axes() -> None:
    from flagscale_agent.react.guard.named_authority import _SWEEP_MESSAGE
    low = _SWEEP_MESSAGE.lower()
    for axis in ("address", "provenance", "local"):
        assert axis in low, f"the sweep lost the {axis!r} axis"
    assert "unconsulted" in low, "the sweep lost its exhausted-case fallback"
    assert "conservative" in low, "the sweep lost its conservative reading rule"

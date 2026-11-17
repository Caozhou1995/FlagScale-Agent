"""NetAwarenessGuard: the two tiers, and the no-task-literal gate.

The guard used to carry ONE advisory. It now carries two, because the two
worlds need different reminders:

  tier 1 (network_acquire) — a network step is about to be / has been taken and
    the agent needs to know how to READ a failure (dns / conn / http / total,
    direct-vs-proxy, host-root-vs-path).
  tier 2 (network_refused) — the attempt's OWN OUTPUT already carries a
    refusal/lookup verdict, so the diagnostic ladder is behind it. What it needs
    to hear is that a refused ROUTE is not a missing FACT, and to switch axes
    (address / provenance / client / local) before reporting the thing as
    unattainable.

Measured motivation (2 trials, same task, reward=0 both): the agent spent every
lookup on one refused host and then reported the named standard itself as
unreadable, while the same fact was reachable from the same machine by a
different route. Tier 1's advice had already been delivered and already been
read; tier 2 is the part that was missing.

ALSO IN THIS FILE — the task-literal gate. While adding tier 2 to this guard I
twice wrote a real task's own vocabulary into the message as an illustration
("A <task keyword> IS NOT A CLUE"). That is answer leakage: the message is
rendered into every model context, so an illustration built from a live task's
words hands the model the very step the task means to test. The gate below
makes it mechanical instead of a promise: the message text must contain no
hostname, no memory-key literal, and none of the vocabulary of any task the
harness is known to run.
"""

from __future__ import annotations

import re

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.net_awareness import (
    NetAwarenessGuard,
    _looks_refused,
    is_network_acquire,
)


def _ctx(tool_name: str, command: str = "", tool_result: str | None = None):
    return GuardContext(
        tool_name=tool_name,
        tool_args={"command": command} if command else {},
        tool_result=tool_result,
    )


# ── 1. Tier selection ────────────────────────────────────────────────────────
def test_first_attempt_gets_the_diagnostic_tier() -> None:
    """No refusal in the output → the how-to-READ-the-number tier."""
    g = NetAwarenessGuard()
    v = g.check_post(_ctx("shell", "pip install x", "Successfully installed x"))
    assert v is not None
    assert v.reason == "network_acquire"
    assert "time_namelookup" in v.message, "tier 1 lost its DNS/TCP field"


@pytest.mark.parametrize(
    "output",
    [
        "403",                       # bot-block / at-the-door refusal
        "HTTP 403 Forbidden",
        "404",                       # wrong slug
        "curl: (22) The requested URL returned error: 404",
        "000",                       # nothing connected
        "429 Too Many Requests",
        "410 Gone",
    ],
)
def test_refused_output_gets_the_route_enumeration_tier(output: str) -> None:
    g = NetAwarenessGuard()
    v = g.check_post(_ctx("shell", "curl -sL https://example.invalid/x", output))
    assert v is not None
    assert v.reason == "network_refused", (
        f"a refused output ({output!r}) got the tier-1 reminder; the ladder is "
        "behind it and it needs the route-enumeration half"
    )
    low = v.message.lower()
    for axis in ("address", "provenance", "local"):
        assert axis in low, f"tier 2 lost the {axis!r} axis"
    assert "unconsulted" in low, "tier 2 lost the UNCONSULTED fallback"


def test_successful_output_does_not_get_the_refusal_tier() -> None:
    """Anti-overfire: a 200 is not a refusal, and neither is a 2xx body."""
    g = NetAwarenessGuard()
    for out in ("200", "HTTP/1.1 200 OK\n<8000 bytes>", "total=0.31s http=200"):
        v = g.check_post(_ctx("shell", "curl -s https://example.invalid/x", out))
        assert v is not None and v.reason == "network_acquire", out


def test_no_result_yet_gets_the_diagnostic_tier() -> None:
    g = NetAwarenessGuard()
    v = g.check_post(_ctx("shell", "apt-get install -y x", None))
    assert v is not None and v.reason == "network_acquire"


def test_non_network_command_stays_silent_in_both_tiers() -> None:
    g = NetAwarenessGuard()
    for out in (None, "403", "404"):
        assert g.check_post(_ctx("shell", "ls -la", out)) is None
    assert g.check_post(_ctx("read_file", "", "404")) is None


def test_never_blocks_in_pre() -> None:
    g = NetAwarenessGuard()
    assert g.check_pre(_ctx("shell", "pip install x", "403")) is None


# ── 2. The refusal detector itself ───────────────────────────────────────────
@pytest.mark.parametrize(
    "out,expect",
    [
        ("403", True), ("404", True), ("000", True), ("429", True),
        ("HTTP 403 Forbidden", True),
        ("200", False), ("", False), ("<huge body with 4030 matches>", False),
        ("wrote 4040 bytes", False), ("12.403s", False),
    ],
)
def test_looks_refused_is_word_bounded(out: str, expect: bool) -> None:
    """A code inside a larger number is not a status.

    Measured risk while writing this: an unbounded substring test makes
    'wrote 4040 bytes' and '12.403s' read as refusals and fires tier 2 on a
    successful transfer.
    """
    assert _looks_refused(out) is expect


# ── 3. Generality + no task literal (the mechanical form of the user's rulings) ─
_HOSTNAME = re.compile(
    r"(?:[a-z0-9-]+\.)+(?:com|org|net|cn|io|sh|dev|edu|gov)\b"
    r"|\b(?:tuna|aliyun|huaweicloud|tencent|tsinghua|douban|ustc)\b",
    re.I,
)
_MEMKEY = re.compile(
    r"memory_read\(\s*['\"][^'\"]+['\"]"
    r"|memory_list\(\s*keyword\s*=\s*['\"][^'\"]+['\"]"
    r"|['\"](?:fact|pitfall|insight)/"
)

#: Vocabulary drawn from real tasks in this harness's own benchmark set. The
#: guard's text is rendered into EVERY model context, so none of it may appear
#: here: an illustration built from a live task's words is answer leakage.
#: Kept as data (not prose) so adding a task means adding one token.
_TASK_VOCAB = (
    "neb", "bsai", "ggtctc", "clamp", "golden gate", "oligotm", "overhang",
    "primer", "plasmid", "dna", "restriction", "anneal", "tm",
    "make-doom", "mips", "codegolf", "raman", "tensor-parallel",
)


def _all_message_text() -> str:
    import flagscale_agent.react.guard.net_awareness as m
    return "\n".join(
        getattr(m, n) for n in ("_MESSAGE", "_RETRY_MESSAGE")
    )


def test_guard_contains_no_hostname() -> None:
    hits = _HOSTNAME.findall(_all_message_text())
    assert not hits, f"hostname/mirror literal baked into a general guard: {hits}"


def test_guard_contains_no_memory_key_literal() -> None:
    hits = _MEMKEY.findall(_all_message_text())
    assert not hits, f"memory-key literal baked into a general guard: {hits}"


def test_guard_contains_no_task_vocabulary() -> None:
    """THE GATE FOR THE MISTAKE THIS SESSION ACTUALLY MADE."""
    low = _all_message_text().lower()
    hits = [w for w in _TASK_VOCAB if w in low]
    assert not hits, (
        f"task vocabulary leaked into a message rendered into every context: "
        f"{hits}. An example is the answer; replace it with the abstract form."
    )


def test_task_vocabulary_gate_can_go_red() -> None:
    """A gate that cannot fail is not a gate."""
    probe = "A CLAMP IS NOT A CLUE"
    assert [w for w in _TASK_VOCAB if w in probe.lower()], (
        "the vocabulary gate would pass a message containing a real task's word"
    )


def test_registered_in_agent_build() -> None:
    """The guard must actually be wired, or nothing above reaches a run."""
    import flagscale_agent.react.agent as a
    src = open(a.__file__, encoding="utf-8").read()
    assert "NetAwarenessGuard" in src, "guard not registered in _build_kernel"

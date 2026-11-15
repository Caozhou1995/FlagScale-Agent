"""NetAwarenessGuard: a failed acquisition removes the ROUTE, not the REQUIREMENT.

Origin (measured, batch 2026-11-15__21-24-25 / dna-assembly):
The statement's last line told the agent to check an external vendor standard.
Its network attempts were, in order:
  · `curl ...` -> `/bin/sh: 1: curl: not found` (exit 127) — the CLIENT was
    absent; this says nothing about the network at all;
  · `urllib` -> `HTTP Error 403: Forbidden`;
  · same site via proxy -> `Tunnel connection failed: 500`;
then it wrote "NEB's site is 403-blocked from here (I'll label that read
UNCONSULTED)" and stopped. It tried one address plus one encyclopaedia page and
no other source. The standard was never read, and the element it would have
supplied was silently dropped.

The guard must therefore say, generically and without naming any host: a failed
fetch is not the end of the requirement — there are OTHER routes (a different
client, another address for the same content, the source reached indirectly,
or the material already present locally), and if all of them fail the reading is
a gap to fill conservatively, never a requirement to drop.

Four properties, each of which could regress:
1. FIRE     — the guard still fires on transfer-tool commands (no regression the
              old coverage) and still does NOT fire on local commands.
2. CONTENT  — the new route clauses are in the injected message.
3. GENERAL  — the guard names no host, no mirror, and no task/benchmark.
4. RED-ABLE — the content assertions reject a message with the clause removed.
"""

from __future__ import annotations

import re

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.net_awareness import (
    _MESSAGE,
    NetAwarenessGuard,
    is_network_acquire,
)


def _ctx(cmd: str, result: str | None = None) -> GuardContext:
    return GuardContext(tool_name="shell", tool_args={"command": cmd},
                        tool_result=result)


# ── 1. Fire / do-not-fire (no regression over the old coverage) ──────────────
def test_fires_on_a_transfer_tool_probe():
    """A bare `curl -sI <url>` still pays the network — it must still fire."""
    g = NetAwarenessGuard()
    assert is_network_acquire("curl -sI https://host.invalid/path")
    assert g.check_post(_ctx("curl -sI https://host.invalid/path")) is not None


def test_fires_regardless_of_the_result():
    """The guard is a REMINDER, not a judge: success and failure both fire.

    Deliberate: deciding whether a transfer 'worked' from output text is a
    guess, and the same command can succeed on a retry. What changes with the
    result is which route the agent should consider next, and that is stated in
    one message rather than branched on here.
    """
    g = NetAwarenessGuard()
    ok = _ctx("curl -o f.tgz https://host.invalid/f", "100%|##########| 42k")
    fail = _ctx("curl -o f.tgz https://host.invalid/f", "HTTP Error 403: Forbidden")
    assert g.check_post(ok) is not None
    assert g.check_post(fail) is not None


def test_does_not_fire_on_local_commands():
    g = NetAwarenessGuard()
    for cmd in ["ls -la", "python3 x.py", "git status", "cat f | head",
                "echo 'pip install x'", "grep -n foo bar.py"]:
        assert not is_network_acquire(cmd), f"false positive on {cmd!r}"
        assert g.check_post(_ctx(cmd)) is None, f"guard fired on {cmd!r}"


def test_ignores_non_shell_tools():
    g = NetAwarenessGuard()
    assert g.check_post(GuardContext(tool_name="read_file",
                                     tool_args={"path": "/x"},
                                     tool_result="")) is None


# ── 2. Content of the new route clauses ─────────────────────────────────────
def test_message_states_a_failed_acquisition_removes_route_not_requirement():
    m = _MESSAGE
    assert "a failed acquisition removes the ROUTE, not the REQUIREMENT".lower() \
        in m.lower() or "removes the ROUTE, not the REQUIREMENT" in m
    assert "the routes you have not tried yet" in m.lower()


def test_message_covers_the_four_generic_routes():
    m = _MESSAGE
    # (a) the client itself may be absent — the measured exit-127 case
    assert "the client itself may be absent or wrong" in m
    assert "says nothing about the network" in m
    # (b) same content at more than one address
    assert "MORE THAN ONE address" in m
    assert "do not generalise from one address refusing you" in m
    # (c) the source may be reachable only indirectly
    assert "reachable only indirectly" in m
    # (d) it may already be present locally
    assert "may already be present locally" in m


def test_message_forbids_dropping_the_requirement_after_all_routes_fail():
    m = _MESSAGE
    assert "record it as UNCONSULTED" in m
    assert "most conservative reading" in m
    # the two conversions that must not happen
    assert "never a reason to drop what it was going to tell you" in m
    assert "never a licence to call the requirement satisfied" in m
    # and bounded effort is required, not unbounded retries
    assert "a few genuinely DIFFERENT routes, not the same one repeated" in m


# ── 3. Generality (the guard must fit any environment) ──────────────────────
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
_TASKWORD = re.compile(
    r"terminal-?bench|dna-assembly|raman-fitting|sanitize-git-repo|"
    r"write-compressor|ggtctc|ggctac|bsai|clamp|neb\.com|1580|2670|1e7",
    re.I,
)


def _guard_source() -> str:
    import flagscale_agent.react.guard.net_awareness as m
    return open(m.__file__, encoding="utf-8").read()


def test_new_clauses_name_no_host_and_no_memory_key():
    src = _guard_source()
    assert not _HOSTNAME.findall(src), "hostname/mirror literal in a general guard"
    assert not _MEMKEY.findall(src), "memory key literal in a general guard"


def test_new_clauses_name_no_task_or_benchmark_value():
    """The reminder must widen the view, never hand over an answer."""
    src = _guard_source()
    hits = _TASKWORD.findall(src)
    assert not hits, f"task-specific value baked into the guard: {hits}"


# ── 4. The content assertions must be able to go red ────────────────────────
def test_content_check_can_go_red():
    m = _MESSAGE
    stripped = m.replace("the routes you have not tried yet", "")
    assert "the routes you have not tried yet" in m
    assert "the routes you have not tried yet" not in stripped

    stripped2 = m.replace("record it as UNCONSULTED", "")
    assert "record it as UNCONSULTED" in m
    assert "record it as UNCONSULTED" not in stripped2

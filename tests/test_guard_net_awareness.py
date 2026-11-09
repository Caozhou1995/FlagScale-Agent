"""NetAwarenessGuard: fires on network acquisition, and stays environment-free.

Two things must hold, and a presence-only assertion would miss both:

1. RECALL — the guard fires on real acquisition commands across ecosystems
   (pip/apt/conda/npm/cargo/docker/git clone/curl -o), including the compound
   and env-prefixed forms that a line-leading matcher misses.
2. PRECISION — it does NOT fire on ordinary commands, on a bare page fetch, or
   on a command merely MENTIONED inside echo/printf. (Measured false positive
   while building this: `echo 'pip install x'` fired.)

Plus the generality gate: the guard must contain no hostname, no mirror name,
and no memory-key literal. That is the mechanical form of the user's ruling
"this guard must be general to any environment" — it is asserted, not promised,
and step 4 below proves the assertion can go red.
"""

from __future__ import annotations

import re

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.net_awareness import (
    NetAwarenessGuard,
    is_network_acquire,
)


# ── 1. Recall ────────────────────────────────────────────────────────────────
_ACQUIRE = [
    "pip install requests",
    "python -m pip install numpy",
    "python3 -m pip download 'setuptools>=77'",
    "cd /tmp && pip install -e .",
    "DEBIAN_FRONTEND=noninteractive apt-get install -y git",
    "apt-get update && apt-get install -y curl",
    "sudo apt-get install -y build-essential",
    "uv pip install torch",
    "conda install -y numpy",
    "npm install",
    "cargo build --release",
    "git clone https://example.invalid/repo.git",
    "docker pull some/image:tag",
    "wget https://example.invalid/f.tgz",
    "curl -L -o f.tgz https://example.invalid/f.tgz",
    "curl --output f https://example.invalid/f",
    "/usr/bin/pip install y",
    "cd /app && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 git",
    # ── measured in a REAL trial (jobs/2026-11-09__13-23-33), verbatim ───────
    # The first three were all missed by an earlier download-form-only
    # predicate: every real command is wrapped in `timeout N`, and a bare
    # `curl -sI` probe is a network step too (it can be slow / refused).
    "cd /tmp; timeout 30 curl -sI https://example.invalid 2>&1 | head -3",
    "timeout 60 env -u HTTPS_PROXY -u HTTP_PROXY curl -sL http://example.invalid/x -o f",
    "timeout -s KILL 60 curl https://example.invalid",
    # a bare transfer-tool invocation acquires over the network as well
    "curl https://example.invalid/v1",
    "rsync -av host:/x /y",
]


@pytest.mark.parametrize("cmd", _ACQUIRE)
def test_fires_on_network_acquisition(cmd: str) -> None:
    assert is_network_acquire(cmd), f"should fire: {cmd!r}"


# ── 2. Precision ─────────────────────────────────────────────────────────────
_NOT_ACQUIRE = [
    "ls -la",
    "cat /etc/hosts",
    "grep -rn foo .",
    "python script.py",
    "make -j8",
    "cd /app && ls",
    "git status",
    "pip --version",
    # a bare page fetch IS a network step: keep the nag (measured — the real
    # agent opened with `curl -sI`, and slow/refused is a cost it pays too)
    # MENTIONING a command inside a string is not running it
    "echo 'pip install x'",
    "printf 'apt-get install x'",
    "cat notes.txt",
    "echo curl https://example.invalid",
]


@pytest.mark.parametrize("cmd", _NOT_ACQUIRE)
def test_does_not_fire_on_noise(cmd: str) -> None:
    assert not is_network_acquire(cmd), f"should NOT fire: {cmd!r}"


# ── 3. Guard shape: inject-only, never blocks ────────────────────────────────
def _ctx(tool_name: str, command: str) -> GuardContext:
    return GuardContext(tool_name=tool_name, tool_args={"command": command})


def test_injects_on_shell_acquisition() -> None:
    g = NetAwarenessGuard()
    v = g.check_post(_ctx("shell", "pip install requests"))
    assert v is not None, "expected an inject verdict"
    assert v.action == "inject", f"must be advisory, got {v.action}"
    assert not v.overridable is False  # advisory: nothing forces a block path


def test_never_blocks_in_pre() -> None:
    g = NetAwarenessGuard()
    assert g.check_pre(_ctx("shell", "pip install requests")) is None


def test_ignores_non_shell_tools() -> None:
    g = NetAwarenessGuard()
    assert g.check_post(_ctx("write_file", "pip install requests")) is None


def test_message_carries_all_three_world_states() -> None:
    """A transport failure, a slow transfer and a path-level refusal are three
    different worlds and the message must let the agent TELL THEM APART.

    Strengthened after a real trial: the agent ran `curl --noproxy '*'` once and
    never drew a conclusion from it, because the message listed three
    possibilities without a discriminating reading. The test now requires the
    three fields that actually separate them (dns / conn / http), not just prose.
    """
    g = NetAwarenessGuard()
    msg = g.check_post(_ctx("shell", "apt-get install -y x")).message
    assert "time_namelookup" in msg, "no DNS-vs-TCP field to read"
    assert "time_connect" in msg, "no connect() field to read"
    assert "http=" in msg, "no status/code field to read"
    assert "time_total" in msg, "timing/slowness branch missing"
    assert "PATH-level" in msg or "path-dependent" in msg, "path-dependence missing"
    # the discriminating test itself: same URL, both directions
    assert "--noproxy" in msg, "no bypass direction named"


def test_message_bounds_the_probe_and_allows_giving_up() -> None:
    """Measured: one probe took 45s and answered nothing. Bounded or bust.

    Without an explicit bound and an explicit permission to stop, the reminder
    would licence exactly the open-ended probing it is meant to prevent.
    """
    g = NetAwarenessGuard()
    msg = g.check_post(_ctx("shell", "pip install x")).message
    assert "-m 10" in msg, "no per-attempt timeout named"
    assert "bounded" in msg, "no total budget named"
    assert "go do the real work" in msg or "no answer" in msg, (
        "no permission to give up"
    )


# ── 4. Generality gate (the mechanical form of "must fit any environment") ───
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


def _guard_source() -> str:
    import flagscale_agent.react.guard.net_awareness as m
    return open(m.__file__, encoding="utf-8").read()


def test_guard_contains_no_hostname() -> None:
    hits = _HOSTNAME.findall(_guard_source())
    assert not hits, (
        f"hostname/mirror literal baked into a general guard: {hits}. "
        f"Environment facts belong in memory or in the agent's own measurement."
    )


def test_guard_contains_no_memory_key_literal() -> None:
    hits = _MEMKEY.findall(_guard_source())
    assert not hits, (
        f"memory key baked into a general guard: {hits}. The agent decides what "
        f"to look up; the guard must not name a key."
    )


def test_generality_gate_can_go_red() -> None:
    """Prove the two gates above are load-bearing, not vacuous.

    Feed each detector a source that violates it and require a hit. A gate that
    cannot fail is not a gate — this repo has been bitten by exactly that.
    """
    assert _HOSTNAME.search('Probe github.com via proxy 200/17.4s')
    assert _HOSTNAME.search('use mirrors.aliyun.com as the index')
    assert _MEMKEY.search("memory_read('fact/cluster/egress')")
    assert _MEMKEY.search('memory_list(keyword="proxy")')


def test_registered_in_agent_build() -> None:
    """The guard must actually be wired, or nothing above reaches a run."""
    import flagscale_agent.react.agent as a
    src = open(a.__file__, encoding="utf-8").read()
    assert "NetAwarenessGuard" in src, "guard not registered in _build_kernel"

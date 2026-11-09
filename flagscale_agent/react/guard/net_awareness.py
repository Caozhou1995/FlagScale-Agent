"""One inject-only advisory for shell commands that touch the network.

DESIGN (user ruling 2026-11-09, final): ONE guard. No duration plumbing, no
kernel change, no separate "slow path" mechanism, no mirror table, no memory
key. The guard's whole job is: notice that this shell command acquires
something over the network, and inject ONE reminder that raises the agent's
awareness that a network step can fail, be slow, or be path-dependent -- then
let the agent handle it with its own judgement.

Why awareness instead of machinery:
  * Measured, 5 batches / 195 trials: real external-egress errors appear in
    0.5% of trials. Any fixed-cost mechanism is paid 100% of the time to serve
    0.5%.
  * Measured: two runs of the SAME probe took 45.0s and 16.0s and both timed
    out, while a different path under the same host answered in 1.3s in the same
    window. So "is this host up?" is the wrong granularity and a probe can
    itself be expensive -- a guard that DECIDES would be guessing. A guard that
    REMINDS leaves the deciding to the agent, which has the actual command, the
    actual output, and the budget.

What this guard deliberately does NOT contain:
  * no hostname, no mirror name, no IP, no port (those are environment facts,
    they expire, and they differ per deployment)
  * no memory key literal (the agent decides what to look up, if anything)
  * no fixed second threshold (that is a baked-in assumption about someone
    else's machine)
The generality gate in tests/test_guard_net_awareness.py enforces all three
mechanically rather than by promise.
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# ── Is this shell command acquiring something over the network? ──────────────
# Deliberately broad in KIND and narrow in FORM: it matches package/artifact
# acquisition verbs across ecosystems, never a hostname. Kept as a tuple of
# substrings so a reader can audit it by eye and a test can parametrize it.
_NET_ACQUIRE_TOKENS = (
    # version control
    "git clone", "git fetch", "git pull", "git submodule update", "gh repo clone",
    # python package managers (bare, module form, uv/poetry/pdm/pipx/conda family)
    "pip install", "pip3 install", "pip download", "pip3 download",
    "python -m pip install", "python3 -m pip install",
    "python -m pip download", "python3 -m pip download",
    "uv pip install", "uv pip download", "uv sync", "uv add", "uv run",
    "uv tool install", "uv venv",
    "poetry install", "poetry add", "poetry update",
    "pipx install", "pdm install", "pdm add",
    "conda install", "conda create", "conda env create",
    "mamba install", "micromamba install",
    # os package managers
    "apt-get install", "apt install", "apt-get update", "apt update",
    "apt-get build-dep", "apt-get source",
    "yum install", "dnf install", "apk add", "brew install", "brew update",
    "zypper install", "pacman -S",
    # other language ecosystems
    "npm install", "npm ci", "npm i ", "npx ",
    "yarn install", "yarn add", "pnpm install", "pnpm add", "pnpm i ",
    "cargo build", "cargo install", "cargo fetch",
    "go mod download", "go get", "go install",
    "gem install", "bundle install", "composer install",
    "mvn dependency", "gradle build",
    # model / dataset / container / artifact pulls
    "huggingface-cli download", "hf download", "modelscope download",
    "docker pull", "docker build", "podman pull",
    "oras pull", "skopeo copy",
    # generic downloaders — download FORMS only; a bare page fetch is not
    # acquisition and should not be nagged about. Kept as separate flags rather
    # than a "--output" substring so their ORDER does not matter (measured
    # miss: 'curl -L -o f.tgz URL' has -o second, and a 'curl -o' token
    # therefore never matched it).
    "wget http", "wget ftp", "wget -c", "wget --",
    "aria2c ", "axel ",
)

#: Flags that turn a generic transfer tool into an ACQUISITION (writing a file)
#: rather than a page fetch. Matched as tokens, in any order.
_DOWNLOAD_FLAGS = ("-o", "-O", "--output", "--output-document", "--remote-name",
                   "--remote-name-all")


def _is_download_form(seg_low: str) -> bool:
    """True when a curl/wget-like segment writes a file (i.e. acquires)."""
    toks = seg_low.replace("=", " ").split()
    if not toks:
        return False
    head = toks[0].rsplit("/", 1)[-1]          # tolerate /usr/bin/curl
    if head not in ("curl", "wget", "http", "https", "aria2c", "axel"):
        return False
    return any(t in _DOWNLOAD_FLAGS for t in toks[1:])


def _segments(cmd: str) -> list[str]:
    """Split on operators that start a new command.

    Coarse but safe: we only use it to find where a new command begins, and
    every real separator does start one. This is what lets the guard see
    'cd repo && pip install x' rather than only line-leading verbs.
    """
    parts = re.split(r"&&|\|\||[;\n|]", cmd)
    return [p.strip() for p in parts if p and p.strip()]


def _strip_env_and_prefixes(seg: str) -> str:
    """Drop leading VAR=value assignments and common wrappers.

    A very common real form is 'DEBIAN_FRONTEND=noninteractive apt-get install
    -y x' or 'sudo -E apt-get install x'; without this the verb is never the
    first token and the guard silently never fires.
    """
    changed = True
    while changed:
        changed = False
        seg = seg.strip()
        # leading VAR=value
        m = re.match(r"^[A-Za-z_][A-Za-z0-9_]*=\S*\s+", seg)
        if m:
            seg = seg[m.end():]
            changed = True
            continue
        for prefix in ("sudo ", "env ", "time ", "nohup ", "command ",
                       "exec ", "stdbuf -oL -eL ", "DEBIAN_FRONTEND=noninteractive "):
            if seg.startswith(prefix):
                seg = seg[len(prefix):]
                changed = True
                break
    return seg


def is_network_acquire(cmd: str) -> bool:
    """True if any command segment acquires something over the network.

    Segment-aware and assignment-aware so compound install commands are caught
    (measured blind spot in the older probe: 'cd x && python -m pip install y'
    and 'VAR=1 apt-get install -y z' were both missed by a line-leading match).
    """
    if not cmd:
        return False
    for seg in _segments(cmd):
        stripped = _strip_env_and_prefixes(seg)
        low = stripped.lower()
        # A command whose FIRST token is just an echo/printf/cat-heredoc is
        # MENTIONING a command, not running it. Measured false positive:
        # `echo 'pip install x'` fired. Require the verb to be at the head of
        # the segment (after wrappers/assignments) rather than anywhere in it.
        head = low.split()[0].rsplit("/", 1)[-1] if low.split() else ""
        if head in ("echo", "printf", "cat", "tee") and not _is_download_form(low):
            continue
        if _is_download_form(low):
            return True
        if any(tok in low for tok in _NET_ACQUIRE_TOKENS):
            return True
    return False


_MESSAGE = (
    "[NetAwareness] That command acquires something over the network. Three "
    "things can happen and they need different responses:\n"
    "  1. It FAILS with a transport error (proxy 5xx / 'CONNECT tunnel failed' / "
    "DNS failure / connect timeout) — that is NOT proof the network is down. A "
    "proxy refusing a target and a target being unreachable are different "
    "worlds: try the other direction before you go changing sources "
    "('--noproxy *' / 'curl -x \"\"' / 'Acquire::http::Proxy \"false\"').\n"
    "  2. It SUCCEEDS but is SLOW. Slow is a cost, not an error — and it is "
    "worth knowing the cost before you pay it again. Time it: "
    "curl -m 5 -o /dev/null -w '%{http_code}:%{time_total}' <url>. Compare "
    "yourself against yourself (this attempt vs the last one, this path vs "
    "that one) — do not compare against a number that came from someone "
    "else's machine.\n"
    "  3. It is path-dependent. The same host can serve one path fine and hang "
    "on another (measured: one endpoint under a host timed out 3/3 while the "
    "host's root returned in 1.3s). So 'is this site up?' is the wrong "
    "question — measure the URL you actually need.\n"
    "\n"
    "Before you spend budget on discovery: check what you already know. If you "
    "have a note or memory entry recording this machine's egress behaviour, use "
    "it and stop. If you do not, measure — but BOUND it (~15s total, -m 5 per "
    "attempt, both directions), and if measuring gives you no answer, go on and "
    "do the real work. A failed attempt is one data point, never a verdict.\n"
    "Judge success by the artifact actually arriving (install completed, file "
    "fetched), never by an HTTP status code alone."
)


class NetAwarenessGuard(Guard):
    """Inject-only advisory after shell commands that acquire over the network.

    Fires on EVERY matching command (no per-turn latch): each acquisition is an
    independent chance to pay a slow path or misread a proxy error, and the
    reminder is cheap relative to what it prevents. This mirrors the shape of
    the existing IP/port advisory.

    check_pre is a no-op on purpose: this guard never blocks. The agent decides
    what to do with the reminder; nothing here makes that decision for it.
    """

    name = "net_awareness"
    priority = 14  # alongside the other advisory post-checks; order barely matters

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if ctx.tool_name != "shell":
            return None
        cmd = str(ctx.tool_args.get("command", ""))
        if not is_network_acquire(cmd):
            return None
        return GuardVerdict.inject(
            message=_MESSAGE,
            reason="network_acquire",
            category="net_awareness",
        )

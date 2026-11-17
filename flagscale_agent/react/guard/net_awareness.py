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


def _split_unquoted(cmd: str) -> list[str]:
    """Split a command line on separators that are NOT inside quotes.

    An operator inside a quoted string is data, not a separator: measured
    false positive, `grep -oE "(curl|wget)" log` — the `|` inside the search
    pattern started a phantom segment whose head read as a transfer tool.
    Written as a scanner rather than a split pattern because the property is
    "is this byte inside a quoted span", which a scanner states directly.
    """
    parts: list[str] = []
    buf: list[str] = []
    quote = ""
    i = 0
    n = len(cmd)
    while i < n:
        ch = cmd[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < n:
                buf.append(cmd[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "&" and i + 1 < n and cmd[i + 1] == "&":
            parts.append("".join(buf)); buf = []; i += 2; continue
        if ch == "|" and i + 1 < n and cmd[i + 1] == "|":
            parts.append("".join(buf)); buf = []; i += 2; continue
        if ch in (";", "\n", "|", "&"):
            parts.append("".join(buf)); buf = []; i += 1; continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p and p.strip()]


def _segments(cmd: str) -> list[str]:
    """Split on operators that start a new command, ignoring quoted operators.

    Coarse but safe: we only use it to find where a new command begins, and
    every real separator does start one. This is what lets the guard see
    'cd repo && pip install x' rather than only line-leading verbs.
    """
    return _split_unquoted(cmd)


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
        for prefix in ("sudo ", "time ", "nohup ", "command ",
                       "exec ", "stdbuf -oL -eL ", "DEBIAN_FRONTEND=noninteractive "):
            if seg.startswith(prefix):
                seg = seg[len(prefix):]
                changed = True
                break
        # `timeout 30 curl ...` — the coreutils wrapper takes a DURATION before
        # the real command. Measured blind spot: every curl in a real trial was
        # wrapped in `timeout N`, so the head was always 'timeout' and the
        # download-form check never even looked at `curl`. Strip the numeric
        # operand too (handles `timeout 30`, `timeout -s KILL 60`, `timeout 1.5h`).
        m = re.match(r"^timeout\s+(?:-s\s*\S+\s+|-k\s*\S+\s+)*[\d.]+[smhd]?\s+", seg)
        if m:
            seg = seg[m.end():]
            changed = True
            continue
        # `env -u HTTPS_PROXY -u HTTP_PROXY curl ...` — env takes FLAGS before
        # the real command. Measured miss: the agent's own no-proxy probe used
        # exactly this form, so the head stayed 'env'. Strip env's flags
        # (-u NAME, -i, -0, --unset=NAME, VAR=val pairs).
        m = re.match(r"^env\s+(?:(?:-[ui0]|--unset=\S*)\s+\S*\s*|[A-Za-z_]\w*=\S*\s+)*", seg)
        if m and m.end() > 4:
            rest = seg[m.end():].strip()
            if rest:
                seg = rest
                changed = True
                continue
    return seg


def _is_transfer_tool(seg_low: str) -> bool:
    """True when the segment's command IS a network transfer tool.

    Broader than _is_download_form on purpose: a bare `curl -sI <url>` probe
    still pays the network (it can be slow, it can be refused by a path-sensitive
    proxy) and the agent should hear that once. What stays excluded is a command
    that merely MENTIONS the tool inside a string/echo (handled by the head check
    in is_network_acquire).
    """
    toks = seg_low.replace("=", " ").split()
    if not toks:
        return False
    head = toks[0].rsplit("/", 1)[-1]
    # NB: bare `git` is deliberately NOT here — `git status`/`git diff` are
    # local. The network forms (clone/fetch/pull/submodule) are covered by
    # _NET_ACQUIRE_TOKENS instead.
    return head in ("curl", "wget", "http", "https", "aria2c", "axel",
                    "ftp", "sftp", "scp", "rsync", "ssh")


#: Commands whose ARGUMENTS are data to be searched, not commands to be run.
#: A pattern naming a transfer tool says what the user is LOOKING FOR, not
#: what the shell will DO. Kept as a name list (head-token match) rather than
#: anything pattern-based, for the same reason the refusal detector scans
#: whole tokens: the property being tested is identity of the command word.
_LOCAL_READERS = frozenset({
    "grep", "egrep", "fgrep", "rg", "ag", "ack", "ack-grep",
    "sed", "awk", "gawk", "cut", "tr", "sort", "uniq", "head", "tail",
    "wc", "find", "ls", "file", "md5sum", "sha256sum", "diff", "less", "more",
})


def _strip_quoted(text: str) -> str:
    """Remove quoted spans, keeping their quotes as a stand-in.

    A quoted span is DATA the command will read, never a command itself. This
    is the layer that makes 'grep -oE "(curl|wget)" log' safe even when the
    pattern is split awkwardly: whatever survives segmentation, the words
    inside quotes never reach the acquire-command test.
    """
    out: list[str] = []
    quote = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == quote:
                out.append(quote)
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _mentions_acquire_command(seg_low: str) -> bool:
    """True when the segment's TOKENS contain an acquisition command phrase.

    Whole-token matching, not substring: 'pip install' must appear as the
    words ``pip`` then ``install``, so a search pattern that merely quotes
    the phrase, a filename like ``my-pip-install-notes``, or a longer word
    that happens to contain it cannot be mistaken for the command itself.

    A token that is an absolute/relative path is compared by its BASENAME, so
    '/usr/bin/pip install y' is the same command as 'pip install y'.
    """
    toks = []
    for t in _strip_quoted(seg_low).replace("=", " ").split():
        toks.append(t.rsplit("/", 1)[-1] if "/" in t else t)
    for phrase in _NET_ACQUIRE_TOKENS:
        pt = phrase.split()
        n = len(pt)
        if n == 0 or n > len(toks):
            continue
        for i in range(len(toks) - n + 1):
            if toks[i:i + n] == pt:
                return True
    return False


def is_network_acquire(cmd: str) -> bool:
    """True if any command segment acquires something over the network.

    Segment-aware and assignment-aware so compound install commands are caught
    (measured blind spot in the older probe: 'cd x && python -m pip install y'
    and 'VAR=1 apt-get install -y z' were both missed by a line-leading match).

    "Acquires" is read broadly: package installs, clones, artifact pulls AND a
    plain transfer-tool invocation (curl/wget/scp/rsync/ssh). Measured reality:
    in a real trial the agent's FIRST network action was `curl -sI <host>` — a
    probe, not a download — and a download-form-only predicate stayed silent
    across every such command. The reminder is about the COST of a network step
    (slow, refused, path-dependent), which a probe pays too.
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
        # A filter/reader whose ARGUMENTS include a pattern does not run the
        # pattern. Measured false positive: `grep -n "pip install" README.md`
        # and `grep -oE "(curl|wget|pip download)" log` both fired — the
        # search TERM named a transfer, the command read a local file. Same
        # family as the echo/printf case above: the words are quoted data.
        if head in _LOCAL_READERS and not _is_download_form(low):
            continue
        if _is_download_form(low) or _is_transfer_tool(low):
            return True
        if _mentions_acquire_command(low):
            return True
    return False


#: Status codes that mean "this ADDRESS was refused", never "the content does
#: not exist". Kept as three groups because they are three different worlds and
#: the agent must not collapse them: a bot-block, a missing slug, and dropped
#: egress have nothing in common except that they are not a 200.
_REFUSAL_CODES = frozenset({"401", "403", "404", "410", "429", "000"})


def _word_tokens(text: str) -> list[str]:
    """Split on anything that is not a word character.

    Written as a scanner rather than a pattern on purpose: the only thing this
    module needs from the text is WHOLE tokens, and a whole-token test is the
    property that keeps 'wrote 4040 bytes' and '12.403s' from reading as
    statuses. Expressing it as a scan makes that property visible instead of
    hiding it inside a boundary assertion.
    """
    tokens: list[str] = []
    start = -1
    for i, ch in enumerate(text):
        if ch.isalnum() or ch == "_":
            if start < 0:
                start = i
        elif start >= 0:
            tokens.append(text[start:i])
            start = -1
    if start >= 0:
        tokens.append(text[start:])
    return tokens


def _looks_refused(tool_result: str) -> bool:
    """True when a transfer attempt's output carries a refusal/lookup verdict.

    Deliberately reuses the codes the DIAGNOSE block already teaches the agent
    to read (403 bot-block, 404 missing path, 000 no connection, 410/429) — the
    guard must not invent a second vocabulary for the same three worlds.

    A code embedded in a larger token is not a status: 'wrote 4040 bytes' and
    '12.403s' are a successful transfer and a duration, and treating either as
    a refusal would fire this tier on a transfer that worked.
    """
    if not tool_result:
        return False
    return any(tok in _REFUSAL_CODES for tok in _word_tokens(tool_result))


_RETRY_MESSAGE = (
    "[NetAwareness] That attempt came back REFUSED or UNRESOLVED — the output "
    "carries a status, not an answer. Read what it distinguishes before you "
    "change anything:\n"
    "  A REFUSED ROUTE IS NOT A MISSING FACT. Statuses that get collapsed into "
    "'unavailable' when they are three different worlds:\n"
    "    · 403 / 429 / a bot check — this ADDRESS refused THIS CLIENT. The "
    "content almost certainly exists; you are being turned away at the door. "
    "A browser-shaped request may not help and is not the only exit.\n"
    "    · 404 / 410 — this ADDRESS has no such path. The slug is wrong, not "
    "the fact. A different path, an API form, or a raw/plain serialisation of "
    "the same document is a different question.\n"
    "    · 000 / conn==0 — nothing connected. That is egress or a silent "
    "proxy, and it says nothing whatsoever about the target's contents.\n"
    "  A REFUSED ROUTE IS NOT A MISSING FACT: none of these three is evidence about a "
    "REQUIREMENT. They are statements about a ROUTE. Letting one refused route "
    "become the premise of your whole answer is how a reachable fact gets "
    "reported as 'I could not check the standard'.\n"
    "  SO SWITCH AXES BEFORE CONCLUDING — at minimum these, and name which you "
    "tried:\n"
    "    · ADDRESS — mirror / regional variant / different slug / raw or plain "
    "form / API or feed instead of the rendered page.\n"
    "    · PROVENANCE — search for the FACT, not for the one site you were "
    "pointed at. The same fact is usually ALSO published by a party that is not "
    "the original publisher: an open-access full-text index, a preprint or "
    "archive, a standards-body mirror, a package that vendors the data, a "
    "reference implementation that encodes the rule in CODE, the tool's own "
    "bundled documentation. Search in the wording a PRACTITIONER of the field "
    "would use — not the wording the task used, which may be a paraphrase, and "
    "not a term that has a different dominant meaning in that field.\n"
    "    · LOCAL — an installed package's source, a mounted cache, the image, "
    "or the material you were given may already hold it.\n"
    "  Bounded: a few genuinely DIFFERENT axes, not the same address retried. "
    "If every axis is exhausted, THAT is a real finding — then record it as "
    "UNCONSULTED, carry on under the most conservative reading of what it "
    "would have told you, and say which parts stayed unverified. An unreadable "
    "source is a gap to fill conservatively; it is never a reason to drop what "
    "it was going to tell you, and never a licence to call the requirement "
    "satisfied.\n"
)

_MESSAGE = (
    "[NetAwareness] That command acquires something over the network. A network "
    "failure is not one thing — it is at least three, and they are told apart by "
    "NUMBER, not by guesswork.\n"
    "  DIAGNOSE (one command, then READ the three fields):\n"
    "    curl -s -m 10 -o /dev/null -w 'dns=%{time_namelookup} "
    "conn=%{time_connect} http=%{http_code} total=%{time_total}\\n' <url>\n"
    "    dns>0 & conn==0.000000  → DNS resolved but TCP never connected. This is "
    "EGRESS being dropped or a proxy silently discarding this target — NOT a DNS "
    "problem, and NOT 'the site is down'.\n"
    "    http=000 & total≈your -m value → nothing came back before your own "
    "deadline. That is a verdict about YOUR deadline first: raise -m once before "
    "calling it unreachable (a 10s cap turns a slow-but-working path into a fake "
    "'blocked').\n"
    "    http=200 but body stalls → headers arrived, transfer cost is the "
    "problem. Slow is a cost, not an error: compare your own attempt against your "
    "own previous attempt on THIS path; never against a number from another "
    "machine.\n"
    "  THEN run the ONE discriminating test — same URL, both directions, read both "
    "results:\n"
    "    curl -s --noproxy '*' -m 10 -o /dev/null -w 'direct %{http_code}\\n' <url>\n"
    "    curl -s               -m 10 -o /dev/null -w 'proxy  %{http_code}\\n' <url>\n"
    "    direct fails + proxy works → the proxy is fine; use it.\n"
    "    proxy fails  + direct works → bypass it ('--noproxy *', curl -x \"\", "
    "Acquire::http::Proxy \"false\").\n"
    "    BOTH fail → do not conclude 'network down'. Test the HOST ROOT (scheme + "
    "host, no path). If the root answers, this is PATH-level (one path allowed, "
    "another refused) — change the URL, not the machine.\n"
    "  Before spending budget on discovery: use what you already know (a note or "
    "memory entry about this machine's egress ends the question). Otherwise "
    "measure, bounded — and if the numbers give no answer, go do the real work. "
    "One failed attempt is one data point, never a verdict. Judge success by the "
    "artifact actually arriving (install completed, file fetched), never by an "
    "HTTP status alone.\n"
    "  IF IT STILL DID NOT ARRIVE — the thing you were acquiring is still "
    "required. A failed acquisition removes the ROUTE, not the REQUIREMENT. "
    "Before you report it as unavailable, SWITCH AT LEAST THREE AXES — one "
    "failed address is one data point about one address:\n"
    "    · ADDRESS — the same content usually lives at more than one address: a "
    "mirror or regional variant of the host, a different path or slug on the "
    "same host, an archived or raw form instead of the rendered page, a "
    "different serialisation (plain text, XML, JSON, feed, API) of the same "
    "document.\n"
    "    · PROVENANCE — the same fact is usually ALSO published by a party that "
    "is not the original publisher: an open-access full-text index, a preprint "
    "or archive, a standards body's mirror, a package that vendors the data, a "
    "reference implementation that encodes the rule in code, the tool's own "
    "bundled docs. Search for the FACT, not for the one site you were pointed "
    "at — and search in the wording a practitioner would use, not in the "
    "wording the task used.\n"
    "    · CLIENT — the client itself may be absent or wrong (a transfer tool "
    "that is not installed fails with 'not found' and says nothing about the "
    "network) — use whatever client this environment does have, and install "
    "one from a route that works.\n"
    "    · LOCAL — whatever you could not fetch may already be present: in the "
    "image, in a mounted cache, in an installed package's source, in the "
    "material you were given. Look before you conclude it cannot be had.\n"
    "  A REFUSED ROUTE IS NOT A MISSING FACT: a site that refuses a bot, or one path that 404s, "
    "is not evidence about a REQUIREMENT. Those are statements about a route. "
    "Do not let one refused route become the premise of your whole answer.\n"
    "  Bound this: a few genuinely DIFFERENT routes, not the same one repeated. "
    "If every route fails, that is a real finding — record it as UNCONSULTED, "
    "carry on under the most conservative reading of what it would have told "
    "you, and say which parts stayed unverified. An unreadable source is a gap "
    "to fill conservatively; it is never a reason to drop what it was going to "
    "tell you, and never a licence to call the requirement satisfied.\n"
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
        # TWO TIERS, because the two worlds need different reminders. A first
        # attempt earns the diagnostic ladder (how to READ the number). An
        # attempt whose own output already carries a refusal/lookup verdict
        # earns the ROUTE-ENUMERATION half — the ladder is behind it, and what
        # it needs to hear is "a refused address is not a missing fact; switch
        # axes before you call this unavailable".
        #
        # Measured motivation: a trial spent all of its lookups on one refused
        # host and then treated the requirement itself as unreadable, while the
        # same fact was reachable elsewhere from the same machine. The first
        # tier's advice had already been given and had already been read; what
        # was missing was the second tier.
        if _looks_refused(str(ctx.tool_result or "")):
            return GuardVerdict.inject(
                message=_RETRY_MESSAGE,
                reason="network_refused",
                category="net_awareness",
            )
        return GuardVerdict.inject(
            message=_MESSAGE,
            reason="network_acquire",
            category="net_awareness",
        )

"""Network-acquire detection must read COMMANDS, not the words inside them.

Measured on a live trial: a purely local `grep -oE "(curl|wget|pip download|
web_fetch)" log` fired the network advisory, because the search PATTERN named
transfer tools and the segment splitter cut on the `|` inside the quotes. The
agent was told to diagnose a network it had never touched.

The distinction these tests pin: an argument is DATA the command reads, a head
token is the COMMAND that runs. Both directions matter — a filter that stops
firing on real installs would be a worse bug than the one being fixed.
"""

from flagscale_agent.react.guard.net_awareness import is_network_acquire


# ── 1. Quoted/pattern arguments are not commands ─────────────────────────────
def test_local_search_for_transfer_words_is_not_a_network_command() -> None:
    for cmd in (
        'grep -oE "(curl|wget|pip download|web_fetch).{0,70}" /tmp/log',
        'grep -n "pip install" README.md',
        'rg "git clone" docs/',
        'grep -c "403" log',
        "tail -c 500 log | grep curl",
        'sed -n "/curl/p" notes.txt',
        'awk "/pip install/" log',
    ):
        assert is_network_acquire(cmd) is False, cmd


def test_echo_still_silent() -> None:
    assert is_network_acquire('echo "pip install x"') is False
    assert is_network_acquire("printf 'git clone y\\n'") is False


# ── 2. Real acquisition still fires (the positive bound) ─────────────────────
def test_real_acquisition_still_fires() -> None:
    for cmd in (
        "pip install requests",
        "pip download primer3-py -d /tmp/p3",
        "/usr/bin/pip install y",
        "git clone https://example.invalid/x",
        "curl -sI https://example.invalid/simple/",
        "wget https://example.invalid/x.tar.gz",
        "cd /tmp && python -m pip install foo",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y z",
        "cat payload | curl -d @- https://example.invalid/x",
    ):
        assert is_network_acquire(cmd) is True, cmd


# ── 3. The splitter is quote-aware ───────────────────────────────────────────
def test_pipe_inside_quotes_does_not_create_a_segment() -> None:
    """A phantom segment is what made the pattern read as a tool invocation."""
    from flagscale_agent.react.guard.net_awareness import _segments

    assert _segments('grep -oE "(curl|wget)" log') == ['grep -oE "(curl|wget)" log']
    # A real pipe still splits.
    assert len(_segments("cat log | grep curl")) == 2


def test_acquire_phrase_must_be_whole_tokens() -> None:
    """A word that CONTAINS the phrase is not the phrase."""
    from flagscale_agent.react.guard.net_awareness import _mentions_acquire_command

    assert _mentions_acquire_command("notes about pip install") is True
    assert _mentions_acquire_command("my-pip-install-notes.txt") is False
    assert _mentions_acquire_command("pipper installer") is False
    # Path-prefixed verb is the same command.
    assert _mentions_acquire_command("/usr/local/bin/pip install x") is True

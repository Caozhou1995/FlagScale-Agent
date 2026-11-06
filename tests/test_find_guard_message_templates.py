"""§3.5 contract: every command form the guard MESSAGE recommends must pass the
guard that recommends it.

The plan's §3.5 requires the guard message to carry a paste-ready legal form, and
requires that template to be verified locally — a message that hands back a form
the guard itself blocks is worse than no template, because the agent pastes it,
gets blocked again, and learns the message cannot be trusted.

This test is the executable half of that requirement: it extracts each command
line from the message's COPY-PASTE block and asserts FindGuard passes it. It also
carries a mutation-proof negative control (the blocked counterparts ARE blocked),
so a broken matcher or a wrong extraction cannot make it green for free.
"""

from __future__ import annotations

import re

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.find_guard import (
    FindGuard,
    _FIND_MESSAGE,
    _GREP_MESSAGE,
)


def _shell(cmd: str) -> GuardContext:
    return GuardContext(tool_name="shell", tool_args={"command": cmd})


def _copy_paste_commands(message: str) -> list[str]:
    """Extract the command column from the message's COPY-PASTE block.

    The block is introduced by "COPY-PASTE FORMS" and each entry is indented two
    spaces followed by the command and a trailing "# comment". Returns the command
    part only. Fails loudly if the block is missing or empty, so that DELETING the
    template cannot silently satisfy this test.
    """
    if "COPY-PASTE FORMS" not in message:
        pytest.fail("guard message carries no COPY-PASTE FORMS block (§3.5)")
    tail = message.split("COPY-PASTE FORMS", 1)[1]
    # The marker line itself continues into prose ("... that pass this guard ...").
    # Drop everything up to the first newline, otherwise that prose fragment is
    # harvested as if it were a command — it passes the guard for the wrong reason
    # (it is not a find/grep invocation at all) and inflates the count, masking a
    # genuinely missing template. Found by feeding the extractor a synthetic block.
    tail = tail.split("\n", 1)[1] if "\n" in tail else ""
    # Stop at the next prose paragraph that follows the templates.
    for stop in ("Blocked counterparts", "A literal path is what buys"):
        tail = tail.split(stop, 1)[0]
    cmds: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("("):
            continue
        # Drop the trailing "# ..." annotation, which is documentation not shell.
        cmd = re.split(r"\s+#\s", stripped, maxsplit=1)[0].strip()
        if cmd:
            cmds.append(cmd)
    if not cmds:
        pytest.fail("COPY-PASTE block present but no command lines extracted")
    return cmds


FIND_TEMPLATES = _copy_paste_commands(_FIND_MESSAGE)
GREP_TEMPLATES = _copy_paste_commands(_GREP_MESSAGE)


@pytest.mark.parametrize("cmd", FIND_TEMPLATES)
def test_find_message_template_passes_the_guard(cmd):
    """Every recommended find form must pass the guard that recommends it."""
    verdict = FindGuard().check_pre(_shell(cmd))
    assert verdict is None, (
        f"the find_guard message recommends a form the guard itself blocks: "
        f"{cmd!r} -> {verdict.action}/{verdict.reason}"
    )


@pytest.mark.parametrize("cmd", GREP_TEMPLATES)
def test_grep_message_template_passes_the_guard(cmd):
    verdict = FindGuard().check_pre(_shell(cmd))
    assert verdict is None, (
        f"the find_guard message recommends a form the guard itself blocks: "
        f"{cmd!r} -> {verdict.action}/{verdict.reason}"
    )


def test_templates_are_not_vacuous():
    """Negative control: the extraction really yields the forms we intend.

    Without this, a broken extractor returning [] or a single trivial entry would
    let the parametrized tests above pass while covering nothing.
    """
    assert len(FIND_TEMPLATES) >= 5, FIND_TEMPLATES
    assert len(GREP_TEMPLATES) >= 3, GREP_TEMPLATES
    joined = " ".join(FIND_TEMPLATES)
    assert "./src" in joined            # literal path, not a placeholder
    assert "-maxdepth" in joined        # a real scope filter
    assert "head" in joined             # a cap against unbounded output


@pytest.mark.parametrize(
    "cmd",
    ["find / -name '*.py'", "find . -name '*.py'", "grep -rn TODO /"],
)
def test_blocked_counterparts_are_still_blocked(cmd):
    """The message tells the agent these are the WRONG forms; if the guard stopped
    blocking them the message would be actively misleading. This is also the
    red-phase control for the template tests: the same matcher that must return
    None above must return a block here, so a matcher broken into always-PASS
    cannot make the suite green."""
    verdict = FindGuard().check_pre(_shell(cmd))
    assert verdict is not None and verdict.action == "block"


def test_cd_then_dot_is_still_blocked_despite_maxdepth():
    """The message makes a specific claim: `cd ./src && find . -name '*.py'` still
    blocks because the guard reads the ROOT TOKEN. Pin that claim to the code —
    otherwise the message could drift from the matcher and nobody would notice."""
    assert FindGuard().check_pre(_shell("cd ./src && find . -name '*.py'")) is not None
    assert FindGuard().check_pre(_shell("cd ./src && find . -maxdepth 3 -name '*.py'")) is not None
    # ... while the literal-path form the message recommends passes:
    assert FindGuard().check_pre(_shell("find ./src -maxdepth 3 -name '*.py'")) is None

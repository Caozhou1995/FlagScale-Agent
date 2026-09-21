"""Regression: ShellTool must NOT let children inherit the agent's tty as stdin.

Bug (fa_ds_tmp hang): shell children were launched without ``stdin=``, so a
child that reads stdin (``head -1``, ``read``, ``cat`` with no file, ...)
inherited the agent's controlling tty as fd0. On a tty the input queue is
shared: the child blocked in ``n_tty_read`` drained every keystroke before the
agent's prompt_toolkit epoll reader could see it, so the live prompt went
permanently deaf after such a command ran (a "long turn" that left an orphan
``head -1`` behind). The fix passes ``stdin=subprocess.DEVNULL`` to Popen.

The load-bearing assertion is the spy test: it pins the exact kwarg at the
Popen call site. The behavioral tests document the observable contract.
"""

import os
import pty
import subprocess
import sys
from unittest import mock

from flagscale_agent.react.tools.shell import ShellTool


def test_popen_receives_devnull_stdin():
    """The exact fix: ShellTool must pass stdin=subprocess.DEVNULL to Popen.

    This is the regression guard — it fails on the pre-fix code regardless of
    what stdin the test process itself happens to have.
    """
    sh = ShellTool()
    real_popen = subprocess.Popen
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_popen(*args, **kwargs)

    with mock.patch("subprocess.Popen", side_effect=spy):
        sh.execute(command="echo hi")

    assert "stdin" in seen, "ShellTool Popen missing stdin= kwarg (tty leak)"
    assert seen["stdin"] == subprocess.DEVNULL, (
        f"child stdin not DEVNULL: {seen.get('stdin')!r}"
    )


def test_child_fd0_is_null_under_a_real_tty():
    """Under a real pty stdin, ShellTool's child fd0 must NOT be that pty.

    This is the behavioral guard that actually reproduces the hang condition:
    the test harness gives the child a controlling pty as stdin (like the live
    agent), so a pre-fix Popen would leak the pty into the grandchild and the
    assertion below would see a ``/dev/pts/`` path.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "from flagscale_agent.react.tools.shell import ShellTool;"
        "print('FD0=' + ShellTool()"
        ".execute(command='readlink /proc/self/fd/0').strip())"
    )
    master, slave = pty.openpty()
    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": repo_root},
        )
    finally:
        os.close(master)

    assert "FD0=/dev/null" in p.stdout, (
        f"child fd0 leaked to a tty: stdout={p.stdout!r} stderr={p.stderr!r}"
    )
    assert "/dev/pts/" not in p.stdout


def test_child_stdin_is_the_null_device():
    """End-to-end: the launched shell's own fd0 resolves to the null device."""
    sh = ShellTool()
    out = sh.execute(command="readlink /proc/self/fd/0 || echo NOFD")
    assert "/dev/pts/" not in out, f"child inherited a pts tty: {out!r}"


def test_stdin_reading_command_does_not_block():
    """A command that reads stdin returns immediately (EOF from /dev/null)."""
    sh = ShellTool()
    out = sh.execute(command="head -1; echo DONE")
    assert "DONE" in out


def test_pipeline_stdin_still_works():
    """DEVNULL on the pipeline head must not break the pipe consumer's stdin."""
    sh = ShellTool()
    out = sh.execute(command="printf 'a\\nb\\nc\\n' | head -1")
    assert "a" in out
    assert "b" not in out

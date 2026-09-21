# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Watchdog for a wedged interactive prompt.

Long interactive sessions occasionally end up sitting at the prompt (``>``)
while the input loop no longer consumes keystrokes: the process is alive and
the pane still renders, but typing does nothing (``tmux send-keys`` is
ignored). From the outside this looks like a frozen terminal; from the inside
the ``prompt_toolkit`` event loop is wedged on its input fd (observed as a
main thread parked in ``ep_poll`` with bytes available on the tty that were
never read).

This watchdog runs on a daemon thread and watches for exactly that condition:
input is waiting but unconsumed while a prompt is supposed to be showing, and
that state persists for ``threshold`` seconds. "Waiting" is detected two ways,
OR-ed: bytes still readable on stdin (the kernel-buffer wedge), and a
non-empty userspace backlog via an injected ``pending_probe`` (the second wedge
class, where prompt_toolkit's reader thread already consumed the keystroke — so
``select`` on the fd is blind to it — yet the loop stalled before dispatching
it). On detection it first nudges the terminal with ``SIGWINCH`` (a cheap,
harmless re-arm); if the input is *still* unconsumed after ``winch_grace`` more
seconds it escalates to ``SIGINT``, which breaks the wedged ``prompt()`` so the
REPL can rebuild a fresh :class:`PromptSession` and accept input again.

Properties that make this safe:
  * It never fires during a turn (guarded by ``is_at_prompt``), so a long
    model call or tool run cannot trip it.
  * It never steals input: it only *queries* readability with ``select``;
    ``prompt_toolkit`` performs the actual read. When the loop is healthy it
    consumes each keystroke immediately, so the pending window resets and the
    watchdog stays silent.
  * All effects are injectable, so the decision logic is unit-testable without
    a real tty (see ``tests/test_prompt_watchdog.py``).
"""

import os
import select
import signal
import threading
import time


class PromptWatchdog:
    """Detects an interactive prompt whose input loop has stopped consuming
    keystrokes, and escalates (SIGWINCH -> SIGINT) to unstick it."""

    def __init__(
        self,
        is_at_prompt,
        on_sigint=None,
        fd=None,
        *,
        pending_probe=None,
        tty_reader_probe=None,
        on_tty_reader=None,
        interval=2.0,
        threshold=60.0,
        winch_grace=15.0,
        select_fn=select.select,
        monotonic=time.monotonic,
        kill_fn=os.kill,
        getpid=os.getpid,
        logger=None,
    ):
        self._is_at_prompt = is_at_prompt
        self._on_sigint = on_sigint
        self._pending_probe = pending_probe
        self._tty_reader_probe = tty_reader_probe
        self._on_tty_reader = on_tty_reader
        self._fd = fd if fd is not None else self._default_fd()
        self._interval = interval
        self._threshold = threshold
        self._winch_grace = winch_grace
        self._select = select_fn
        self._monotonic = monotonic
        self._kill = kill_fn
        self._getpid = getpid
        self._logger = logger or (lambda msg: None)
        self._pending_since = None
        self._nudged_at = None
        self._fired_winch = False
        self._fired_sigint = False
        self._reader_since = None
        self._fired_reader = False
        self._thread = None

    @staticmethod
    def _default_fd():
        try:
            import sys

            return sys.stdin.fileno()
        except Exception:
            return None

    def _input_pending(self) -> bool:
        """True iff input is known to be waiting but not consumed.

        Two independent signals, OR-ed:

        * **Kernel bytes** — ``select([fd])`` reports the tty readable. Covers
          the classic wedge where bytes sit in the kernel buffer and the loop
          never reads them.
        * **Userspace backlog** — ``pending_probe()`` reports keys that were
          *already read off the fd* but not yet processed (prompt_toolkit's
          ``KeyProcessor.input_queue`` / ``Vt100Input._buffer``). Covers the
          second wedge class where the reader thread consumed the keystroke, so
          the fd is no longer readable, yet the application stalled before
          dispatching it — the exact case ``select`` alone is blind to.
        """
        if self._fd is not None:
            try:
                r, _, _ = self._select([self._fd], [], [], 0)
                if r:
                    return True
            except Exception:
                pass
        if self._pending_probe is not None:
            try:
                if self._pending_probe():
                    return True
            except Exception:
                pass
        return False

    def _foreign_reader_present(self) -> bool:
        """True iff a foreign process is draining our terminal's input.

        Third wedge class: another process in this terminal's foreground
        process group holds the tty as stdin and reads it, so every keystroke
        is consumed *before* prompt_toolkit's selector ever sees fd0 ready.
        ``select(fd)`` reports nothing (the bytes are gone) and the userspace
        backlog is empty (we never got them), so both signals above are blind.
        A shell child launched without ``stdin=DEVNULL`` — e.g. a stranded
        ``head -1`` — is the classic culprit.

        The injected ``tty_reader_probe`` decides this (see
        :func:`foreign_tty_readers`); any exception is swallowed so a probe
        failure can never trip the watchdog.
        """
        if self._tty_reader_probe is None:
            return False
        try:
            return bool(self._tty_reader_probe())
        except Exception:
            return False

    def tick(self):
        """Run one evaluation.

        Returns the action taken: ``None``, ``"winch"``, ``"sigint"`` or
        ``"tty_reader"``. Kept side-effect-injectable so tests can drive it
        deterministically.
        """
        # Only act while we believe a prompt is showing.
        if not self._is_at_prompt():
            self._reset()
            return None

        now = self._monotonic()

        # Stage 0 — root-cause recovery. If a foreign process is draining our
        # tty, no amount of SIGWINCH/SIGINT to *this* process helps: the thief
        # keeps eating the keystrokes. Detect that reader and, once it has
        # persisted past the threshold, reap it. This is checked FIRST because
        # it removes the cause rather than the symptom, and because while a
        # thief is active the two input signals below are structurally blind.
        if self._foreign_reader_present():
            if self._reader_since is None:
                self._reader_since = now
                return None
            if not self._fired_reader and (now - self._reader_since) >= self._threshold:
                self._fired_reader = True
                if self._on_tty_reader is not None:
                    try:
                        self._on_tty_reader()
                    except Exception:
                        pass
                self._logger(
                    "prompt watchdog: foreign process has held the tty as stdin "
                    f"for {now - self._reader_since:.0f}s — reaping it to recover "
                    "prompt input"
                )
                return "tty_reader"
            return None
        # No foreign reader: clear its persistence timer and fall through to the
        # (symptom-level) input-pending checks.
        self._reader_since = None
        self._fired_reader = False

        if not self._input_pending():
            # Healthy: nothing waiting, or the loop already consumed it.
            self._reset()
            return None

        if self._pending_since is None:
            self._pending_since = now
            return None

        # Stage 1 — gentle re-arm. Some prompt_toolkit wedges (e.g. after a
        # terminal resize desync) clear on a fresh SIGWINCH.
        if not self._fired_winch and (now - self._pending_since) >= self._threshold:
            self._fired_winch = True
            self._nudged_at = now
            self._kill(self._getpid(), signal.SIGWINCH)
            self._logger(
                "prompt watchdog: stdin readable but unconsumed for "
                f"{now - self._pending_since:.0f}s — sent SIGWINCH to re-arm"
            )
            return "winch"

        # Stage 2 — escalate. The nudge did not help, so break the wedged
        # prompt(); the REPL catches KeyboardInterrupt and rebuilds.
        if (
            self._fired_winch
            and not self._fired_sigint
            and self._nudged_at is not None
            and (now - self._nudged_at) >= self._winch_grace
        ):
            self._fired_sigint = True
            if self._on_sigint is not None:
                try:
                    self._on_sigint()
                except Exception:
                    pass
            self._kill(self._getpid(), signal.SIGINT)
            self._logger(
                "prompt watchdog: input still unconsumed — sent SIGINT to break "
                "the wedged prompt and rebuild the session"
            )
            return "sigint"

        return None

    def _reset(self):
        self._pending_since = None
        self._nudged_at = None
        self._fired_winch = False
        self._fired_sigint = False
        self._reader_since = None
        self._fired_reader = False

    def _loop(self):
        while True:
            time.sleep(self._interval)
            try:
                self.tick()
            except Exception:
                pass

    def start(self):
        """Start the background watcher (idempotent)."""
        if (
            self._fd is None
            and self._pending_probe is None
            and self._tty_reader_probe is None
        ) or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="prompt-watchdog"
        )
        self._thread.start()


def _read_fd_link(pid, fd=0):
    """Return the target of ``/proc/<pid>/fd/<fd>`` or ``None``."""
    try:
        return os.readlink(f"/proc/{pid}/fd/{fd}")
    except Exception:
        return None


def _our_tty_path(fd=0):
    """Return the device path of our own stdin if it is a tty, else ``None``."""
    try:
        return os.ttyname(fd)
    except Exception:
        return None


def foreign_tty_readers(tty_path=None, *, getpid=os.getpid, our_fd=0):
    """Return PIDs of foreign processes draining **our** terminal's stdin.

    A "foreign tty reader" is a process OTHER than ourselves whose stdin
    (fd 0) is *the very same terminal device we read from*. Such a process,
    left running while we sit at the prompt, competes with us for every
    keystroke and can drain input before ``prompt_toolkit`` sees it — the
    third wedge class this watchdog defends against.

    Scoping is essential and deliberately narrow. ``tty_path`` defaults to
    the device behind our own stdin (``our_fd``); a process is reported only
    when its stdin points at exactly that device. We NEVER scan the host for
    "any process reading a tty": on a shared machine that would flag every
    other user's shell and turn recovery into a host-wide process massacre.
    If our own stdin is not a tty, there is nothing to protect — return [].

    ``tty_path`` may be passed explicitly for tests; passing an empty string
    is treated the same as ``None`` (derive our own device).
    """
    me = getpid()
    if tty_path is None or tty_path == "":
        tty_path = _our_tty_path(our_fd)
        if tty_path is None:
            return []
    found = []
    try:
        entries = os.listdir("/proc")
    except Exception:
        return found
    for name in entries:
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == me:
            continue
        target = _read_fd_link(pid, 0)
        if target != tty_path:
            continue
        # Skip our own ancestors (shell, tmux, init) — they legitimately share
        # the tty but are not competing thieves.
        if _is_ancestor(pid, me):
            continue
        found.append(pid)
    return found


def _is_ancestor(pid, me):
    """True iff ``pid`` is an ancestor of ``me`` (walk /proc PPid chain)."""
    cur = me
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        if cur == pid:
            return True
        try:
            with open(f"/proc/{cur}/status") as fh:
                ppid = 0
                for line in fh:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
        except Exception:
            return False
        cur = ppid
    return False


def reap_tty_readers(tty_path=None, *, kill_fn=os.kill, getpid=os.getpid, logger=None):
    """SIGKILL every foreign process draining our terminal's stdin.

    Returns the list of PIDs that were signalled. Best-effort: each kill is
    wrapped so one failure cannot abort the sweep.
    """
    victims = foreign_tty_readers(tty_path, getpid=getpid)
    killed = []
    for pid in victims:
        try:
            kill_fn(pid, signal.SIGKILL)
            killed.append(pid)
        except Exception:
            pass
    if killed and logger is not None:
        try:
            logger(f"prompt watchdog: reaped tty-reading processes {killed}")
        except Exception:
            pass
    return killed

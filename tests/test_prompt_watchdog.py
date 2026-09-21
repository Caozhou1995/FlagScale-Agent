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

"""Tests for the interactive-prompt watchdog.

Scenario: a long interactive session's prompt_toolkit loop wedges — the
process is alive and rendering, but stdin has pending bytes it never reads, so
keystrokes are silently dropped (looks like a frozen terminal).  PromptWatchdog
detects "at prompt + stdin readable but unconsumed" and escalates
SIGWINCH -> SIGINT to force the loop to break and rebuild.

All effects are injected so the decision logic is verified deterministically.
"""

import signal

import pytest

from flagscale_agent.react.prompt_watchdog import PromptWatchdog


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class Recorder:
    def __init__(self, pending=True):
        self.pending = pending
        self.kills = []

    def select(self, r, w, x, timeout):
        return ([r[0]], [], []) if self.pending else ([], [], [])

    def kill(self, pid, sig):
        self.kills.append(sig)


def make_watchdog(at_prompt=True, pending=True, clock=None, interval_effects=None):
    clock = clock or FakeClock()
    rec = Recorder(pending=pending)
    wd = PromptWatchdog(
        is_at_prompt=lambda: at_prompt,
        fd=0,
        threshold=60.0,
        winch_grace=15.0,
        select_fn=rec.select,
        monotonic=clock,
        kill_fn=rec.kill,
        getpid=lambda: 999,
        logger=lambda m: None,
    )
    return wd, rec, clock


class TestPromptWatchdog:
    def test_no_action_when_not_at_prompt(self):
        wd, rec, clock = make_watchdog(at_prompt=False)
        clock.advance(120)
        assert wd.tick() is None
        assert rec.kills == []

    def test_no_action_when_input_not_pending(self):
        # Healthy loop: prompt_toolkit consumed every keystroke, so nothing
        # is pending -> watchdog must stay silent forever.
        wd, rec, clock = make_watchdog(pending=False)
        for _ in range(100):
            clock.advance(10)
            assert wd.tick() is None
        assert rec.kills == []

    def test_no_action_before_threshold(self):
        wd, rec, clock = make_watchdog()
        clock.advance(30)  # < 60s threshold
        assert wd.tick() is None
        assert rec.kills == []

    def test_winch_after_threshold(self):
        wd, rec, clock = make_watchdog()
        clock.advance(1)
        wd.tick()  # arm pending_since
        clock.advance(60)
        assert wd.tick() == "winch"
        assert rec.kills == [signal.SIGWINCH]

    def test_winch_fires_only_once(self):
        wd, rec, clock = make_watchdog()
        clock.advance(1)
        wd.tick()
        clock.advance(60)
        wd.tick()
        clock.advance(5)
        assert wd.tick() is None
        assert rec.kills == [signal.SIGWINCH]

    def test_sigint_after_winch_grace(self):
        wd, rec, clock = make_watchdog()
        clock.advance(1)
        wd.tick()
        clock.advance(60)
        wd.tick()  # SIGWINCH
        clock.advance(15)  # winch_grace
        assert wd.tick() == "sigint"
        assert rec.kills == [signal.SIGWINCH, signal.SIGINT]

    def test_consumed_input_resets_and_never_fires(self):
        # The bug-avoidance case: input becomes consumed (loop healthy again)
        # before the threshold -> pending window resets, no signal.
        wd, rec, clock = make_watchdog()
        clock.advance(1)
        wd.tick()
        clock.advance(30)
        rec.pending = False  # loop consumed it
        assert wd.tick() is None
        rec.pending = True
        clock.advance(30)  # was reset, only 30s since re-arm
        assert wd.tick() is None
        assert rec.kills == []

    def test_on_sigint_callback_invoked(self):
        fired = {"n": 0}

        def cb():
            fired["n"] += 1

        clock = FakeClock()
        rec = Recorder(pending=True)
        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            on_sigint=cb,
            fd=0,
            threshold=60.0,
            winch_grace=15.0,
            select_fn=rec.select,
            monotonic=clock,
            kill_fn=rec.kill,
            getpid=lambda: 1,
        )
        clock.advance(1)
        wd.tick()
        clock.advance(60)
        wd.tick()
        clock.advance(15)
        wd.tick()
        assert fired["n"] == 1

    def test_missing_fd_never_fires(self):
        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=None,
            select_fn=lambda *a: ([], [], []),
            monotonic=FakeClock(),
            kill_fn=lambda *a: None,
            getpid=lambda: 1,
        )
        wd._fd = None
        assert wd.tick() is None

    def test_backlog_probe_fires_when_fd_clean(self):
        # Second wedge class: the reader consumed the keystroke (fd no longer
        # readable -> select sees nothing) but the key never got dispatched, so
        # the userspace backlog is non-empty. The watchdog must still fire.
        dead = lambda *a: ([], [], [])  # fd permanently clean
        backlog = {"n": 1}
        kills = []
        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=0,
            pending_probe=lambda: backlog["n"] > 0,
            threshold=60.0,
            winch_grace=15.0,
            select_fn=dead,
            monotonic=(clk := FakeClock()),
            kill_fn=lambda pid, sig: kills.append(sig),
            getpid=lambda: 1,
        )
        clk.advance(1)
        wd.tick()  # arm
        clk.advance(60)
        assert wd.tick() == "winch"
        clk.advance(15)
        assert wd.tick() == "sigint"
        assert kills == [signal.SIGWINCH, signal.SIGINT]

    def test_backlog_probe_cleared_resets(self):
        # If the backlog drains (loop recovered), the window resets — no fire.
        dead = lambda *a: ([], [], [])
        backlog = {"n": 1}
        kills = []
        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=0,
            pending_probe=lambda: backlog["n"] > 0,
            threshold=60.0,
            winch_grace=15.0,
            select_fn=dead,
            monotonic=(clk := FakeClock()),
            kill_fn=lambda pid, sig: kills.append(sig),
            getpid=lambda: 1,
        )
        clk.advance(1)
        wd.tick()
        clk.advance(30)
        backlog["n"] = 0  # consumed -> healthy again
        assert wd.tick() is None
        clk.advance(60)
        assert wd.tick() is None
        assert kills == []

    def test_backlog_probe_exception_is_ignored(self):
        # A probe that raises must not crash the watchdog; it degrades to
        # fd-only detection.
        def boom():
            raise RuntimeError("unexpected")

        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=0,
            pending_probe=boom,
            select_fn=(lambda *a: ([], [], [])),
            monotonic=FakeClock(),
            kill_fn=lambda *a: None,
            getpid=lambda: 1,
        )
        assert wd.tick() is None

    def test_start_runs_with_only_probe(self):
        # fd is None but a probe exists -> the watcher must still start.
        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=None,
            pending_probe=lambda: False,
            interval=100.0,
            kill_fn=lambda *a: None,
            getpid=lambda: 1,
        )
        wd._fd = None
        wd.start()
        assert wd._thread is not None and wd._thread.daemon


class TestForeignTtyReader:
    """Third wedge class: a foreign process drains our terminal's input.

    Detection must fire even though ``select`` sees nothing and the userspace
    backlog is empty (the thief consumed the keys before us). Recovery must
    remove the thief, not SIGINT ourselves.
    """

    @staticmethod
    def _wd(reader, clock, fd_dead=True, backlog=None, **kw):
        fired = {"n": 0, "kills": []}
        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=0,
            pending_probe=backlog,
            tty_reader_probe=(lambda: reader["present"]),
            on_tty_reader=lambda: fired.__setitem__("n", fired["n"] + 1),
            threshold=60.0,
            winch_grace=15.0,
            select_fn=(lambda *a: ([], [], []) if fd_dead else ([a[0][0]], [], [])),
            monotonic=clock,
            kill_fn=lambda pid, sig: fired["kills"].append(sig),
            getpid=lambda: 1,
        )
        return wd, fired

    def test_reader_fires_and_recovers_after_threshold(self):
        # Foreign reader + fd clean + backlog empty: the exact blind spot.
        clock = FakeClock()
        reader = {"present": True}
        wd, fired = self._wd(reader, clock)
        clock.advance(1)
        assert wd.tick() is None  # arm reader_since
        clock.advance(60)
        assert wd.tick() == "tty_reader"
        assert fired["n"] == 1
        assert fired["kills"] == []  # we do NOT signal ourselves

    def test_reader_fires_only_once(self):
        clock = FakeClock()
        reader = {"present": True}
        wd, fired = self._wd(reader, clock)
        clock.advance(1)
        wd.tick()
        clock.advance(60)
        wd.tick()
        clock.advance(60)
        assert wd.tick() is None
        assert fired["n"] == 1

    def test_no_reader_never_fires(self):
        clock = FakeClock()
        reader = {"present": False}
        wd, fired = self._wd(reader, clock)
        for _ in range(100):
            clock.advance(10)
            assert wd.tick() is None
        assert fired["n"] == 0

    def test_reader_gone_before_threshold_resets(self):
        clock = FakeClock()
        reader = {"present": True}
        wd, fired = self._wd(reader, clock)
        clock.advance(1)
        wd.tick()
        clock.advance(30)
        reader["present"] = False  # thief died / never existed
        assert wd.tick() is None
        reader["present"] = True
        clock.advance(30)  # reset -> only 30s, below threshold
        assert wd.tick() is None
        assert fired["n"] == 0

    def test_reader_takes_precedence_over_input_signals(self):
        # Reader present AND input pending: root-cause recovery wins; we must
        # not send SIGWINCH/SIGINT to ourselves while a thief is active.
        clock = FakeClock()
        reader = {"present": True}
        wd, fired = self._wd(reader, clock, fd_dead=False)
        clock.advance(1)
        wd.tick()
        clock.advance(120)  # well past winch + grace
        assert wd.tick() == "tty_reader"
        assert fired["kills"] == []
        assert fired["n"] == 1

    def test_reader_probe_exception_is_ignored(self):
        def boom():
            raise RuntimeError("unexpected")

        wd = PromptWatchdog(
            is_at_prompt=lambda: True,
            fd=0,
            tty_reader_probe=boom,
            select_fn=(lambda *a: ([], [], [])),
            monotonic=FakeClock(),
            kill_fn=lambda *a: None,
            getpid=lambda: 1,
        )
        assert wd.tick() is None

    def test_not_at_prompt_resets_reader_window(self):
        clock = FakeClock()
        reader = {"present": True}
        at = {"v": True}
        fired = {"n": 0}
        wd = PromptWatchdog(
            is_at_prompt=lambda: at["v"],
            fd=0,
            tty_reader_probe=lambda: reader["present"],
            on_tty_reader=lambda: fired.__setitem__("n", fired["n"] + 1),
            threshold=60.0,
            select_fn=lambda *a: ([], [], []),
            monotonic=clock,
            kill_fn=lambda *a: None,
            getpid=lambda: 1,
        )
        clock.advance(1)
        at["v"] = False
        wd.tick()  # not at prompt -> reset
        at["v"] = True
        clock.advance(30)
        assert wd.tick() is None  # window was reset
        assert fired["n"] == 0


class TestForeignTtyReadersModule:
    """The /proc scanner that feeds the probe and the reaper."""

    def test_scans_only_tty_stdin_and_excludes_ancestors_and_self(self, monkeypatch):
        import flagscale_agent.react.prompt_watchdog as pw

        victims = {
            "100": "/dev/pts/3",      # foreign tty reader (the thief)
            "200": "/dev/null",       # healthy background job -> skip
            "300": "/dev/pts/3",      # ancestor (our shell) -> skip
            "999": "/dev/pts/3",      # ourselves -> skip
        }
        monkeypatch.setattr(pw.os, "listdir", lambda p: list(victims))

        def fake_readlink(path):
            pid = path.split("/")[2]
            return victims.get(pid, "")

        monkeypatch.setattr(pw.os, "readlink", fake_readlink)
        # 300 is an ancestor of 999.
        monkeypatch.setattr(pw, "_is_ancestor", lambda pid, me: pid == 300)
        monkeypatch.setattr(pw, "_blocked_in_tty_read", lambda pid: True)

        got = pw.foreign_tty_readers("/dev/pts/3", getpid=lambda: 999)
        assert got == [100]

    def test_only_same_device_matches_not_any_tty(self, monkeypatch):
        # LOAD-BEARING safety: a process on a DIFFERENT tty (another user's
        # shell on a shared host) must never be reported for our device.
        import flagscale_agent.react.prompt_watchdog as pw

        victims = {"100": "/dev/pts/3", "101": "/dev/pts/99"}
        monkeypatch.setattr(pw.os, "listdir", lambda p: list(victims))
        monkeypatch.setattr(
            pw.os, "readlink", lambda path: victims.get(path.split("/")[2], "")
        )
        monkeypatch.setattr(pw, "_is_ancestor", lambda pid, me: False)
        monkeypatch.setattr(pw, "_blocked_in_tty_read", lambda pid: True)
        assert pw.foreign_tty_readers("/dev/pts/3", getpid=lambda: 999) == [100]

    def test_default_derives_our_own_tty(self, monkeypatch):
        import flagscale_agent.react.prompt_watchdog as pw

        monkeypatch.setattr(pw, "_our_tty_path", lambda fd=0: "/dev/pts/5")
        victims = {"100": "/dev/pts/5", "101": "/dev/pts/6"}
        monkeypatch.setattr(pw.os, "listdir", lambda p: list(victims))
        monkeypatch.setattr(
            pw.os, "readlink", lambda path: victims.get(path.split("/")[2], "")
        )
        monkeypatch.setattr(pw, "_is_ancestor", lambda pid, me: False)
        monkeypatch.setattr(pw, "_blocked_in_tty_read", lambda pid: True)
        # No tty_path: derives /dev/pts/5 and matches only pid 100.
        assert pw.foreign_tty_readers(getpid=lambda: 999) == [100]

    def test_no_tty_returns_empty(self, monkeypatch):
        # If our own stdin is not a tty, there is nothing to protect.
        import flagscale_agent.react.prompt_watchdog as pw

        monkeypatch.setattr(pw, "_our_tty_path", lambda fd=0: None)
        assert pw.foreign_tty_readers(getpid=lambda: 999) == []

    def test_tty_path_filter(self, monkeypatch):
        import flagscale_agent.react.prompt_watchdog as pw

        victims = {"100": "/dev/pts/3", "101": "/dev/pts/7"}
        monkeypatch.setattr(pw.os, "listdir", lambda p: list(victims))
        monkeypatch.setattr(
            pw.os, "readlink", lambda path: victims.get(path.split("/")[2], "")
        )
        monkeypatch.setattr(pw, "_is_ancestor", lambda pid, me: False)
        monkeypatch.setattr(pw, "_blocked_in_tty_read", lambda pid: True)
        assert pw.foreign_tty_readers("/dev/pts/7", getpid=lambda: 999) == [101]

    def test_reap_kills_every_reader(self, monkeypatch):
        import flagscale_agent.react.prompt_watchdog as pw

        monkeypatch.setattr(pw, "foreign_tty_readers", lambda *a, **k: [10, 11])
        killed = []

        def kill_fn(pid, sig):
            killed.append((pid, sig))

        out = pw.reap_tty_readers(kill_fn=kill_fn, getpid=lambda: 1)
        assert out == [10, 11]
        assert killed == [(10, signal.SIGKILL), (11, signal.SIGKILL)]

    def test_reap_tolerates_kill_failure(self, monkeypatch):
        import flagscale_agent.react.prompt_watchdog as pw

        monkeypatch.setattr(pw, "foreign_tty_readers", lambda *a, **k: [10, 11])
        killed = []

        def kill_fn(pid, sig):
            if pid == 10:
                raise ProcessLookupError()
            killed.append(pid)

        out = pw.reap_tty_readers(kill_fn=kill_fn, getpid=lambda: 1)
        assert out == [11]  # only the successful kill is reported
        assert killed == [11]

    def test_listdir_failure_returns_empty(self, monkeypatch):
        import flagscale_agent.react.prompt_watchdog as pw

        def boom(p):
            raise OSError("no /proc")

        monkeypatch.setattr(pw.os, "listdir", boom)
        assert pw.foreign_tty_readers(getpid=lambda: 1) == []

    def test_shares_tty_but_not_reading_is_not_a_reader(self, monkeypatch):
        # LOAD-BEARING false-positive guard: a process whose stdin is the SAME
        # tty device but which is NOT blocked in the tty read path (tmux/ssh/
        # sibling shell/sleep) must never be reported. Matching the device
        # alone reaped innocent processes on an idle terminal.
        import flagscale_agent.react.prompt_watchdog as pw

        victims = {"100": "/dev/pts/3", "101": "/dev/pts/3"}
        monkeypatch.setattr(pw.os, "listdir", lambda p: list(victims))
        monkeypatch.setattr(
            pw.os, "readlink", lambda path: victims.get(path.split("/")[2], "")
        )
        monkeypatch.setattr(pw, "_is_ancestor", lambda pid, me: False)
        # 100 really reads the tty; 101 merely shares the device (not reading).
        monkeypatch.setattr(pw, "_blocked_in_tty_read", lambda pid: pid == 100)
        assert pw.foreign_tty_readers("/dev/pts/3", getpid=lambda: 999) == [100]

    def test_unreadable_stack_keeps_candidate(self, monkeypatch):
        # Conservative fallback: when the kernel stack is unavailable (no
        # CAP_SYS_ADMIN on a hardened host) we cannot prove the process is NOT
        # reading, so keep it. Detection must not silently disappear.
        import flagscale_agent.react.prompt_watchdog as pw

        victims = {"100": "/dev/pts/3"}
        monkeypatch.setattr(pw.os, "listdir", lambda p: list(victims))
        monkeypatch.setattr(
            pw.os, "readlink", lambda path: victims.get(path.split("/")[2], "")
        )
        monkeypatch.setattr(pw, "_is_ancestor", lambda pid, me: False)
        monkeypatch.setattr(pw, "_blocked_in_tty_read", lambda pid: None)
        assert pw.foreign_tty_readers("/dev/pts/3", getpid=lambda: 999) == [100]

    def test_blocked_in_tty_read_classifies_stack(self, monkeypatch):
        import flagscale_agent.react.prompt_watchdog as pw

        class _Fh:
            def __init__(self, body): self._body = body
            def read(self): return self._body
            def __enter__(self): return self
            def __exit__(self, *a): return False

        real_open = open

        def fake_open(path, *a, **k):
            if path == "/proc/4242/stack":
                return _Fh("[<0>] wait_woken\n[<0>] n_tty_read+0x5d3\n")
            if path == "/proc/4243/stack":
                return _Fh("[<0>] hrtimer_nanosleep+0x99\n")
            if path == "/proc/4244/stack":
                return _Fh("")
            raise FileNotFoundError(path)

        monkeypatch.setattr(pw, "open", fake_open, raising=False)
        import builtins
        monkeypatch.setattr(builtins, "open", fake_open)
        assert pw._blocked_in_tty_read(4242) is True
        assert pw._blocked_in_tty_read(4243) is False
        assert pw._blocked_in_tty_read(4244) is None
        assert pw._blocked_in_tty_read(9999) is None

    def test_fallback_syscall_when_stack_unreadable(self, monkeypatch):
        # Non-root host: /proc/<pid>/stack is blank. Must NOT lose the fix —
        # fall back to /proc/<pid>/syscall (same-uid readable).
        import flagscale_agent.react.prompt_watchdog as pw
        import builtins

        bodies = {
            "/proc/10/syscall": "0 0x0 0x7ff 0x2000 0x1 0x2 0x3 0x4 0x5",  # read(fd0)
            "/proc/11/syscall": "230 0x0 0x0 0x7ff 0x1 0x2 0x3 0x4 0x5",   # nanosleep
            "/proc/12/syscall": "0 0x3 0x7ff 0x2000 0x1 0x2 0x3 0x4 0x5",   # read(other fd)
        }

        class _Fh:
            def __init__(self, b): self._b = b
            def read(self): return self._b
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_open(path, *a, **k):
            if path.startswith("/proc/") and path.endswith("/stack"):
                return _Fh("")  # blank stack -> permission-denied simulation
            if path in bodies:
                return _Fh(bodies[path])
            raise FileNotFoundError(path)

        monkeypatch.setattr(builtins, "open", fake_open)
        monkeypatch.setattr(pw.platform, "machine", lambda: "x86_64")
        assert pw._blocked_in_tty_read(10) is True   # read on fd0 -> reading
        assert pw._blocked_in_tty_read(11) is False  # nanosleep -> not reading
        assert pw._blocked_in_tty_read(12) is False  # read on fd3 -> not fd0

    def test_fallback_unknown_on_exotic_arch(self, monkeypatch):
        # Syscall numbers are arch-specific; on an unknown arch stay conservative.
        import flagscale_agent.react.prompt_watchdog as pw
        import builtins

        class _Fh:
            def __init__(self, b): self._b = b
            def read(self): return self._b
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_open(path, *a, **k):
            if path.endswith("/stack"):
                return _Fh("")
            raise FileNotFoundError(path)

        monkeypatch.setattr(builtins, "open", fake_open)
        monkeypatch.setattr(pw.platform, "machine", lambda: "riscv64")
        assert pw._blocked_in_tty_read(10) is None


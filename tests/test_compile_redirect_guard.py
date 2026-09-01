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

"""Tests for CompileRedirectGuard — pre-launch block on verbose build commands
that do not persist output to a file."""

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.compile_redirect import CompileRedirectGuard


def _ctx(command):
    return GuardContext(tool_name="shell", tool_args={"command": command})


class TestCompileRedirectGuard:

    def setup_method(self):
        self.g = CompileRedirectGuard()

    # --- should BLOCK: compile command whose output is NOT visible to the monitor ---
    @pytest.mark.parametrize("cmd", [
        "make -j$(nproc)",
        "make -j8",
        "cmake --build build",
        "ninja",
        "cargo build --release",
        "opam install coq",
        "gcc -O2 main.c -o main",
        "g++ -std=c++17 a.cpp",
        "go build ./...",
        "make -j | tail",          # piped to pager = the classic hang
        "make 2>&1 | head -100",
        # bare file redirects are now a BLACK BOX to the monitor -> block
        "make -j$(nproc) > build.log 2>&1",
        "make -j8 >build.log",
        "cargo build &> out.txt",
        "opam install coq >> install.log 2>&1",
        "gcc -O2 main.c -o main > compile.log 2>&1",
        # a multi-package install with env setup and bare redirect:
        'unset HTTP_PROXY && eval $(opam env) && opam install pkg1 pkg2 -y > /tmp/opam_install2.log 2>&1; echo "EXIT: $?"',
    ])
    def test_blocks_compile_not_visible_to_monitor(self, cmd):
        v = self.g.check_pre(_ctx(cmd))
        assert v is not None and v.action == "block", f"should block: {cmd}"
        assert v.category == "compile_redirect"

    # --- should PASS: output stays visible to the monitor (tee / tail -f) ---
    @pytest.mark.parametrize("cmd", [
        "cmake --build build 2>&1 | tee build.log",
        "ninja | tee ninja.log",
        "make -j$(nproc) 2>&1 | tee build.log",
        # background redirect + tail -f streams the log back to the terminal
        "opam install coq > install.log 2>&1 & tail -f install.log",
        "make -j8 > build.log 2>&1 & tail -F build.log",
    ])
    def test_passes_compile_visible_to_monitor(self, cmd):
        v = self.g.check_pre(_ctx(cmd))
        assert v is None, f"should pass (tee / tail -f keeps output visible): {cmd}"

    # --- should PASS: not a compile command ---
    @pytest.mark.parametrize("cmd", [
        "ls -la",
        "cat build.log",
        "grep error build.log",
        "echo make",              # 'make' as an argument, not the driver
        "tail -f build.log",
        "python train.py",
        "rm -rf build",
    ])
    def test_ignores_non_compile(self, cmd):
        v = self.g.check_pre(_ctx(cmd))
        assert v is None, f"should ignore: {cmd}"

    def test_non_shell_tool_ignored(self):
        ctx = GuardContext(tool_name="read_file", tool_args={"path": "Makefile"})
        assert self.g.check_pre(ctx) is None

    def test_empty_command_ignored(self):
        assert self.g.check_pre(_ctx("")) is None

    def test_block_is_overridable(self):
        v = self.g.check_pre(_ctx("make -j"))
        assert v.action == "block"
        assert v.overridable is True
        # default accept_override: any reason > 5 chars
        assert self.g.accept_override("already writes its own log file", _ctx("make -j")) is True
        assert self.g.accept_override("", _ctx("make -j")) is False

    def test_stdbuf_wrapped_compile_still_blocks(self):
        # stdbuf -oL make ... still needs a file to be inspectable after
        v = self.g.check_pre(_ctx("stdbuf -oL -eL make -j4"))
        assert v is not None and v.action == "block"

    def test_stateless_across_calls(self):
        # unlike BackupGuard, this guard keys off the command text every time
        assert self.g.check_pre(_ctx("make -j")).action == "block"
        assert self.g.check_pre(_ctx("make -j 2>&1 | tee log")) is None
        assert self.g.check_pre(_ctx("make -j")).action == "block"

    def test_bare_redirect_is_blackbox_to_monitor(self):
        # regression: pure `> file` was previously treated as OK, but it hides
        # all output from the live monitor for the whole run (opam stall).
        assert self.g.check_pre(_ctx("make -j > build.log 2>&1")).action == "block"
        # tee keeps it visible
        assert self.g.check_pre(_ctx("make -j 2>&1 | tee build.log")) is None

    def test_message_guides_single_threaded_build(self):
        # The block message must steer toward: single-threaded (-j1) builds ONLY,
        # probe first under a timeout, then full single-threaded build. No
        # multi-threaded advice — the eval container has limited memory.
        v = self.g.check_pre(_ctx("make -j > build.log 2>&1"))
        assert v.action == "block"
        msg = v.message
        # single-threaded is the only option
        assert "single-threaded" in msg
        assert "-j1" in msg
        assert "probe" in msg.lower()
        assert "timeout" in msg
        # must NOT recommend multi-threaded parallelism as advice
        # (-j$(nproc) may appear in the "NEVER use" warning — that's fine)
        assert "-j N" not in msg
        assert "-j4" not in msg
        # must NOT tell agent to check nproc / free -h / available memory as advice
        assert "free -h" not in msg
        assert "available memory" not in msg.lower()
        # must NOT mention memory-to-cores mapping
        assert "4GB" not in msg
        assert "32GB" not in msg
        # must warn about OOM from multi-threading
        assert "OOM" in msg or "oom" in msg.lower()
        # must forbid bare unbounded make -j
        assert "make -j" in msg  # as the thing to NEVER use
        # verify exit code after build/install
        assert "EXIT" in msg or "exit code" in msg.lower() or "exit:" in msg.lower()
        assert "which" in msg  # verify binary exists

    def test_message_advises_one_by_one_install(self):
        """The block message must advise installing critical packages ONE BY ONE,
        not all at once — a multi-package install can freeze for 10+ minutes."""
        v = self.g.check_pre(_ctx("cargo build --release"))
        assert v.action == "block"
        msg = v.message
        # must mention one-by-one / individual install
        assert "ONE BY ONE" in msg or "one by one" in msg or "individually" in msg
        # must show per-package install examples
        assert "pip install" in msg
        # must advise checking exit code after each package
        assert "EXIT" in msg or "exit code" in msg.lower()
        # must mention that multi-package install can freeze/stall
        assert "freeze" in msg.lower() or "stall" in msg.lower() or "frozen" in msg.lower()
        # must apply to all package managers
        assert "apt" in msg or "pip" in msg or "cargo" in msg or "npm" in msg
        # must NOT contain task-specific package names or tools (no leaking)
        assert "coq" not in msg.lower()
        assert "menhir" not in msg.lower()
        assert "ocamlfind" not in msg.lower()
        assert "opam" not in msg.lower()

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

"""Tests for ResourceProbeGuard — structured resource profile injected on every
first command that sizes parallelism or launches a background job.

Covers: trigger matrix, fresh-profile re-injection on EVERY trigger (no latch),
container-wrapping hint, low-confidence step-ladder fallback, and the parser
granularity floor (cpu.max three states, v1 cfs fallback, cpuset, memory
max/high/swap, pids.max, effective=min clamp) — must not regress below
guard/compile_redirect.py.
"""

import json

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard import resource_probe as rp
from flagscale_agent.react.guard.resource_probe import (
    ResourceProbeGuard,
    _build_profile,
    _effective_cpu,
    _parse_cpu_list,
    _parse_mem_value,
    _read_cpu_max,
    _read_cpuset,
    _read_memory_high,
    _read_memory_limit,
    _read_pids_max,
)


def _ctx(command, tool="shell"):
    return GuardContext(tool_name=tool, tool_args={"command": command})


class TestTrigger:
    def setup_method(self):
        self.g = ResourceProbeGuard()

    @pytest.mark.parametrize("cmd", [
        "nproc",
        "python -c 'import os; print(os.cpu_count())'",
        "make -j8",
        "make -j $(nproc)",
        "find . -name '*.c' | xargs -P 16 gcc -c",
        "python train.py --num_workers 32",
        "python train.py workers=64",
    ])
    def test_sizing_commands_trigger(self, cmd):
        v = self.g.check_pre(_ctx(cmd))
        assert v is not None and v.action == "inject", f"should inject: {cmd}"
        assert v.category == "resource_probe"

    def test_background_launch_triggers(self):
        # REAL contract: background is a separate tool_args key (see
        # longtimeshell.py L190), not text inside the command string.
        ctx = GuardContext(tool_name="shell",
                           tool_args={"command": "run_long_training_job --epochs 10",
                                      "background": True})
        v = self.g.check_pre(ctx)
        assert v is not None and v.action == "inject"

    def test_bg_launch_regex_fallback_still_hits(self):
        # defensive: serialized variants where the flag leaked into the text
        v = self.g.check_pre(_ctx("run_long_job --epochs 10 background=true"))
        assert v is not None and v.action == "inject"

    def test_non_sizing_commands_pass(self):
        for cmd in ["ls -la", "cat /proc/cpuinfo", "grep -rn TODO .", "echo done"]:
            assert self.g.check_pre(_ctx(cmd)) is None, f"should pass: {cmd}"

    def test_non_shell_tool_never_triggers(self):
        assert self.g.check_pre(_ctx("nproc", tool="read_file")) is None

    # --- no latch: every trigger re-injects a FRESH profile ----------------
    def test_repeats_reinject_within_turn(self):
        v1 = self.g.check_pre(_ctx("nproc"))
        v2 = self.g.check_pre(_ctx("make -j8"))
        assert v1 is not None and v2 is not None

    def test_repeats_reinject_across_reset_turn(self):
        v1 = self.g.check_pre(_ctx("nproc"))
        self.g.reset_turn()
        v2 = self.g.check_pre(_ctx("nproc"))
        assert v1 is not None and v2 is not None

    def test_reset_turn_is_stateless_noop(self):
        # no per-turn state: reset must not raise and must not change behavior
        self.g.check_pre(_ctx("nproc"))
        self.g.reset_turn()
        assert self.g.check_pre(_ctx("nproc")) is not None


class TestMessage:
    def setup_method(self):
        self.g = ResourceProbeGuard()

    @staticmethod
    def _stub_profile(confidence):
        return {
            "exec_domain": {"kind": "container", "evidence": ["/.dockerenv"]},
            "cpu": {"effective": 8.0 if confidence == "high" else None,
                    "source": "stub", "nproc": 256, "nproc_source": "stub",
                    "note": "stub"},
            "memory_gib": {"limit": 16.0, "swap": None, "high": None,
                           "source": "stub"},
            "pids": {"max": 512, "source": "stub"},
            "confidence": confidence,
            "ladder_available": confidence == "low",
        }

    def test_low_confidence_carries_ladder_and_kill_floor(self, monkeypatch):
        monkeypatch.setattr(rp, "_build_profile",
                            lambda: self._stub_profile("low"))
        v = self.g.check_pre(_ctx("nproc"))
        assert v is not None
        msg = v.message
        assert "confidence" in msg and "exec_domain" in msg  # JSON profile present
        assert "step-ladder" in msg                          # low-conf fallback hint
        assert "SIGTERM before SIGKILL" in msg               # kill-discipline floor
        assert "effective" in msg and "observation" in msg   # constraint vs observation

    def test_high_confidence_omits_ladder_hint(self, monkeypatch):
        monkeypatch.setattr(rp, "_build_profile",
                            lambda: self._stub_profile("high"))
        g = ResourceProbeGuard()  # fresh instance
        v = g.check_pre(_ctx("nproc"))
        assert v is not None
        assert "step-ladder" not in v.message
        assert "host number" not in v.message


    def test_container_wrap_hint_without_execution(self):
        g = ResourceProbeGuard()
        v = g.check_pre(_ctx("docker run --rm img python -j 4 train.py"))
        assert v is not None
        assert "docker" in v.message and "Nothing was auto-executed" in v.message

    def test_container_wrap_regex_variants(self):
        assert rp._CONTAINER_WRAP_RE.search("nerdctl exec c1 nproc")
        assert rp._CONTAINER_WRAP_RE.search("podman run img make -j8")
        assert not rp._CONTAINER_WRAP_RE.search("docker ps")


class TestParsers:
    # --- cpu.max: three states ---------------------------------------------
    def test_cpu_max_v2_numeric(self, tmp_path):
        (tmp_path / "cpu.max").write_text("200000 100000\n")
        quota, src = _read_cpu_max(str(tmp_path))
        assert quota == 2.0 and "200000/100000" in src

    def test_cpu_max_v2_max(self, tmp_path):
        (tmp_path / "cpu.max").write_text("max 100000\n")
        quota, src = _read_cpu_max(str(tmp_path))
        assert quota is None and "unlimited" in src

    def test_cpu_max_v2_malformed(self, tmp_path):
        (tmp_path / "cpu.max").write_text("garbage\n")
        quota, src = _read_cpu_max(str(tmp_path))
        assert quota is None and "unreadable" in src

    def test_cpu_max_v1_cfs_fallback(self, tmp_path):
        d = tmp_path / "cpu"
        d.mkdir()
        (d / "cpu.cfs_quota_us").write_text("400000\n")
        (d / "cpu.cfs_period_us").write_text("100000\n")
        quota, src = _read_cpu_max(str(tmp_path))
        assert quota == 4.0 and "v1 cfs" in src

    # --- cpuset ------------------------------------------------------------
    def test_cpuset_restricts(self, tmp_path):
        (tmp_path / "cpuset.cpus.effective").write_text("0-7,16-23\n")
        cpus, src = _read_cpuset(str(tmp_path))
        assert len(cpus) == 16 and "0-7,16-23" in src

    def test_cpuset_v1_controller_dir(self, tmp_path):
        d = tmp_path / "cpuset"
        d.mkdir()
        (d / "cpuset.cpus.effective").write_text("0-3\n")
        cpus, _ = _read_cpuset(str(tmp_path))
        assert len(cpus) == 4

    def test_cpuset_v2_configured_fallback(self, tmp_path):
        # .effective absent -> configured .cpus still counts
        (tmp_path / "cpuset.cpus").write_text("2-5\n")
        cpus, _ = _read_cpuset(str(tmp_path))
        assert len(cpus) == 4

    def test_cpuset_absent_is_unlimited_not_zero(self, tmp_path):
        cpus, src = _read_cpuset(str(tmp_path))
        assert cpus is None and "unreadable" in src

    # --- effective = min(quota, cpuset) --------------------------------------
    def test_effective_takes_min(self, tmp_path):
        (tmp_path / "cpu.max").write_text("200000 100000\n")
        (tmp_path / "cpuset.cpus.effective").write_text("0-7\n")
        eff, src = _effective_cpu(str(tmp_path))
        assert eff == 2.0 and "effective=2" in src

    def test_effective_no_constraint(self, tmp_path, monkeypatch):
        # No constraint files + nproc available -> nproc becomes an explicit
        # UPPER-BOUND estimate, marked in the source string (never silently).
        import flagscale_agent.react.guard.resource_probe as rp
        monkeypatch.setattr(rp, "_read_nproc", lambda: (64, "nproc-src"))
        eff, src = _effective_cpu(str(tmp_path))
        assert eff == 64.0 and "estimated-from-nproc=64" in src
        assert "upper bound" in src

    def test_effective_no_constraint_no_nproc(self, tmp_path, monkeypatch):
        # Degraded double-miss (no files, no nproc) -> honest None.
        import flagscale_agent.react.guard.resource_probe as rp
        monkeypatch.setattr(rp, "_read_nproc", lambda: (None, "unreadable"))
        eff, src = _effective_cpu(str(tmp_path))
        assert eff is None and "no CPU constraint readable" in src

    # --- memory: max three states + high + swap ----------------------------
    def test_memory_limit_numeric(self, tmp_path):
        (tmp_path / "memory.max").write_text(str(16 * 1024 ** 3))
        lim, swap, src = _read_memory_limit(str(tmp_path))
        assert lim == 16.0 and swap is None and "memory.max" in src

    def test_memory_limit_max_is_unlimited(self, tmp_path):
        (tmp_path / "memory.max").write_text("max\n")
        lim, _, src = _read_memory_limit(str(tmp_path))
        assert lim is None and "unlimited" not in src  # reported, not guessed

    def test_memory_v1_limit_in_bytes(self, tmp_path):
        d = tmp_path / "memory"
        d.mkdir()
        (d / "memory.limit_in_bytes").write_text(str(8 * 1024 ** 3))
        lim, _, _ = _read_memory_limit(str(tmp_path))
        assert lim == 8.0

    def test_memory_high_v2(self, tmp_path):
        (tmp_path / "memory.high").write_text(str(4 * 1024 ** 3))
        high, src = _read_memory_high(str(tmp_path))
        assert high == 4.0 and "memory.high" in src

    def test_parse_mem_value_states(self):
        assert _parse_mem_value("max") is None
        assert _parse_mem_value("-1") is None
        assert _parse_mem_value("") is None
        assert _parse_mem_value("abc") is None
        assert _parse_mem_value(str(2 * 1024 ** 3)) == 2.0

    # --- pids ---------------------------------------------------------------
    def test_pids_max_numeric_and_unlimited(self, tmp_path):
        (tmp_path / "pids.max").write_text("512\n")
        val, _ = _read_pids_max(str(tmp_path))
        assert val == 512
        (tmp_path / "pids.max").write_text("max\n")
        val, src = _read_pids_max(str(tmp_path))
        assert val is None and "unlimited" in src

    # --- cpu list parser edge cases ----------------------------------------
    def test_parse_cpu_list(self):
        assert _parse_cpu_list("0-7,16-23") == list(range(8)) + list(range(16, 24))
        assert _parse_cpu_list("5") == [5]
        assert _parse_cpu_list("bad") is None
        assert _parse_cpu_list("8-3") is None
        assert _parse_cpu_list("") == []


class TestProfile:
    def test_profile_fields_and_confidence(self):
        p = _build_profile()
        for k in ("exec_domain", "cpu", "memory_gib", "pids", "confidence",
                  "ladder_available"):
            assert k in p
        assert p["confidence"] in ("high", "low")
        # observation vs constraint labels never regress (the H-axis core)
        assert "observation" in p["cpu"]["note"]
        assert "constraint" in p["cpu"]["note"]


class TestRegressionBatch:
    """Regression coverage for the parser/vantage/trigger fix batch."""

    # --- v1 unlimited sentinel -----------------------------------------------
    def test_parse_mem_value_v1_sentinel(self):
        # v1 kernels lay 2^63 / 2^64-1 for "unlimited" — not an astronomic GiB.
        assert _parse_mem_value(str(2 ** 63)) is None
        assert _parse_mem_value(str(2 ** 64 - 1)) is None
        assert _parse_mem_value(str(2 ** 60)) is None
        # A plausible real limit (32 GiB) must still parse.
        assert _parse_mem_value(str(32 * 1024 ** 3)) == 32.0

    def test_memory_limit_v1_sentinel(self, tmp_path):
        (tmp_path / "memory" ).mkdir()
        (tmp_path / "memory/memory.limit_in_bytes").write_text(str(2 ** 63))
        lim, _, src = _read_memory_limit(str(tmp_path))
        assert lim is None and "limit_in_bytes" in src

    # --- v1 cpuset effective filename ---------------------------------------
    def test_cpuset_v1_effective_filename(self, tmp_path):
        d = tmp_path / "cpuset"
        d.mkdir()
        (d / "cpuset.effective_cpus").write_text("0-3\n")
        cpus, src = _read_cpuset(str(tmp_path))
        assert cpus == [0, 1, 2, 3] and "0-3" in src

    # --- vantage: own leaf cgroup before mount root --------------------------
    def test_cgroup_root_prefers_own_leaf(self, tmp_path, monkeypatch):
        import flagscale_agent.react.guard.resource_probe as rp

        leaf = tmp_path / "system.slice" / "myjob.scope"
        leaf.mkdir(parents=True)
        (leaf / "cpu.max").write_text("200000 100000\n")
        (tmp_path / "cpu.max").write_text("max 100000\n")

        fake_self = tmp_path / "proc_self_cgroup"
        fake_self.write_text(f"0::/{leaf.relative_to(tmp_path)}\n")

        monkeypatch.setattr(rp, "_CGROUP_BASE", str(tmp_path))
        monkeypatch.setattr(rp, "_SELF_CGROUP", str(fake_self))
        assert rp._cgroup_root() == str(leaf)

    def test_cgroup_root_no_v2_line_uses_mount_root(self, tmp_path, monkeypatch):
        import flagscale_agent.react.guard.resource_probe as rp

        (tmp_path / "cpu" ).mkdir()
        (tmp_path / "cpu/cpu.cfs_quota_us").write_text("200000\n")
        (tmp_path / "cpu/cpu.cfs_period_us").write_text("100000\n")

        fake_self = tmp_path / "proc_self_cgroup"
        fake_self.write_text("5:cpu:/legacy\n")  # no v2 "0::" line

        monkeypatch.setattr(rp, "_CGROUP_BASE", str(tmp_path))
        monkeypatch.setattr(rp, "_SELF_CGROUP", str(fake_self))
        assert rp._cgroup_root() == str(tmp_path)

    # --- profile: estimate marker, suggestion, conflicts ---------------------
    def test_profile_estimate_and_conflicts(self, monkeypatch):
        import flagscale_agent.react.guard.resource_probe as rp

        monkeypatch.setattr(rp, "_cgroup_root", lambda: "/nonexistent")
        monkeypatch.setattr(rp, "_read_nproc", lambda: (64, "nproc"))
        monkeypatch.setattr(rp, "_detect_exec_domain",
                            lambda: ("host", ["ev"]))
        p = rp._build_profile()
        assert p["cpu"]["estimated_upper_bound"] is True
        assert p["cpu"]["suggested_max_cpu"] is None
        assert p["cpu"]["conflicts"] == []
        assert p["confidence"] == "low" and p["ladder_available"] is True

    def test_profile_measured_conflict_capped_by_nproc(self, monkeypatch):
        import flagscale_agent.react.guard.resource_probe as rp

        fake_root = "/tmp/probe-fixture-capped"
        import os, shutil
        shutil.rmtree(fake_root, ignore_errors=True)
        os.makedirs(fake_root)
        with open(os.path.join(fake_root, "cpu.max"), "w") as f:
            f.write("8000000 100000\n")  # 80 cores > nproc 64

        monkeypatch.setattr(rp, "_cgroup_root", lambda: fake_root)
        monkeypatch.setattr(rp, "_read_nproc", lambda: (64, "nproc"))
        monkeypatch.setattr(rp, "_detect_exec_domain", lambda: ("host", ["ev"]))
        p = rp._build_profile()
        assert p["cpu"]["effective"] == 64.0
        assert p["cpu"]["suggested_max_cpu"] == 64
        assert any("capped at nproc" in c for c in p["cpu"]["conflicts"])
        shutil.rmtree(fake_root, ignore_errors=True)

    # --- trigger regex: previously missed forms ------------------------------
    def test_trigger_new_forms(self):
        assert rp._SIZING_RE.search("python -c 'import os; os.fork()' ")
        assert rp._SIZING_RE.search("./run --workers 16 --epochs 1")
        assert rp._SIZING_RE.search("parallel -P 8 -j0 sanity.sh")
        assert rp._SIZING_RE.search("ls") is None


_ABSENT = object()
"""Sentinel: attribute genuinely absent from the fake os module."""


def _raise(exc):
    def _inner(pid=0):
        raise exc
    return _inner


class FakeOsModule:
    """Stand-in for the os module inside resource_probe (seam for _read_nproc).

    Attributes are attached per-instance ONLY when requested; anything left
    _ABSENT is a genuine missing attribute (getattr default path). sched may
    be an int (mask size) or an exception instance to raise on call."""

    def __init__(self, *, sched=_ABSENT, cpu_count=_ABSENT):
        if sched is not _ABSENT:
            if isinstance(sched, Exception):
                self.sched_getaffinity = _raise(sched)
            else:
                self.sched_getaffinity = lambda pid=0, _n=sched: list(range(_n))
        if cpu_count is not _ABSENT:
            self.cpu_count = lambda: cpu_count


class TestReadNprocAffinityAware:
    """Kernel affinity mask (a true constraint) is preferred; cpu_count is the
    fallback observation; a missing/unreadable affinity degrades cleanly."""

    def test_affinity_mask_preferred(self):
        real_sched = getattr(rp.os, "sched_getaffinity", None)
        if real_sched is None:
            pytest.skip("os.sched_getaffinity unavailable on this platform")
        n, src = rp._read_nproc()
        assert n == len(real_sched(0))
        assert "sched_getaffinity" in src

    def test_affinity_oserror_falls_back(self, monkeypatch):
        monkeypatch.setattr(rp, "os", FakeOsModule(sched=OSError("denied"), cpu_count=42))
        # FakeOs attaches a sched_getaffinity that raises OSError when called
        n, src = rp._read_nproc()
        assert n == 42
        assert "os.cpu_count" in src

    def test_no_sched_attr_falls_back(self, monkeypatch):
        monkeypatch.setattr(rp, "os", FakeOsModule(sched=_ABSENT, cpu_count=17))
        n, src = rp._read_nproc()
        assert n == 17
        assert "os.cpu_count" in src

    def test_cpu_count_none_is_unreadable(self, monkeypatch):
        monkeypatch.setattr(rp, "os", FakeOsModule(sched=OSError("x"), cpu_count=None))
        n, src = rp._read_nproc()
        assert n is None
        assert "unreadable" in src

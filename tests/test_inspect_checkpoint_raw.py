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

"""The standard-library raw-dump path of inspect_checkpoint.

Measured motivation (2026-11-27, gpt2-codegolf trial): the task ships
`gpt2-124M.ckpt` — 497,759,232 bytes of headerless little-endian float32
(numel 124,439,808, the published GPT-2 small parameter count exactly) — and
the task container has NEITHER torch NOR numpy. `inspect_checkpoint` was
registered and advertised as a checkpoint reader, but its core path imports
torch at the first line, so it raised ModuleNotFoundError and was never called
(0 uses). The agent's post-mortem: "I had no reference logits to diff against
(no numpy, no network)"; it spent four of seven correction rounds finding
layer-stride and orientation bugs "by arithmetic rather than by an oracle".

WHY THIS FILE EXISTS SEPARATELY: `tests/test_inspect_checkpoint.py` opens with
`pytest.importorskip("torch")`, so in a torch-less environment the ENTIRE file
is skipped — the tool's behaviour was unverified in precisely the environment
that broke it. These tests import no torch and no numpy, so they run there.

They also pin the memory bound: a 497MB dump must be read through windowed
mmap slices, never materialised as a whole-file Python array (~2GB of floats).
"""

import array
import json
import os
import re
import struct
import sys

import pytest

from flagscale_agent.react.tools.inspect_checkpoint import (
    _RAW_DTYPE_CODE,
    _raw_window_stats,
    inspect_checkpoint,
    inspect_raw_binary,
)
from flagscale_agent.react.tools.inspect_checkpoint import InspectCheckpointTool


def _write_raw(path, values, fmt="f"):
    with open(path, "wb") as f:
        f.write(struct.pack("<%d%s" % (len(values), fmt), *values))
    return path


def _machine_readable(report):
    m = re.search(r"MACHINE_READABLE: (\{.*\})", report)
    assert m, f"report carries no machine-readable tail:\n{report}"
    return json.loads(m.group(1))


def _expected(values):
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return {"min": min(values), "max": max(values), "mean": mean, "std": var ** 0.5}


# --------------------------------------------------------------------------
# The claim: element count and windowed statistics, standard library only.
# --------------------------------------------------------------------------

def test_numel_is_bytes_over_itemsize(tmp_path):
    p = _write_raw(tmp_path / "x.ckpt", [0.0] * 16)
    mr = _machine_readable(inspect_raw_binary(str(p)))
    assert mr["numel"] == 16
    assert mr["bytes"] == 64
    assert mr["dtype"] == "float32"
    assert mr["mode"] == "raw"


def test_window_statistics_match_planted_ground_truth(tmp_path):
    """Each window's min/max/mean/std must equal the values actually written."""
    vals = [1.0, 2.0, 3.0, 4.0,
            -1.0, -2.0, -3.0, -4.0,
            10.0, 20.0, 30.0, 40.0,
            0.5, 0.5, 0.5, 0.5]
    p = _write_raw(tmp_path / "x.ckpt", vals)
    with open(p, "rb") as f:
        mv = memoryview(f.read())
    for i in range(4):
        chunk = vals[i * 4:(i + 1) * 4]
        got = _raw_window_stats(mv, "f", i * 4, 4)
        want = _expected(chunk)
        for key in ("min", "max", "mean", "std"):
            assert got[key] == pytest.approx(want[key], rel=1e-6), (
                f"window {i} {key}: got {got[key]} want {want[key]}")


def test_a_headerless_dump_is_not_reported_as_a_verified_layout(tmp_path):
    """The report must refuse to imply key/offset/shape knowledge."""
    p = _write_raw(tmp_path / "x.ckpt", [1.0, 2.0, 3.0, 4.0])
    report = inspect_raw_binary(str(p))
    assert "no key names" in report
    assert "hypothesis" in report


# --------------------------------------------------------------------------
# Negative controls — the checks above must be able to go red.
# --------------------------------------------------------------------------

def test_unsupported_dtype_is_an_error_not_a_silent_int_read(tmp_path):
    p = _write_raw(tmp_path / "x.ckpt", [1.0, 2.0])
    assert inspect_raw_binary(str(p), dtype="bogus").startswith("ERROR")


def test_size_that_is_not_a_clean_multiple_is_rejected(tmp_path):
    """3 bytes cannot be float32 — that means a header or another dtype.

    Rejecting this is what stops a corrupt/containered file from being silently
    re-read as if it were raw.
    """
    p = tmp_path / "x.ckpt"
    p.write_bytes(b"\x00\x00\x00")
    report = inspect_raw_binary(str(p))
    assert report.startswith("ERROR")
    assert "not a multiple" in report


# --------------------------------------------------------------------------
# Self-selection: the registered tool must still answer when torch is absent.
# --------------------------------------------------------------------------

def test_execute_returns_a_report_when_torch_is_absent(tmp_path, monkeypatch):
    """The exact failure that made the tool useless in the task container.

    torch is made unimportable, then the REAL tool entry point is called. It
    must return a numeric report, not a ModuleNotFoundError.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    p = _write_raw(tmp_path / "gpt2-124M.ckpt", [1.0, 2.0, 3.0, 4.0])
    report = InspectCheckpointTool().execute(path=str(p))
    assert "MODE: raw" in report
    assert _machine_readable(report)["numel"] == 4


def test_execute_does_not_swallow_the_new_raw_parameters(tmp_path, monkeypatch):
    """A parameter exposed in the schema but dropped by execute() is a lie.

    The previous `execute()` hard-coded four arguments; the raw knobs must
    reach the implementation or the advertised interface is fiction.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    p = _write_raw(tmp_path / "x.ckpt", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    report = InspectCheckpointTool().execute(
        path=str(p), raw_window=2, raw_windows=3, raw_dtype="float32")
    rows = [ln for ln in report.splitlines() if re.match(r"^  \d+\s", ln)]
    assert len(rows) == 3, f"raw_windows not honoured, got {len(rows)} rows"


def test_tool_schema_advertises_the_raw_path(tmp_path):
    """The description must tell the agent the raw mode exists.

    A capability the schema never mentions is a capability the agent will never
    reach for — which is how a working tool stays at zero uses.
    """
    params = InspectCheckpointTool.parameters
    desc = InspectCheckpointTool.description.lower()
    assert "headerless" in desc or "raw" in desc
    for key in ("raw_dtype", "raw_window", "raw_windows"):
        assert key in params["properties"], f"schema does not expose {key}"


# --------------------------------------------------------------------------
# The accumulation bound: 497MB must not become a whole-file Python array.
# --------------------------------------------------------------------------

def test_large_dump_is_read_through_bounded_windows(tmp_path):
    """Peak RSS growth must stay far below the file size.

    A 497MB dump materialised as Python floats would allocate ~2GB (8 bytes of
    pointer + float object per element). Reading through windowed mmap slices
    keeps it flat. The file is created SPARSE (seek + one byte) so this costs no
    disk while presenting the real byte count to the reader.
    """
    numel = 124_439_808          # the real GPT-2 small element count
    path = tmp_path / "big.ckpt"
    with open(path, "wb") as f:
        f.seek(numel * 4 - 1)
        f.write(b"\x00")
    assert os.path.getsize(path) == 497_759_232

    def peak_rss_kb():
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM"):
                    return int(line.split()[1])
        return 0

    before = peak_rss_kb()
    report = inspect_raw_binary(str(path), window=4096, windows=8)
    after = peak_rss_kb()

    assert _machine_readable(report)["numel"] == numel
    grew_kb = after - before
    assert grew_kb < 200 * 1024, (
        f"RSS grew {grew_kb} KB on a 497MB dump — the file was materialised "
        f"instead of window-read")


# --------------------------------------------------------------------------
# torch PRESENT but the file is not a container it can parse.
#
# The container-side tasks ship a perfectly good raw dump into an interpreter
# that may well HAVE torch (the harness venv does). If that branch only reports
# "Failed to load checkpoint", the tool stays useless for a raw dump. No
# interpreter here has torch, so this branch is exercised with a stub — an
# untested branch is one that can only fail in production.
# --------------------------------------------------------------------------

import types


def _stub_torch(monkeypatch):
    """A torch stand-in good enough to enter the tensor branch, whose loaders raise."""
    torch = types.ModuleType("torch")

    class _T:
        pass

    torch.Tensor = _T

    def _load(*_a, **_k):
        raise RuntimeError("stub: not a torch container")

    torch.load = _load
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_raw_dump_still_read_when_torch_IS_present_but_cannot_load(tmp_path, monkeypatch):
    """The likely real case on the harness venv: torch exists, the file is raw."""
    _stub_torch(monkeypatch)
    p = _write_raw(tmp_path / "gpt2.ckpt", [1.0, 2.0, 3.0, 4.0])
    report = InspectCheckpointTool().execute(path=str(p))
    assert "MODE: raw" in report, report[:400]
    assert _machine_readable(report)["numel"] == 4


def test_a_corrupt_container_is_not_silently_reread_as_raw(tmp_path, monkeypatch):
    """The guard ON the fallback: a size that is not a clean multiple stays an ERROR.

    Without this, a corrupt .pt whose byte count happened to be a multiple of 4
    would be re-read as raw numbers and reported as a valid checkpoint — hiding
    the corruption this tool exists to catch.
    """
    _stub_torch(monkeypatch)
    p = tmp_path / "corrupt.pt"
    p.write_bytes(b"\x00\x01\x02")          # 3 bytes: not a multiple of 4
    report = InspectCheckpointTool().execute(path=str(p))
    assert report.startswith("ERROR"), report[:300]
    assert "Failed to load checkpoint" in report

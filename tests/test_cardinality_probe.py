"""cardinality_probe must be able to FAIL, and must never fake a PASS.

The probe exists to close a specific self-deception: an agent recovers a
collection by its own search, then reports the size its search produced, so a
missed member silently lowers the "expected" count and the answer looks
self-consistent. The probe's whole value is being INDEPENDENT of that search —
so these tests assert the three outcomes that make it trustworthy:

  * a COMPLETE delivery passes,
  * a SHORT delivery fails and names the missed members,
  * with no function to probe it reports UNMEASURED rather than PASS.

All fixtures are synthetic with a known hidden-unit count; the file carries no
task identifier and no target-specific constant.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[1] / "flagscale_agent" / "tools" / "cardinality_probe.py"


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cardfixture")
    (d / "fw.py").write_text(textwrap.dedent("""
        import numpy as np

        H = 6
        COLS = 4

        def forward(x):
            x = np.asarray(x, dtype=float).reshape(-1)
            rng = np.random.default_rng(7)
            A1 = rng.standard_normal((H, COLS)) * 0.7
            b1 = rng.standard_normal(H) * 0.3
            A2 = rng.standard_normal(H)
            return float(A2 @ np.maximum(A1 @ x + b1, 0.0) + 0.25)
    """), encoding="utf-8")
    (d / "mk.py").write_text(textwrap.dedent("""
        import numpy as np
        rng = np.random.default_rng(7)
        A1 = rng.standard_normal((6, 4)) * 0.7
        np.save("full.npy", A1)
        np.save("short.npy", A1[:4])
    """), encoding="utf-8")
    subprocess.run([sys.executable, "mk.py"], cwd=d, check=True)
    return d


def _run(d, *args):
    return subprocess.run(
        [sys.executable, str(PROBE), *args],
        cwd=d, capture_output=True, text=True, timeout=300,
    )


def test_probe_file_exists():
    assert PROBE.is_file(), f"missing {PROBE}"
    assert "cardinality_probe" in PROBE.name


def test_complete_delivery_passes(fixture_dir):
    r = _run(fixture_dir, "--forward", "fw.py", "--cols", "4",
             "--lines", "2", "--points", "20001", "--matrix", "full.npy")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "VERDICT: PASS" in r.stdout
    assert "6 regions" in r.stdout


def test_short_delivery_fails_and_reports_missed_members(fixture_dir):
    r = _run(fixture_dir, "--forward", "fw.py", "--cols", "4",
             "--lines", "2", "--points", "20001", "--matrix", "short.npy")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "VERDICT: FAIL" in r.stdout
    assert "MISMATCH: delivered rows=4" in r.stdout
    # The span-coverage check must independently point at the missing members.
    assert "no delivered row" in r.stdout


def test_unmeasurable_is_not_a_pass(fixture_dir):
    r = _run(fixture_dir, "--matrix", "full.npy")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "VERDICT: UNMEASURED" in r.stdout
    assert "VERDICT: PASS" not in r.stdout


def test_probe_carries_no_task_identifier():
    txt = PROBE.read_text(encoding="utf-8").lower()
    for tok in ("mteb", "stolen_a1", "primers", "cli_tool", "oligotm", "seed5", "terminal-bench"):
        assert tok not in txt, tok

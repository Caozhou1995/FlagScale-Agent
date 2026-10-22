"""convention_discrimination.py must never nudge the agent onto a QUALITY metric.

Regression for a live loss (2026-10-22, pytorch-model-cli): the tool printed

    VERDICT: SEPARABLE -- use these observations to decide: [...]
    For each, run the consumer's own reference path and keep the candidate
    that matches it.

The agent read "the consumer's own reference path" as "run it against real MNIST
labels", measured three normalisations (65.9% / 83.3% / 92.07%), and kept the
highest-scoring one. The grader's reference was the author's own raw/255
pipeline (the 65.9% candidate), so the delivery failed. The prompt's "Quality is
not acceptance" clause was present and had been read; the TOOL's own wording
overrode it. These tests pin the wording and the error-handling that made the
false verdict possible.
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "flagscale_agent", "tools", "convention_discrimination.py")


def _load():
    spec = importlib.util.spec_from_file_location("conv_disc", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(args):
    out = subprocess.run([sys.executable, TOOL] + args,
                         capture_output=True, text=True)
    return out.returncode, out.stdout + out.stderr


# ---------------------------------------------------------------- wording gates

def test_separable_verdict_forbids_quality_metric():
    """The SEPARABLE verdict must NOT license scoring candidates by accuracy."""
    m = _load()
    x = np.array([[1.0, 2.0], [5.0, 9.0]])
    with tempfile.TemporaryDirectory() as d:
        spec = os.path.join(d, "s.json")
        inp = os.path.join(d, "i.npy")
        json.dump({"candidates": {"raw": "x", "div255": "x / 255.0"}}, open(spec, "w"))
        np.save(inp, x)
        rc, out = _run(["--spec", spec, "--inputs", inp])
    assert rc == 0
    assert "SEPARABLE" in out
    # the fix: explicitly name the wrong oracle
    assert "GROUND-TRUTH LABELS is NOT that path" in out
    assert "EQUIVALENT" in out
    # the defect: the old bare licence must be gone
    assert "use these observations to decide" not in out


def test_separable_verdict_names_the_real_reference_path():
    m = _load()
    x = np.array([[1.0, 2.0], [5.0, 9.0]])
    with tempfile.TemporaryDirectory() as d:
        spec = os.path.join(d, "s.json")
        inp = os.path.join(d, "i.npy")
        json.dump({"candidates": {"raw": "x", "div255": "x / 255.0"}}, open(spec, "w"))
        np.save(inp, x)
        rc, out = _run(["--spec", spec, "--inputs", inp])
    assert "author's reference/solve program" in out
    assert "golden artifact" in out
    # and it must tell the agent what to do when that path is unavailable
    assert "UNDECIDED" in out
    assert "minimal-uncertainty default" in out


def test_not_separable_still_forbids_quality_metric():
    """The pre-existing NOT SEPARABLE guidance must not regress."""
    m = _load()
    x = np.array([[1.0, 2.0]])
    with tempfile.TemporaryDirectory() as d:
        spec = os.path.join(d, "s.json")
        inp = os.path.join(d, "i.npy")
        # both always collapse to the same observable
        json.dump({"candidates": {"a": "x * 0", "b": "x * 0"}}, open(spec, "w"))
        np.save(inp, x)
        rc, out = _run(["--spec", spec, "--inputs", inp, "--collapse", "np.argmax(v)"])
    assert "NOT SEPARABLE" in out
    assert "quality metric" in out


# ------------------------------------------------- error handling (false verdict)

def test_raising_candidates_do_not_manufacture_not_separable():
    """LIVE DEFECT: every raising candidate became one identical <error:...>
    string, so N broken candidates collapsed to a single 'observable' and the
    tool printed a FALSE 'NOT SEPARABLE' -- while in truth nothing was compared."""
    m = _load()
    x = np.array([[3.0, 4.0]])
    conventions = {"a": "np.not_a_real_attr(x)", "b": "1 / 0"}
    rows = m.discriminability(conventions, x)
    idx, distinct, outs, errored = rows[0]
    # no observable was produced at all -> must NOT be reported as 1 distinct
    assert outs == {}, f"raising candidates leaked an observable: {outs}"
    assert distinct == 1, "no candidates evaluated; nothing can be separable"
    assert set(errored) == {"a", "b"}, errored


def test_one_raising_candidate_does_not_hide_a_real_separation():
    """With one good candidate raising out, the survivors must still be compared
    against each other -- not collapsed by the presence of an error."""
    m = _load()
    x = np.array([[10.0, 20.0]])
    conventions = {"raw": "x", "div255": "x / 255.0", "broken": "np.nope(x)"}
    rows = m.discriminability(conventions, x)
    idx, distinct, outs, errored = rows[0]
    assert set(errored) == {"broken"}
    assert distinct == 2, "raw vs div255 must still be seen as separated"


def test_error_is_surfaced_to_the_user():
    """A silent error marker is how the false verdict shipped: the CLI must SAY
    a candidate failed, and must call the affected inputs inconclusive."""
    x = np.array([[3.0, 4.0], [1.0, 9.0]])
    with tempfile.TemporaryDirectory() as d:
        spec = os.path.join(d, "s.json")
        inp = os.path.join(d, "i.npy")
        json.dump({"candidates": {"good": "x", "bad": "np.nope(x)"}}, open(spec, "w"))
        np.save(inp, x)
        rc, out = _run(["--spec", spec, "--inputs", inp])
    assert "WARNING" in out
    assert "did not evaluate" in out
    assert "INCONCLUSIVE" in out
    assert "'bad'" in out or "bad" in out


def test_demo_still_runs_after_row_shape_change():
    """The demo path unpacks the row tuple; widening it must not crash it."""
    rc, out = _run(["--demo"])
    assert rc == 0, out
    assert "VERDICT:" in out


def test_row_shape_is_stable_for_callers():
    m = _load()
    x = np.array([[1.0, 2.0]])
    rows = m.discriminability({"a": "x", "b": "x * 2"}, x)
    idx, distinct, outs, errored = rows[0]
    assert idx == 0 and distinct == 2
    assert outs["a"] != outs["b"]
    assert errored == {}

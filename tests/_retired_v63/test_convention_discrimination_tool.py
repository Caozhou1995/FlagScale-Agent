"""The convention-discrimination doctrine and its in-package probe.

Prevents the failure observed in three independent pytorch-model-cli trajectories:
several candidate pre-processing conventions all produced the SAME answer on the
one visible sample, the agreement was read as confirmation, and the tie was then
broken with a quality proxy (accuracy 92.4%, logit margin 8.40 vs 1.83) that
pointed at the wrong convention.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT = (ROOT / "flagscale_agent" / "react" / "prompt.py").read_text()
TOOL = ROOT / "flagscale_agent" / "tools" / "convention_discrimination.py"


def test_clause_present():
    assert "A test that every candidate passes is not a test." in PROMPT


def test_requires_separating_observation():
    assert "SEPARATES the candidates" in PROMPT
    assert "zero discriminating power" in PROMPT


def test_judges_at_final_observable():
    assert "FINAL observable" in PROMPT
    assert "collapse" in PROMPT


def test_forbids_agreement_as_evidence():
    assert "do not report the agreement as evidence" in PROMPT


def test_single_sample_caution():
    assert "one example is not a held-out test set" in PROMPT


def test_tool_exists_and_is_importable():
    assert TOOL.is_file(), f"missing {TOOL}"
    code = ("import flagscale_agent.tools.convention_discrimination as m;"
            "print(m.__file__)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert "convention_discrimination.py" in r.stdout


def test_tool_demo_reports_separable():
    r = subprocess.run([sys.executable, str(TOOL), "--demo"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "VERDICT: SEPARABLE" in r.stdout


def test_tool_detects_non_separable_with_collapse():
    """The m50 case: three conventions, ONE observable -> NOT SEPARABLE."""
    spec = {"candidates": {
        "v/255": "x / 255.0",
        "pm1": "x / 255.0 * 2.0 - 1.0",
        "mean_std": "(x / 255.0 - 0.1307) / 0.3081",
    }}
    with tempfile.TemporaryDirectory() as td:
        spec_path = Path(td) / "spec.json"
        spec_path.write_text(json.dumps(spec))
        inputs = Path(td) / "inputs.npy"
        import numpy as np
        np.save(inputs, np.zeros((1, 4)))
        r = subprocess.run(
            [sys.executable, str(TOOL), "--spec", str(spec_path),
             "--inputs", str(inputs), "--collapse", "np.zeros(1)"],
            capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "NOT SEPARABLE" in r.stdout, r.stdout


def test_no_task_identifying_words_in_new_clause():
    banned = ["mteb", "torchmetrics", "sentence-transformers", "primers.fasta",
              "oligotm", "cli_tool", "stolen_A1", "seed5", "anneal", "MNIST",
              "pytorch", "dna", "image.png"]
    start = PROMPT.index("A test that every candidate passes")
    tail = PROMPT[start:]
    stop = tail.find("\n2.9 ")
    clause = (tail[:stop] if stop > 0 else tail).lower()
    for w in banned:
        assert w.lower() not in clause, f"leak word {w!r} in new clause"

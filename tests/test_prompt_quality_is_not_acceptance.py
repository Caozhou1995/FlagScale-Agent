"""The prompt must state that a candidate's own quality metric is not the acceptance predicate."""
import io
import os

PROMPT = os.path.join(os.path.dirname(__file__), "..", "flagscale_agent", "react", "prompt.py")


def _text():
    return io.open(PROMPT, encoding="utf-8").read()


def test_quality_is_not_acceptance_clause_present():
    t = _text()
    assert "Quality is not acceptance." in t
    assert "EQUIVALENCE to whatever the consumer's reference pipeline produced" in t


def test_quality_clause_forbids_selecting_convention_by_score():
    t = _text()
    i = t.index("Quality is not acceptance.")
    body = t[i:i + 2200]
    assert "never let a quality metric decide a convention" in body
    assert "minimal-uncertainty default" in body


def test_quality_clause_covers_shipped_reference_misread():
    """A shipped reference defines structure, not the steps it never named."""
    t = _text()
    i = t.index("Quality is not acceptance.")
    body = t[i:i + 2200]
    assert "does not document steps it never needed to name" in body
    assert "no credit for being better than it" in body


def test_quality_clause_does_not_leak_task_specifics():
    """Generic wording only: no task names, artifact filenames, or task constants."""
    t = _text()
    i = t.index("Quality is not acceptance.")
    body = t[i:i + 2200].lower()
    for banned in ("mnist", "pytorch-model-cli", "weight matrix", "1e-4", "0.99",
                   "primers.fasta", "mteb", "torchmetrics", "sentence-transformers",
                   "0.1307", "0.3081", "255.0", "oligotm", "primer3"):
        assert banned not in body, f"task-specific token leaked: {banned}"

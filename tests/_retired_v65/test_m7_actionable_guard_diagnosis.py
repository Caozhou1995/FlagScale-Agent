"""Regression tests for M7: guard block messages must be ACTIONABLE.

Live observation that motivated this (ds1017 TB2.1 trials): the agent knew the
rule — "observed must be findable in session tool results" — but not the target
exactness, so it re-pasted paraphrase after paraphrase, each round guessing,
until its per-turn wall-clock budget ran out. The fix does not relax any check;
it appends a diagnosis derived from the agent's OWN declaration and the
harness's OWN corpus: the longest slice that really occurs (paste this), or the
recorded output itself when nothing lines up.

These tests fail on the pre-M7 code: there the block message is the bare
structural clause with no echo of what the agent sent and no copyable text.
"""
import pytest

from flagscale_agent.react.guard.verification import (
    _corpus_tail_hint,
    _evidence_observed,
    _evidence_observed_diagnosis,
    _longest_common_snippet,
    _normalize_for_grounding,
    _check_unverified_assumptions_field,
    _check_deliverable_reading_field,
)

CORPUS_RAW = (
    "tool_results\n$ python probe.py\n"
    "measured accuracy 0.9731 over 1200 rows\n"
    "baseline accuracy 0.8814 over 1200 rows\n"
    "delta = +0.0917\n"
)
CORPUS = _normalize_for_grounding(CORPUS_RAW)

PARAPHRASE = (
    "I measured accuracy of about 97.3% across 1200 rows, beating the baseline "
    "by roughly 9 points."
)


def test_snippet_is_always_a_real_substring_of_the_corpus():
    """The promise the message makes to the agent: 'paste THAT and it will match'."""
    snippet = _longest_common_snippet(
        _normalize_for_grounding(PARAPHRASE), CORPUS)
    assert snippet, "a paraphrase of real output must share a run with it"
    assert snippet in CORPUS, "the offered slice MUST be paste-able verbatim"
    # And the fix actually works: pasting the offered slice passes the real check.
    assert _evidence_observed({"command": "x", "observed": snippet}, CORPUS)


def test_min_len_allows_a_short_shared_run():
    """min_len=12 was calibrated live: real paraphrases share ~12-20 chars."""
    shared = "measured accuracy"           # 17 chars, present verbatim
    needle = _normalize_for_grounding(shared + " was roughly ninety seven percent")
    assert shared in CORPUS
    snippet = _longest_common_snippet(needle, CORPUS)
    assert snippet.strip() == shared
    assert 12 <= len(snippet.strip()) < 24, (
        "min_len=12 must admit a run shorter than the old 24-char floor")


def test_diagnosis_echoes_the_copyable_slice_not_just_the_rule():
    msg = _evidence_observed_diagnosis(
        {"command": "python probe.py", "observed": PARAPHRASE}, CORPUS)
    assert "measured accuracy" in msg, "must echo real corpus text"
    assert "Paste THAT text byte-for-byte" in msg


def test_diagnosis_shows_recorded_output_when_nothing_matches():
    """Wrong-from-memory observation -> show what the corpus really holds."""
    msg = _evidence_observed_diagnosis(
        {"command": "cat r.json", "observed": '{"score": 0.99, "rows": 999}'},
        CORPUS)
    assert "None of your `observed`" in msg
    assert "0.9731" in msg, "agent must SEE the real recorded text"
    assert "guaranteed to match" in msg


def test_diagnosis_empty_corpus_tells_agent_to_run_the_probe():
    msg = _evidence_observed_diagnosis({"command": "c", "observed": "x"}, "")
    assert "RUN the probe first" in msg


def test_corpus_tail_hint_is_bounded_and_matches():
    hint = _corpus_tail_hint(CORPUS, tail_chars=40)
    assert "0.0917" in hint
    body = hint.split("---8<---", 1)[1].split("--->8---", 1)[0].strip()
    body = body.lstrip("…").strip()
    assert body in CORPUS


def test_corpus_tail_hint_empty_for_no_corpus():
    assert _corpus_tail_hint("") == ""


def test_assumptions_gate_end_to_end_carries_the_fix():
    """The real gate (not a unit stub) must append the diagnosis on failure."""
    field = [{
        "claim": "the baseline is 0.88",
        "affects_result": "yes",
        "probe": "python probe.py",
        "probe_output": "accuracy looked around 97%, much better",
    }]
    v = _check_unverified_assumptions_field(field, CORPUS)
    assert v is not None and v.reason == "assumptions_affecting_without_test"
    assert "[how to fix" in v.message
    assert "0.9731" in v.message or "accuracy" in v.message


def test_assumptions_gate_passes_on_verbatim_paste():
    """No regression in the pass direction: copying real text still succeeds."""
    field = [{
        "claim": "the baseline is 0.88",
        "affects_result": "yes",
        "probe": "python probe.py",
        "probe_output": "baseline accuracy 0.8814 over 1200 rows",
    }]
    assert _check_unverified_assumptions_field(field, CORPUS) is None


def test_hedge_union_gate_end_to_end_carries_the_fix():
    dr = {
        "deliverable": "steal.py",
        "grader_visible": False,
        "readings": [{
            "reading": "matrix of stolen logits",
            "source": "write steal.py that when run will output a matrix",
            "acceptance_test": "compare to true activations",
            "normalized_side": "neither",
        }],
        "hedge": {"shape": "none"},
        "if_none": {
            "why_intersection_empty": "only one reading exists",
            "union_attempt": {"command": "python union.py",
                              "observed": "the union output looked fine"},
            "extras_direction": "ignored",
            "readings_rejected": [],
        },
    }
    v = _check_deliverable_reading_field(dr, "", CORPUS)
    assert v is not None
    assert v.reason == "hedge_union_unattempted", v.reason
    assert "[how to fix" in v.message
    assert "0.9731" in v.message, "the real recorded output must be shown"


def test_check_strictness_is_unchanged():
    """M7 must NOT have loosened the gate: a fabricated observation still fails."""
    assert _evidence_observed(
        {"command": "c", "observed": "totally invented text not in corpus"}, CORPUS) is False
    assert _evidence_observed({"command": "c", "observed": ""}, CORPUS) is False
    assert _evidence_observed({"command": "c"}, CORPUS) is False
    assert _evidence_observed(None, CORPUS) is False

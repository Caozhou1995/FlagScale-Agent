"""Regression tests for the calibration tool-argument transport drift.

Measured failure (2026-10-27): the model submitted a perfectly standard
calibration object and was refused SIX times in a row with "calibration is
malformed". Two defects combined:

1. The argument arrived JSON-ENCODED (a str), while check_calibration only
   accepted a dict. `steps` and `oracle` already tolerated this drift; the
   new `calibration` parameter did not -- an asymmetric defence.
2. The refusal text was a fixed restatement of the schema: it never echoed the
   shape that actually arrived, so every retry was identical and the
   refuse-then-correct loop could not close.
"""
import json
import os

import pytest
import yaml

from flagscale_agent.react.instruments import (
    _describe_received,
    check_calibration,
)
from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.tools.plan_create import PlanCreateTool

CAL = {
    "instrument": "oligotm -tp 1 -sc 1 -mv 50 -dv 2 -n 0.8 -d 500 <region>",
    "known_case": "no_known_case",
    "observed": "no_known_case",
    "matches": "no_known_case",
    "instrument_input": {
        "expression": "tm = run(seq[start:end])",
        "source": "the span the task text fixes",
        "mismatch_checked": "second span read 61.8 vs 67.4",
    },
}
ORACLE = {
    "premise_source": "provided",
    "oracle": "the task's own checker, replayed",
    "first_exercise": "echo exercise",
    "named_arbiter": "none",
    "deliverable_path": "/tmp/artifact",
}
STEPS = [{"title": "do the work", "acceptance": ["a runnable assertion"]}]


def _tool(tmp_path, tag="s"):
    plan = TaskPlan(str(tmp_path))
    return plan, PlanCreateTool(plan, session_id=tag)


def _persisted(tmp_path):
    for name in sorted(os.listdir(str(tmp_path))):
        if name.startswith("plan_") and name.endswith(".yaml"):
            with open(os.path.join(str(tmp_path), name)) as fh:
                return yaml.safe_load(fh)
    raise AssertionError("no plan YAML persisted")


# --- defect 1: the JSON-string transport drift ------------------------------

def test_calibration_accepts_json_encoded_string():
    """The live failure: a JSON string must be decoded, not refused."""
    assert check_calibration(json.dumps(CAL)) is None


def test_calibration_json_string_is_persisted_as_an_object(tmp_path):
    plan, tool = _tool(tmp_path)
    out = tool.execute(title="t", steps=STEPS, oracle=ORACLE,
                       calibration=json.dumps(CAL))
    assert not out.startswith("ERROR"), out
    # judge at the FAR end: the YAML the consumer reads, not the return string
    assert _persisted(tmp_path)["calibration"] == CAL


def test_calibration_plain_dict_still_accepted(tmp_path):
    """The decode must not break the already-working path."""
    plan, tool = _tool(tmp_path)
    out = tool.execute(title="t", steps=STEPS, oracle=ORACLE, calibration=CAL)
    assert not out.startswith("ERROR"), out
    assert _persisted(tmp_path)["calibration"] == CAL


def test_unparseable_string_is_still_refused(tmp_path):
    """Decoding must not turn a genuinely bad declaration into a pass."""
    plan, tool = _tool(tmp_path)
    out = tool.execute(title="t", steps=STEPS, oracle=ORACLE,
                       calibration="not json at all")
    assert out.startswith("ERROR")
    assert "malformed" in out


# --- defect 2: the refusal must echo what actually arrived ------------------

def test_refusal_echoes_received_shape_for_undecodable_text():
    """A JSON-ENCODED object is now accepted; the echo contract moves to the
    refusal of genuinely undecodable text (or wrong-typed values)."""
    reason, message = check_calibration("[]")  # valid JSON, not an object
    assert reason == "calibration_not_dict"
    assert "list" in message
    assert "What arrived" in message or "instrument" in message


def test_refusal_echoes_received_keys_for_objects():
    reason, message = check_calibration({"instrument": "x", "matches": "MAYBE"})
    assert reason == "calibration_bad_matches"
    assert "MAYBE" in message          # the actual offending VALUE
    assert "instrument" in message     # the keys that did arrive


def test_refusal_names_the_absent_key():
    reason, message = check_calibration({"instrument": "x"})  # no matches
    assert reason == "calibration_field_missing"
    assert "matches" in message


def test_describe_received_covers_container_and_scalar_shapes():
    assert "list of 2" in _describe_received([1, 2])
    assert "object with keys" in _describe_received({"a": 1})
    assert "None" in _describe_received(None)
    assert "str" in _describe_received("plain text")


# --- defect 3: the REVIEW step must not carry a schema fragment -------------

def test_review_step_carries_no_stray_calibration_schema(tmp_path):
    plan, tool = _tool(tmp_path)
    tool.execute(title="t", steps=STEPS, oracle=ORACLE, calibration=CAL)
    doc = _persisted(tmp_path)
    review = [s for s in doc["steps"] if str(s.get("title", "")).startswith("Review")]
    assert review, "the fixed Review step is missing"
    assert "calibration" not in review[0], (
        "a JSON-schema fragment leaked into the step definition")

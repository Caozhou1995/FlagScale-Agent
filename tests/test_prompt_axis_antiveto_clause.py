"""Anti-decay test: no-reference-artifact is not a veto (lever v10).

The raman-fitting failure class: a candidate axis transform lands the
domain's literature-standard feature positions numerically on observed
peaks, but the agent keeps raw units because "no consumer reference
artifact exists" — a procedural override of a numerically-decided question.
The clause in react/prompt.py (Calibrate-a-measured-dataset's-AXIS section)
must keep the explicit anti-veto rule.
"""
import pathlib

PROMPT = pathlib.Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text()

ANTI_VETO = "THE ABSENCE OF A CONSUMER REFERENCE ARTIFACT IS NOT A VETO"
NUM_AUTHORITY = "that numerical match IS the external authority"


def test_anti_veto_rule_present():
    assert ANTI_VETO in TEXT, "anti-veto rule missing from prompt.py"


def test_numerical_match_is_authority():
    assert NUM_AUTHORITY in TEXT, "numerical-match-is-authority rule missing"


def test_rule_lives_in_axis_calibration_region():
    anchor = TEXT.find("Calibrate a measured dataset")
    veto = TEXT.find(ANTI_VETO)
    assert anchor != -1 and veto != -1
    assert 0 < veto - anchor < 2500, "anti-veto rule drifted from the axis-calibration region"


def test_no_task_literal_leakage():
    window_start = TEXT.find(ANTI_VETO)
    window_end = TEXT.find("\n", window_start + 1)
    window = TEXT[window_start: window_end if window_end != -1 else window_start + 1200]
    for marker in ["1580", "2670", "raman", "graphene", "cm⁻¹", "6328"]:
        assert marker not in window, f"task literal leaked: {marker}"

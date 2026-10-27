"""Test-layer shim: supply a default `calibration` declaration.

WHY THIS EXISTS. `plan_create` now REFUSES to create a plan without a
`calibration` declaration (instrument 1: prove your check works on a case whose
answer you already know, before you trust it). That refusal is the mechanism.

But most tests in this suite are not about calibration at all -- they exercise
step parsing, rendering, ledgers, forbidden lists, acceptance criteria. Those
tests call `PlanCreateTool(...).execute(title=..., steps=...)` and would now fail
on a requirement unrelated to what they measure.

This module supplies the ONE thing those tests need: a default calibration. It is
a test-layer shim and nothing more -- production code has no such default, which
is the whole point (the refusal is only load-bearing because there is no way to
skip it).

IMPORTANT: tests that CHECK the refusal must pass `calibration=` explicitly
(including `None`) so this fixture does not mask it. See
tests/test_instruments_four_levers.py, which drives the tool directly.
"""

DEFAULT_CALIBRATION = {
    "instrument": "pytest fixture default (not a real instrument)",
    "known_case": "fixture-supplied known case",
    "observed": "fixture-supplied observation",
    "matches": "yes",
}


def apply_default_calibration(tp, session_id=""):
    """Return a PlanCreateTool whose execute() injects a default calibration.

    Wraps the real tool rather than monkeypatching the module, so the production
    refusal path is untouched and stays testable.
    """
    from flagscale_agent.react.tools.plan_create import PlanCreateTool

    class _WithCalibration(PlanCreateTool):
        def execute(self, **kwargs):
            kwargs.setdefault("calibration", dict(DEFAULT_CALIBRATION))
            return super().execute(**kwargs)

    return _WithCalibration(tp) if not session_id else _WithCalibration(tp, session_id)

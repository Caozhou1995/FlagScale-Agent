"""The deliverable must carry the DERIVED value, not the raw one.

Measured failure (raman-fitting, 2026-11-04 batch, reward=0, exited NORMALLY
at 513.9s -- not a timeout):

  The agent fitted two Lorentzian peaks on the raw x axis and shipped
  x0=3745.36 / 6327.91 to /app/results.json. The grader's hard-coded
  expectations were x0=1580.3 / 2670.08. The two ratios agree to 5 digits
  (2.370033 and 2.369934) and the peak-separation ratio matches the truth to
  4 digits (1.689533 vs 1.689603) -- i.e. the fit was STRUCTURALLY CORRECT.
  The agent's own reasoning derived the conversion (89 mentions of cm^-1,
  "x * 0.426 cm^-1 ~ 1580"), but fit.py wrote '"x0": float(pG[2])' -- the raw
  axis value. Reasoning concluded one number; the bytes carried another.

This is a distinct discipline from "does my check pass": it is about the
seam between a derived quantity and the field that ships it. The prompt must
name that seam, and must not let the existing "read every shared value off
the target, never compute it locally" rule be read as licence to ship the
untransformed number.
"""

from __future__ import annotations

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def test_derived_value_must_be_what_ships() -> None:
    low = SYSTEM_PROMPT_STATIC.lower()
    # the discipline is named
    assert "transformed value" in low
    # the forbidden alternative is named explicitly (raw axis / index / start point)
    assert "raw axis point" in low or "array index" in low
    # the read-back comparison is the stated detection action
    assert "read it back" in low

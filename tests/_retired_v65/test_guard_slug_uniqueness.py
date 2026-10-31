"""Regression lock for the guard-slug uniqueness invariant.

The `[failing check: {reason}]` diagnostic (see flagscale_agent/react/guard/__init__.py)
surfaces `verdict.reason` to the model. That diagnostic is only actionable if each
`reason` names ONE failing sub-rule. Historically several slugs were COARSE — a single
name (e.g. `hedge_unproven`, `deliverable_reading_malformed`,
`assumptions_declaration_malformed`) was emitted from many distinct code paths, so the
model was told the CATEGORY, not which rule failed, and guessed/retried.

This test fails if any `reason="..."` literal in verification.py is emitted from more
than one site — i.e. if a coarse slug is re-introduced.
"""
import re
from pathlib import Path

_VERIFICATION = (
    Path(__file__).resolve().parent.parent
    / "flagscale_agent" / "react" / "guard" / "verification.py"
)


def _reason_emissions():
    """Return {slug: count} for every literal `reason="..."` in verification.py."""
    src = _VERIFICATION.read_text(encoding="utf-8")
    counts = {}
    for m in re.finditer(r'reason="([a-z_0-9]+)"', src):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def test_every_reason_slug_is_unique():
    """No slug may be emitted from more than one code path — a coarse slug
    collapses distinct sub-conditions and defeats the `[failing check:]` diagnostic."""
    counts = _reason_emissions()
    dupes = {s: c for s, c in counts.items() if c > 1}
    assert not dupes, (
        "coarse guard slug(s) re-introduced — each `reason=` must name ONE "
        "sub-condition. Offenders (slug: sites): %r" % dupes
    )


def test_the_three_historically_coarse_families_are_specialised():
    """The three families this change split must no longer carry their old coarse
    names — a regression here means the split was reverted."""
    counts = _reason_emissions()
    for coarse in ("hedge_unproven",
                   "deliverable_reading_malformed",
                   "assumptions_declaration_malformed"):
        assert coarse not in counts, (
            "coarse family slug %r is back — re-split it into per-sub-condition "
            "slugs" % coarse
        )

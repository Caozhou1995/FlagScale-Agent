# Retired on 2026-11-03 (ruling: current code is the specification)

test_prompt_g3_build_toward_bound.py — the G3 sufficiency clause ("BUILD
TOWARD THE BOUND FROM A BOUND-LEGAL VERSION", commit ce4203e, 2026-10-29) is
absent from the current 10-section THE LOOP prompt (grep = 0 for all four
anchors, HEAD also 0). The land-early mechanism in Section 7 carries the
surviving half of the idea (bank a minimal scorable artifact early), but the
direction-of-construction wording no longer exists in any file, so there is
nothing left to pin. Restore with: git show ce4203e -- <path>.

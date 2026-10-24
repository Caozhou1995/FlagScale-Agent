# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anti-decay tests for the L0-L4 robustness matrix in PRINCIPLE 2.

Measured failure this closes: video-processing trials kept losing to a
detector that latched onto absolute frame positions (takeoff=112 in a
[219,223] window) — a robustness failure the prompt's old prose clause
("generalize to perturbed inputs") was too vague to prevent. The L0-L4
matrix operationalizes generalization as five checkable axes; these tests
pin all five so no future edit silently softens them back to prose.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _principle2() -> str:
    start = SYSTEM_PROMPT_STATIC.index("PRINCIPLE 2")
    end = SYSTEM_PROMPT_STATIC.index("PRINCIPLE 3")
    return SYSTEM_PROMPT_STATIC[start:end]


def test_matrix_present_with_all_five_axes():
    sec = _principle2()
    for axis in ("L0 CALIBRATION", "L1 EQUIVARIANCE", "L2 INVARIANCE",
                 "L3 NEGATIVE CONTROL", "L4 MARGIN"):
        assert axis in sec, f"robustness axis {axis!r} missing from PRINCIPLE 2"


def test_axes_are_ordered_l0_to_l4():
    sec = _principle2()
    idx = [sec.index(a) for a in ("L0 CALIBRATION", "L1 EQUIVARIANCE",
                                  "L2 INVARIANCE", "L3 NEGATIVE CONTROL",
                                  "L4 MARGIN")]
    assert idx == sorted(idx), "L0-L4 axes must appear in escalation order"


def test_each_axis_states_its_law_and_its_probe():
    sec = _principle2()
    # Each axis must carry BOTH the law (what must hold) and the concrete
    # probe (how to falsify) — a bare axis name is not an instruction.
    laws = [
        ("L0 CALIBRATION", "must reproduce"),
        ("L1 EQUIVARIANCE", "MUST move"),
        ("L2 INVARIANCE", "must NOT move"),
        ("L3 NEGATIVE CONTROL", "must NOT be reported"),
        ("L4 MARGIN", "must not flip"),
    ]
    probes = ["Synthesize", "RUN", "synthesize", "perturb", "run", "SWEEP", "jitter", "sweep"]
    for axis, law in laws:
        pos = sec.index(axis)
        chunk = sec[pos:pos + 700]
        assert law in chunk, f"{axis} states no law containing {law!r}"
        assert any(p in chunk for p in probes), f"{axis} states no concrete probe"


def test_metamorphic_testing_anchor_survives():
    # The equivariance axis names the standard method so the model (and the
    # next editor) knows this is metamorphic testing, not ad-hoc perturbation.
    assert "metamorphic testing" in _principle2()


def test_matrix_replaces_not_duplicates_the_old_prose():
    sec = _principle2()
    # The vague prose this matrix replaced must not linger alongside it —
    # a softened duplicate would let an editor "satisfy" the test with prose.
    assert '"generalize" is not a testable instruction' in sec

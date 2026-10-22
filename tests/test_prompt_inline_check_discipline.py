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

"""Regression tests for the inline-check action discipline.

Measured failure this closes: across real graded runs (per-task paired wins vs
losses, same DS-era model, per-task all-wins vs equal losses), winning and
losing trajectories spent the SAME number of actions (medians 16-63 both
sides) but losing runs wrote verification as a separate step (`cat >
check.py`), chased process health (`exit=$?`, `cat err.txt`, `wc -c out`)
instead of the artifact's semantics, and had a lower share of inline checks
(semantic inspection chained onto the producing action: dna-assembly 0.81 vs
0.53, raman-fitting 0.64 vs 0.50, video 0.86 vs 0.81). The new FAR-end bullet
pins that discipline so a future edit cannot silently drop it.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _far_end_block() -> str:
    text = SYSTEM_PROMPT_STATIC
    start = text.index("Verify the FAR end, not the near end")
    end = text.index("Beyond the one visible sample")
    return text[start:end]


def test_action_quality_bullet_present():
    sec = _far_end_block()
    assert "Is each action doing WORK or just HUNTING?" in sec


def test_bullet_names_the_measured_split():
    # It must state the measured fact (same action count, different quality),
    # not a vague exhortation.
    sec = _far_end_block()
    assert "SAME number of actions" in sec
    assert "QUALITY" in sec


def test_bullet_names_inline_check_and_error_hunt():
    # Both contrasted behaviors must be named concretely.
    sec = _far_end_block()
    assert "exit=$?" in sec
    assert "grep -c" in sec


def test_bullet_demands_chained_semantic_check():
    # The operative rule: chain the check onto the producing action, assert the
    # RESULT not the exit code.
    sec = _far_end_block()
    assert "chain the check onto" in sec
    assert "exit code" in sec


def test_bullet_forbids_written_but_unrun_checker():
    sec = _far_end_block()
    assert "you write but never run is worth nothing" in sec


def test_no_task_nouns_leak():
    # The rule is general discipline: no task names, paths, or answers.
    sec = _far_end_block()
    for banned in ("filter-js", "extract-elf", "dna-assembly", "raman",
                   "pytorch-model", "video-processing", "primers.fasta",
                   "primers.fasta"):
        assert banned not in sec

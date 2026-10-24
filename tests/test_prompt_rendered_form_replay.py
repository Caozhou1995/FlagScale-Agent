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

"""RED-phase tests for the DELIVERED-FORM REPLAY discipline.

Failure class this locks (tbench full library, 831 reward=0 trials,
analysis/wrong_taxonomy.py): FORMAT-family assertions = 21.8% of all
wrong-answer failures. Reference instance: a sampler task where every
number was correct (posterior means, Rhat, n_eff all within tolerance)
yet the deliverable failed because stdout lacked three exact lines the
verifier asserts on. The task text names the operation ("print the
sampling summary") but NOT the precise line set — the exact lines are
knowable only by replaying the artifact's own output against the
task's OWN example/format clause, not by re-reading the prose.

The failure shape: the agent verified CONTENT (values, tolerance) and
declared done, without ever RUNNING its artifact and comparing the
rendered output's SHAPE (line names, section headers, file set, key
order) against the form the task's own materials imply. Content
correct + form wrong = reward 0. Existing clauses cover "reconstruct
the grader's contract from the task text" (deriving expectations) and
"replay the derivation" (recomputing derived quantities) — neither
covers RUNNING the artifact and diffing its rendered FORM.
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _prompt_low() -> str:
    return " ".join(SYSTEM_PROMPT_STATIC.split()).lower()


def test_form_replay_clause_present():
    """The clause itself must exist in the prompt."""
    low = _prompt_low()
    assert "rendered-form replay" in low


def test_form_replay_requires_running_the_artifact():
    """The mechanism: RUN the deliverable and capture its actual output."""
    low = _prompt_low()
    assert "run the artifact" in low
    assert "capture its actual output" in low


def test_form_replay_targets_shape_not_content():
    """Line names, headers, key order, file set — the axes FORMAT failures live on."""
    low = _prompt_low()
    for axis in ("line names", "section headers", "key order", "file set"):
        assert axis in low, f"missing form axis: {axis}"


def test_form_replay_diffs_against_task_materials():
    """The comparison target is the task's OWN example/format clause, not a guess."""
    low = _prompt_low()
    assert "the task's own example" in low


def test_form_replay_names_the_failure_shape():
    """Content-correct + form-wrong = rejection; naming it makes the trap visible."""
    low = _prompt_low()
    assert "the values are right yet the form is wrong" in low
    assert "every number inside tolerance" in low


def test_form_replay_allows_skip_when_no_format_is_stated():
    """Anti-guess guard: when the task states NO format, re-read and move on — do not guess one."""
    low = _prompt_low()
    assert "if the task states no format" in low


def test_form_replay_stays_before_the_completion_signal():
    """The replay must happen BEFORE declaring done, not after."""
    low = _prompt_low()
    assert "before you emit" in low


def test_form_replay_clause_generic():
    """No originating-task specifics may leak into the generic prompt."""
    m = re.search(r"RENDERED-FORM REPLAY:.*?(?=\n- )", SYSTEM_PROMPT_STATIC, re.S)
    assert m, "clause block not found"
    low = " ".join(m.group(0).split()).lower()
    for leak in (
        "sampling for model",
        "mcmc",
        "frame.bmp",
        "rhat",
        "n_eff",
        "posterior",
        "sampler",
    ):
        assert leak not in low, f"prompt leaked task-specific identifier: {leak}"
    assert not re.search(r"\b\d+\.\d{4,}\b", low)

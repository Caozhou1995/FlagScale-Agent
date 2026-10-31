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

"""Tests for CompletionArtifactGate.

The failure this guard targets is real and measured, so the tests below are
built from the ACTUAL verifier outputs of the four trials that lost:

  regex-chess            FileNotFoundError: /app/re.json
  make-mips-interpreter  File /tmp/frame.bmp does not exist
  sam-cell-seg           /app/test_output.csv does not exist
  fix-ocaml-gc           assert '40 tests passed' in ''

Every test therefore asserts a DIRECTION (block vs allow) against a case whose
answer is known from those recorded runs — not merely that the code runs.
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.completion_artifact import (
    CompletionArtifactGate,
    named_artifacts,
)


TASK_RE = (
    "Write your regex to /app/re.json. The file must be smaller than 10MB.\n"
    "Then verify the game results."
)

TASK_TWO = (
    "Produce /tmp/frame.bmp from the interpreter, and write the summary to "
    "/app/out/summary.csv."
)


def _ctx(task: str, history: list[dict], assistant_text: str, llm_responded=True):
    return GuardContext(
        messages=[{"role": "user", "content": task}] + history,
        assistant_text=assistant_text,
        llm_responded=llm_responded,
    )


# --------------------------------------------------------------------------
# path extraction
# --------------------------------------------------------------------------


def test_extracts_absolute_deliverable_paths_from_task():
    got = named_artifacts(TASK_TWO)
    assert "/tmp/frame.bmp" in got
    assert "/app/out/summary.csv" in got


def test_ignores_tooling_files_that_are_not_deliverables():
    got = named_artifacts("Install deps with /app/requirements.txt then write /app/answer.md")
    assert "/app/requirements.txt" not in got
    assert "/app/answer.md" in got


def test_relative_names_are_not_gated():
    # A bare "re.json" is ambiguous (cwd unknown) -> gating it would false-block.
    assert named_artifacts("Write re.json in the current directory.") == []


# --------------------------------------------------------------------------
# the recorded failure shapes -> MUST BLOCK
# --------------------------------------------------------------------------


def test_blocks_when_named_artifact_never_touched():
    """regex-chess shape: /app/re.json named, never created, agent completes."""
    ctx = _ctx(
        TASK_RE,
        history=[
            {"role": "tool", "content": "⚡ shell ls -la /app\nno such file"},
        ],
        assistant_text="All done.\n[TASK_COMPLETE]",
    )
    verdict = CompletionArtifactGate().check_pre(ctx)
    assert verdict is not None, "must block: the named artifact was never observed"
    assert verdict.action == "block"
    assert "/app/re.json" in verdict.message
    assert "test -e" in verdict.message


def test_blocks_when_only_one_of_two_artifacts_probed():
    """make-mips shape: frame.bmp probed, summary.csv never mentioned."""
    ctx = _ctx(
        TASK_TWO,
        history=[
            {"role": "tool", "content": "$ ls -la /tmp/frame.bmp\n-rw-r--r-- 1 root root 900 frame.bmp"},
        ],
        assistant_text="[TASK_COMPLETE]",
    )
    verdict = CompletionArtifactGate().check_pre(ctx)
    assert verdict is not None
    assert "/app/out/summary.csv" in verdict.message
    assert "/tmp/frame.bmp" not in verdict.message.split("\n")[1]


def test_blocks_on_empty_file_evidence_is_still_required():
    """fix-ocaml shape: the tests-output path was never even read."""
    ctx = _ctx(
        "Save the test output to /app/tests.txt and confirm 40 tests passed.",
        history=[{"role": "tool", "content": "⚡ shell make test\nAll tests done"}],
        assistant_text="Verified. [TASK_COMPLETE]",
    )
    verdict = CompletionArtifactGate().check_pre(ctx)
    assert verdict is not None, "must block: /app/tests.txt was never observed"


# --------------------------------------------------------------------------
# cases that MUST NOT block (false-positive control)
# --------------------------------------------------------------------------


def test_allows_when_every_named_artifact_was_probed():
    ctx = _ctx(
        TASK_TWO,
        history=[
            {"role": "tool", "content": "$ ls -la /tmp/frame.bmp\n-rw-r--r-- 1 root root 900 frame.bmp"},
            {"role": "tool", "content": "$ wc -c /app/out/summary.csv\n128 /app/out/summary.csv"},
        ],
        assistant_text="Everything checked.\n[TASK_COMPLETE]",
    )
    assert CompletionArtifactGate().check_pre(ctx) is None


def test_allows_when_task_names_no_absolute_artifact():
    ctx = _ctx(
        "Explain how a transformer works.",
        history=[],
        assistant_text="[TASK_COMPLETE]",
    )
    assert CompletionArtifactGate().check_pre(ctx) is None


def test_allows_on_need_user_input_sentinel():
    """Only [TASK_COMPLETE] is gated; handing back needs no artifact proof."""
    ctx = _ctx(TASK_RE, history=[], assistant_text="I need a decision.\n[NEED_USER_INPUT]")
    assert CompletionArtifactGate().check_pre(ctx) is None


def test_ignores_stale_history_sentinel_without_llm_responded():
    """A prior turn's trailing marker must not re-trigger at a new turn's top."""
    ctx = _ctx(TASK_RE, history=[], assistant_text="[TASK_COMPLETE]", llm_responded=False)
    assert CompletionArtifactGate().check_pre(ctx) is None


def test_agent_prose_mentioning_the_path_does_not_satisfy_the_check():
    """The agent's OWN text is not an observation — only a tool result is."""
    ctx = _ctx(
        TASK_RE,
        history=[
            {"role": "assistant", "content": "I wrote /app/re.json successfully."},
        ],
        assistant_text="Done. [TASK_COMPLETE]",
    )
    verdict = CompletionArtifactGate().check_pre(ctx)
    assert verdict is not None, "agent's own claim must not count as a probe"


def test_verdict_is_overridable_so_trivial_runs_cannot_livelock():
    ctx = _ctx(TASK_RE, history=[], assistant_text="[TASK_COMPLETE]")
    verdict = CompletionArtifactGate().check_pre(ctx)
    assert verdict is not None
    assert verdict.overridable is True


# --------------------------------------------------------------------------
# both REAL harness message shapes must be recognised (history._is_tool_result)
# --------------------------------------------------------------------------


def test_recognises_anthropic_style_tool_result_block():
    """Anthropic form: role=user, content=[{type:tool_result, content:...}]."""
    ctx = GuardContext(
        messages=[
            {"role": "user", "content": TASK_RE},
            {"role": "user", "content": [
                {"type": "tool_result", "content": "$ ls -la /app/re.json\n-rw-r--r-- 1 root root 220 /app/re.json"},
            ]},
        ],
        assistant_text="[TASK_COMPLETE]",
        llm_responded=True,
    )
    assert CompletionArtifactGate().check_pre(ctx) is None, (
        "a tool_result block IS an observation -- must not block"
    )


def test_anthropic_block_without_the_path_still_blocks():
    """Negative control on the same shape: the block exists but never touched it."""
    ctx = GuardContext(
        messages=[
            {"role": "user", "content": TASK_RE},
            {"role": "user", "content": [
                {"type": "tool_result", "content": "$ ls -la /app\nother.txt"},
            ]},
        ],
        assistant_text="[TASK_COMPLETE]",
        llm_responded=True,
    )
    verdict = CompletionArtifactGate().check_pre(ctx)
    assert verdict is not None and verdict.action == "block"
    assert "/app/re.json" in verdict.message


# --------------------------------------------------------------------------
# stateful absorption: eviction must not erase an earlier REAL observation
# --------------------------------------------------------------------------


def test_evicted_early_observation_is_not_forgotten():
    """Guard is stateful: an early probe absorbed before eviction still counts
    even when the history snapshot later shows only the evict placeholder."""
    guard = CompletionArtifactGate()
    # Turn 1: a real observation of the artifact.
    early = GuardContext(
        messages=[
            {"role": "user", "content": TASK_RE},
            {"role": "tool", "content": "ls -la /app/re.json\n-rw-r--r-- 1 root root 220 /app/re.json"},
        ],
        assistant_text="running...", llm_responded=False,
    )
    assert guard.check_pre(early) is None  # absorbs silently

    # Turn 2: context pressure evicted the tool result; only placeholder remains.
    late = GuardContext(
        messages=[
            {"role": "user", "content": TASK_RE},
            {"role": "user", "content": "[evicted | index=2 | role=tool | 120 tokens]"},
        ],
        assistant_text="[TASK_COMPLETE]", llm_responded=True,
    )
    assert guard.check_pre(late) is None, (
        "absorbed evidence must survive eviction -- blocking here would punish a "
        "run that did everything right"
    )

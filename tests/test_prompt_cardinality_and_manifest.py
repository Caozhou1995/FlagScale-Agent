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

"""Regression tests for CARDINALITY and DELIVERABLE-MANIFEST disciplines.

Evidence class this locks (two independent trials of the same 4-task set):
  * model-extraction-relu-logits: the agent delivered stolen_A1.npy with
    shape (20, 10) and reported "rows 20 true 20" plus a per-row residual of
    2.8e-9; the grader constructs its reference with np.random.randn(30, 10),
    so it asserts over 30 rows and failed with "Failed to match rows: [24]".
    The gap was a COLLECTION CARDINALITY shortfall (20 vs 30), invisible to
    any per-item accuracy metric the agent measured.
  * pytorch-model-cli: the grader does open("/app/prediction.txt") and
    subprocess.run("/app/cli_tool /app/weights.json /app/image.png"); the
    trial failed with FileNotFoundError on prediction.txt and stdout '9'
    where '2' was expected. A deliverable that was never written at the
    expected path is a deliverable that was never shipped.

Both are invisible to a self-check that only re-reads what the agent chose
to produce, so the load-bearing fix is prompt text.

These tests assert the clauses are present in the STATIC prompt (a future
edit must not silently drop them) and that they stay GENERIC (no trial
identifier, no magic number copied out of one graded instance).
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

CARDINALITY_CLAUSE = "Does my answer have the right NUMBER of items"
MANIFEST_CLAUSE = "Which artifacts does the grader OPEN"


def test_cardinality_clause_present():
    assert CARDINALITY_CLAUSE in SYSTEM_PROMPT_STATIC


def test_manifest_clause_present():
    assert MANIFEST_CLAUSE in SYSTEM_PROMPT_STATIC


def test_cardinality_clause_is_generic():
    """The clause must not hard-code one graded instance's numbers or task name."""
    start = SYSTEM_PROMPT_STATIC.index(CARDINALITY_CLAUSE)
    end = SYSTEM_PROMPT_STATIC.index("\n- ", start)
    clause = SYSTEM_PROMPT_STATIC[start:end]
    for banned in ("model-extraction", "stolen_A1", "randn(30", "randn(20"):
        assert banned not in clause, f"clause leaked trial-specific text: {banned}"


def test_manifest_clause_is_generic():
    start = SYSTEM_PROMPT_STATIC.index(MANIFEST_CLAUSE)
    end = SYSTEM_PROMPT_STATIC.index("\n- ", start)
    clause = SYSTEM_PROMPT_STATIC[start:end]
    for banned in ("pytorch-model-cli", "cli_tool /app/weights.json"):
        assert banned not in clause, f"clause leaked trial-specific text: {banned}"


def test_cardinality_clause_states_independent_requirement():
    """The load-bearing content: cardinality is NOT a byproduct of the search."""
    start = SYSTEM_PROMPT_STATIC.index(CARDINALITY_CLAUSE)
    end = SYSTEM_PROMPT_STATIC.index("\n- ", start)
    clause = SYSTEM_PROMPT_STATIC[start:end]
    assert re.search(r"INDEPENDENT requirement", clause)
    assert re.search(r"compare counts explicitly", clause)


def test_manifest_clause_states_exact_path_requirement():
    start = SYSTEM_PROMPT_STATIC.index(MANIFEST_CLAUSE)
    end = SYSTEM_PROMPT_STATIC.index("\n- ", start)
    clause = SYSTEM_PROMPT_STATIC[start:end]
    assert re.search(r"AT THE EXACT PATH", clause)
    assert re.search(r"FileNotFoundError", clause)

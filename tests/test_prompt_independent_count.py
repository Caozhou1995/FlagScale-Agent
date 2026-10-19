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

"""Regression tests for the INDEPENDENT-COUNT requirement.

Evidence class: a cardinality clause that only says "determine the expected
size" does NOT prevent the failure it targets. Observed on a real trial:
the agent read the clause, asked "how many neurons are there? Unknown shape",
then resolved the question by CLUSTERING ITS OWN SEARCH RESULTS and counting
the clusters ("Number of clusters should be 20. Then output matrix of 20
rows"). Counting what your own search found measures recall, not the truth:
every member the search missed silently lowers the "expected" size to match
the shortfall, so the check agrees with the defect.

The load-bearing content is therefore that the count must be confirmed by an
estimator that does NOT share the search's recall, and that a disagreement
between the two estimators is evidence of missing members rather than noise.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

COUNT_CLAUSE = "Crucially, the count must be confirmed by an estimator"


def _clause() -> str:
    start = SYSTEM_PROMPT_STATIC.index(COUNT_CLAUSE)
    return SYSTEM_PROMPT_STATIC[start:SYSTEM_PROMPT_STATIC.index("\n- ", start)]


def test_independent_count_clause_present():
    assert COUNT_CLAUSE in SYSTEM_PROMPT_STATIC


def test_clause_rejects_counting_own_search_results():
    """The clause must name the failure mode: your own search's recall."""
    clause = _clause()
    assert re.search(r"does NOT share your search's recall", clause)
    assert re.search(r"how many you found", clause)


def test_clause_requires_a_second_estimator():
    clause = _clause()
    assert re.search(r"Re-derive the count a second, independent way", clause)
    assert re.search(r"bound it algebraically", clause)


def test_clause_treats_disagreement_as_evidence():
    """A mismatch must mean 'I am missing members', not 'average it away'."""
    clause = _clause()
    assert re.search(r"proof that you are missing members", clause)
    assert re.search(r"not as noise to average away", clause)


def test_independent_count_clause_is_generic():
    """No instance-specific numbers, task names, or artifacts may leak in."""
    clause = _clause()
    for banned in ("20 rows", "30 rows", "neuron", "model-extraction", "A1"):
        assert banned not in clause, f"clause leaked trial-specific text: {banned}"

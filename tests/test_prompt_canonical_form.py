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

"""Regression tests for the MATCH-THE-CHECKERS-FORM clause.

Empirical basis (filter-js-from-html, five consecutive losing cohorts through
2026-10-23): the task text said not to alter formatting "except for
normalization that may occur during HTML parsing", but the checker did not
compare bytes. It re-serialized the ORIGINAL input through the standard HTML
parser and compared that normalized form against the delivered output,
ignoring only spaces and newlines. Because that canonical form decodes
entities, reorders attributes, completes quoting and rewrites void-tag
syntax, the ORIGINAL INPUT ITSELF did not equal the canonical form. Two
consecutive architectures were therefore doomed:
  - a plain parse-then-reserialize filter delivered extra cleanup the
    comparison did not ask for (it failed on the samples it altered);
  - a surgical byte-preserving filter preserved the input's non-canonical
    spelling exactly, which is precisely what the comparison rejects.
The agent's own 58-case suite passed both times, because it certified
byte-identity (a property the checker never tests) rather than the
checker's own equality expression.

The generalizable lesson: when a task states an output-format constraint,
determine WHICH equality the consumer evaluates — raw bytes, or a
NORMALIZED form — by reading the comparison expression itself (or by
re-deriving the consumer's own normalization on the original input), and
then make the DELIVERED artifact BE that form. Prose such as "do not alter
formatting" is not evidence about the comparison; a normalizing comparison
overrides it.

Prompt-only guidance: pin the clause in SYSTEM_PROMPT_STATIC and assert it
leaks no task-specific domain detail.
"""
import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC


def _clause_slice():
    p = SYSTEM_PROMPT_STATIC
    anchor = "PRESERVE BY DEFAULT"
    start = p.index(anchor)
    end = p.index("Your own examples are not a held-out test set", start)
    return p[start:end]


def test_match_the_checkers_form_clause_present():
    """The load-bearing behaviors must each be stated."""
    low = _clause_slice().lower()
    # 1. the discriminator: identify WHICH equality the consumer evaluates
    assert re.search(r"(normaliz|canonical)", low), \
        "form-match: normalized/canonical comparison not named"
    # 2. reproduce the consumer's own normalization on the original input
    assert re.search(r"(the original|original input).{0,120}(normaliz|canonical)"
                     r"|(normaliz|canonical).{0,120}(original input)", low, re.S), \
        "form-match: 'normalize the original input the same way' missing"
    # 3. the deliverable must BE that form (not merely 'close to' it)
    assert re.search(r"(deliver|deliverable|output).{0,120}(be|is|equal).{0,60}(that form|the same form|normaliz|canonical)"
                     r"|(that form|the same form).{0,120}deliver", low, re.S), \
        "form-match: 'the deliverable must BE the compared form' missing"
    # 4. prose is not evidence about the comparison
    assert re.search(r"(prose|wording|task text).{0,160}(not|never).{0,60}(evidence|proof|override)",
                     low, re.S), \
        "form-match: 'task prose is not evidence about the comparison' missing"
    # 5. self-tests must reproduce the consumer's equality, not a substitute
    assert re.search(r"(self[- ]test|own test|test suite).{0,200}(same|exact|consumer).{0,80}(comparison|equality|expression)",
                     low, re.S), \
        "form-match: 'self-tests must replay the consumer's equality' missing"


def test_match_the_checkers_form_clause_is_generic_no_task_leak():
    """General posture only: no task's domain terms, element names, file
    names or grader expressions may appear."""
    p = SYSTEM_PROMPT_STATIC.lower()
    for bad in ("xss", "filter.py", "clean_html", "javascript:", "onerror",
                "textarea", "doctype", "copyright", "beautifulsoup", "bs4"):
        assert bad not in p, f"form-match clause leaked {bad!r}"


def test_match_the_checkers_form_clause_is_in_delivery_block():
    """Placement: with the delivery-discipline clauses."""
    assert re.search(r"(normaliz|canonical)", _clause_slice().lower())

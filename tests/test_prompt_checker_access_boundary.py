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

"""One boundary, stated consistently: what the agent may use to learn the
consumer's comparison, and what it may not.

The prompt previously said BOTH of these, contradicting itself:

  Principle 2:  "do not fetch, read, reconstruct or reverse-engineer the
                 checker ... What you DO legitimately have is the task text
                 plus the inputs it hands you."
  A later clause: "Read the comparison expression itself when you can (a test
                 file, a verification script, the harness that grades you)."

Nothing caught it, because the existing leak detector in
test_prompt_no_checker_lookup.py enumerated other phrasings. These tests pin
the ONE coherent boundary:

  PERMITTED: derive the comparison from the task text plus the inputs the
             task hands you — e.g. run the consumer's own normalization on
             the ORIGINAL input and observe how it changes.
  FORBIDDEN: reading, fetching or reconstructing the checker itself (its
             source, its test file, the grading harness, an official
             solution).

Both halves are asserted, so the clause cannot be "fixed" by deleting the
legitimate derivation route either.
"""

import re

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC

# Wording that tells the reader to go OBTAIN the hidden checker rather than
# derive the comparison from the inputs it was given.
CHECKER_ACCESS_DIRECTIVES = [
    r"read the comparison expression",
    r"read (?:the )?(?:checker|grader|harness|test file|verification script)",
    r"(?:read|open|inspect|obtain|fetch|run)\b.{0,60}\b(?:test file|"
    r"verification script|harness that grades|grading harness)",
    r"official solution",
    r"reverse-engineer the checker",
    r"a test file, a verification script",
    # siblings found by auditing every sentence that names a checker-side
    # object: each told the reader to go LOOK at the comparison rather than
    # reconstruct it from the task's own materials.
    r"read how the grader scores",
    r"read the comparison'?s loop structure",
]

# The legitimate route must survive: deriving the comparison from permitted
# inputs is exactly what Principle 2 sanctions.
DERIVATION_ROUTES = [
    r"original input.{0,200}(normaliz|canonical)"
    r"|(derivation|round-trip).{0,120}probe",
    r"task text plus the inputs",
]


_PROHIBITION_MARKERS = re.compile(
    r"\bdo not\b|\bnever\b|\bmust not\b|disqualif|hard boundary|turns? a capability",
    re.IGNORECASE,
)

# A clause GRANTS access when it uses an imperative or a permissive modal to
# point at the checker object, or names an intent to extract information from
# it. A prohibition merely NAMES the object while forbidding it ("do not fetch
# ... the checker"), which must not be flagged.
_PERMISSION_MARKERS = re.compile(
    r"\b(when you can|if you can|you may|you should|feel free|"
    r"read the|open the|inspect the|obtain the|go read)\b"
    r"|\b(?:checker|grader|harness|solution)\b.{0,40}\bto (?:learn|find out|copy|see|discover|match)\b",
    re.IGNORECASE,
)


def _hits(patterns, text):
    """Sentences that point the agent AT the checker, as opposed to sentences
    that forbid doing so. Classified per sentence, because the prohibition and
    the permission can name the same object."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not any(re.search(p, sentence, re.IGNORECASE | re.S) for p in patterns):
            continue
        if _PROHIBITION_MARKERS.search(sentence):
            continue
        if _PERMISSION_MARKERS.search(sentence):
            out.append(sentence.strip()[:160])
    return out


def test_prompt_never_instructs_reading_the_checker():
    hits = _hits(CHECKER_ACCESS_DIRECTIVES, SYSTEM_PROMPT_STATIC)
    assert hits == [], (
        "prompt contradicts Principle 2 by pointing the agent at the hidden "
        f"checker: {hits}"
    )


def test_prompt_keeps_the_permitted_derivation_route():
    """The permitted route is one route with two halves: derive from the inputs
    the task hands you (not the checker), by RUNNING the derivation probe. Both
    halves must be present — deleting the probe while leaving the source line
    would still leave the agent with no legitimate way to settle the form."""
    hits = [
        p for p in DERIVATION_ROUTES
        if re.search(p, SYSTEM_PROMPT_STATIC, re.IGNORECASE | re.S)
    ]
    missing = [p for p in DERIVATION_ROUTES if p not in hits]
    assert not missing, (
        "the permitted route is incomplete; missing half: "
        f"{missing}. The prompt must tell the agent to settle the comparison "
        "by running the derivation probe over the original input it was given."
    )


def test_boundary_is_stated_as_a_prohibition_not_only_absence():
    low = SYSTEM_PROMPT_STATIC.lower()
    assert "do not fetch" in low
    assert "transcription test" in low


def test_detectors_are_not_vacuous():
    """Negative control: re-injecting each forbidden phrasing must be caught,
    and the permitted route must not be flagged as a violation."""
    for dirty in (
        "Read the comparison expression itself when you can (a test file).",
        "read the grading harness and copy its assertion",
        "obtain the official solution script",
        "reverse-engineer the checker to learn the tolerance",
    ):
        assert _hits(CHECKER_ACCESS_DIRECTIVES, dirty) != [], (
            f"detector missed injected violation: {dirty!r}"
        )
    for clean in (
        "take the ORIGINAL input, run it through the standard parser and "
        "observe how it changes under that normalization",
        "what you legitimately have is the task text plus the inputs it hands you",
    ):
        assert _hits(CHECKER_ACCESS_DIRECTIVES, clean) == [], (
            f"detector wrongly flags permitted derivation: {clean!r}"
        )

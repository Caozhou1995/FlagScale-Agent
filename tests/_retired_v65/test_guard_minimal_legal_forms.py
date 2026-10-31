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
"""prop_c467f364: the three fixed-field MISSING messages and the merged
multi-field header must each lead with (or carry) the MINIMAL LEGAL FORM of
the missing declaration, so a first-offense step_done can be fixed by
copy-paste instead of a full-message read; and the system prompt must inline
the same three shapes (brace-escaped) so the first attempt is correct.

Anti-decay: each test asserts the copy-paste line VERBATIM — softening the
message (removing the minimal form) or re-breaking the prompt braces fails
these tests.
"""

import pytest

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC
from flagscale_agent.react.guard.verification import (
    _CONVENTION_MISSING,
    _DELIVERABLE_READING_MISSING,
    _UNVERIFIED_ASSUMPTIONS_MISSING,
    _collect_fixed_field_failures,
)

MIN_FORMS = (
    'convention={"name": "none"}',
    'deliverable_reading={"deliverable": "none"}',
    'assumptions={"unverified": "none"}',
)


def test_convention_missing_message_leads_with_minimal_form():
    head = _CONVENTION_MISSING[:400]
    assert 'MINIMAL LEGAL FORM' in head
    assert MIN_FORMS[0] in head


def test_assumptions_missing_message_leads_with_minimal_form():
    head = _UNVERIFIED_ASSUMPTIONS_MISSING[:400]
    assert 'MINIMAL LEGAL FORM' in head
    assert MIN_FORMS[2] in head


def test_deliverable_missing_message_leads_with_minimal_form():
    head = _DELIVERABLE_READING_MISSING[:400]
    assert 'MINIMAL LEGAL FORM' in head
    assert MIN_FORMS[1] in head


def test_merged_multi_field_header_carries_all_three_quick_forms():
    # Simulate a three-field-missing step_done through the real merge helper.
    from flagscale_agent.react.guard.verification import (
        _check_convention_field,
        _check_deliverable_reading_field,
        _check_unverified_assumptions_field,
    )
    checks = [
        _check_convention_field(None),
        _check_deliverable_reading_field(None),
        _check_unverified_assumptions_field(None),
    ]
    merged = _collect_fixed_field_failures(checks)
    assert merged is not None
    for frag in MIN_FORMS:
        assert frag in merged.message, f"merged header missing quick form: {frag}"


def test_system_prompt_renders_all_three_minimal_forms():
    # Braces must be escaped ({{ }}) so .format() leaves the literal form.
    core = SYSTEM_PROMPT_STATIC.format(
        cwd="/x", tools="t", skills="s", knowledge="k"
    )
    for frag in MIN_FORMS:
        assert frag in core, f"rendered prompt missing minimal form: {frag}"


def test_system_prompt_static_uses_escaped_braces_not_bare():
    # The SOURCE text must carry escaped braces; a bare { would break .format()
    # for every future field added to the template.
    assert 'convention={{"name": "none"}}' in SYSTEM_PROMPT_STATIC
    assert 'deliverable_reading={{"deliverable": "none"}}' in SYSTEM_PROMPT_STATIC
    assert 'assumptions={{"unverified": "none"}}' in SYSTEM_PROMPT_STATIC

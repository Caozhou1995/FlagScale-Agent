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

"""Tests: a JSON-encoded object passed as a string must be named as such.

Live incident this pins: the agent passed a structured argument (the
`deliverable_reading` object) as a JSON STRING. The guard's type error was
technically correct ("expected object, got str") but the value looked right in a
log, so the agent simply re-sent it. The message must say the content parsed
fine and that the ENCODING is the problem.
"""

import json

from flagscale_agent.react.guard.arg_type import ArgTypeGuard, _try_decode_container
from flagscale_agent.react.guard import GuardContext


class _FakeTool:
    def __init__(self, props):
        self.parameters = {"properties": props}


class _FakeRegistry:
    def __init__(self, props):
        self._tool = _FakeTool(props)

    def get(self, name):
        return self._tool


def _ctx(tool_name, args):
    return GuardContext(tool_name=tool_name, tool_args=args)


SCHEMA = {"deliverable_reading": {"type": "object"},
          "plan": {"type": "array"},
          "n": {"type": "integer"}}


class TestJsonEncodedContainerDiagnosis:
    def test_json_encoded_object_is_named_as_an_encoding_problem(self):
        encoded = json.dumps({"deliverable": "/tmp/x.md", "readings": []})
        guard = ArgTypeGuard(_FakeRegistry(SCHEMA))
        v = guard.check_pre(_ctx("plan_update", {"deliverable_reading": encoded}))
        assert v is not None
        assert "JSON-encoded" in v.message
        assert "ENCODING" in v.message

    def test_json_encoded_array_is_named_too(self):
        guard = ArgTypeGuard(_FakeRegistry(SCHEMA))
        v = guard.check_pre(_ctx("plan_update", {"plan": '[{"a": 1}]'}))
        assert v is not None
        assert "JSON-encoded" in v.message

    def test_plain_string_mismatch_gets_no_encoding_advice(self):
        guard = ArgTypeGuard(_FakeRegistry(SCHEMA))
        v = guard.check_pre(_ctx("plan_update",
                                 {"deliverable_reading": "not json at all"}))
        assert v is not None
        assert "JSON-encoded" not in v.message

    def test_json_to_wrong_type_gets_no_encoding_advice(self):
        guard = ArgTypeGuard(_FakeRegistry(SCHEMA))
        v = guard.check_pre(_ctx("plan_update", {"deliverable_reading": "123"}))
        assert v is not None
        assert "JSON-encoded" not in v.message

    def test_message_still_reports_field_expected_and_actual(self):
        guard = ArgTypeGuard(_FakeRegistry(SCHEMA))
        v = guard.check_pre(_ctx("plan_update", {"n": "5"}))
        assert v is not None
        assert "'n'" in v.message
        assert "expected integer" in v.message
        assert "got str" in v.message

    def test_correct_types_are_not_blocked(self):
        guard = ArgTypeGuard(_FakeRegistry(SCHEMA))
        v = guard.check_pre(_ctx("plan_update",
                                 {"deliverable_reading": {"a": 1},
                                  "plan": [1, 2], "n": 3}))
        assert v is None


class TestTryDecodeContainer:
    def test_decodes_object(self):
        assert _try_decode_container('{"a": 1}', "object") == {"a": 1}

    def test_decodes_array(self):
        assert _try_decode_container("[1, 2]", "array") == [1, 2]

    def test_rejects_non_json(self):
        assert _try_decode_container("plain words", "object") is None

    def test_rejects_wrong_container_type(self):
        assert _try_decode_container("[1, 2]", "object") is None
        assert _try_decode_container('{"a": 1}', "array") is None

    def test_empty_string_is_none(self):
        assert _try_decode_container("", "object") is None

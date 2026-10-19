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

"""Regression tests: every SURFACED block/escalate names its failing sub-rule.

Root cause: a block message is a template enumerating EVERY sub-rule a
structured declaration could violate; the specific failing check lives only in
verdict.reason (used internally for override matching). The agent could not see
it and re-fired the same declaration in a guess-retry loop until its budget ran
out (observed: extract-elf retry27, 9 consecutive verification blocks).

Fix: GuardRegistry._resolve appends a diagnostic naming verdict.reason to every
surfaced block/escalate. Generic — the slug describes the agent's own
declaration, never the task's answer.
"""

from flagscale_agent.react.guard import (
    Guard, GuardVerdict, GuardContext, GuardRegistry,
)


class _StructuredBlock(Guard):
    name = "structured"
    priority = 50

    def check_pre(self, ctx):
        return GuardVerdict.block(
            "[structured] declaration malformed (lists every sub-rule)",
            reason="hedge_direction_evidence_not_distinct",
            category="verification_required",
            overridable=False,
        )


class _OverridableBlock(Guard):
    name = "soft"
    priority = 50

    def check_pre(self, ctx):
        return GuardVerdict.block(
            "[soft] malformed", reason="deliverable_reading_malformed",
            category="soft",
        )


class _EscalateBlock(Guard):
    name = "stop"
    priority = 50

    def check_pre(self, ctx):
        return GuardVerdict.escalate(
            "[stop] dead end", reason="single_shot_completion_without_plan",
            category="stop",
        )


def _ctx(override=None):
    return GuardContext(tool_name="shell", tool_args={"command": "x"},
                        override_reason=override)


def _resolve(guard):
    reg = GuardRegistry()
    reg.register(guard)
    return reg._resolve(_ctx(), "pre")


class TestReasonDiagnostic:
    def test_non_overridable_block_names_reason(self):
        v = _resolve(_StructuredBlock())
        assert v is not None
        assert "hedge_direction_evidence_not_distinct" in v.message
        assert "failing check" in v.message

    def test_overridable_block_names_reason(self):
        v = _resolve(_OverridableBlock())
        assert "deliverable_reading_malformed" in v.message

    def test_escalate_names_reason(self):
        v = _resolve(_EscalateBlock())
        assert "single_shot_completion_without_plan" in v.message

    def test_reason_slug_present_alongside_owner_tag(self):
        v = _resolve(_StructuredBlock())
        # owner tag (which guard) AND reason tag (which sub-rule) both present
        assert "[blocked by guard: structured]" in v.message
        assert "hedge_direction_evidence_not_distinct" in v.message

    def test_reason_is_generic_no_grader_leak(self):
        # The diagnostic must not name any task/answer; it echoes only the slug.
        v = _resolve(_StructuredBlock())
        tail = v.message.split("failing check", 1)[1]
        assert "hedge_direction_evidence_not_distinct" in tail

    def test_inject_carries_no_reason_tag(self):
        # Non-block paths must NOT gain the tag (only surfaced blocks do).
        class _Inject(Guard):
            name = "adv"
            priority = 50

            def check_post(self, ctx):
                return GuardVerdict.inject("[adv] note", reason="some_reason",
                                           category="adv")

        reg = GuardRegistry()
        reg.register(_Inject())
        v = reg._resolve(_ctx(), "post")
        assert v is not None and v.action == "inject"
        assert "failing check" not in v.message

    def test_diagnostic_makes_no_overclaim(self):
        # M14: every failing check evaluated at a decision point is listed in
        # the merged message, so the diagnostic must not claim later sub-rules
        # "already passed" nor promise field-level precision (a slug can span
        # many sub-check sites).
        v = _resolve(_StructuredBlock())
        assert "already passed" not in v.message
        assert "listed above" in v.message
        assert "CATEGORY" in v.message

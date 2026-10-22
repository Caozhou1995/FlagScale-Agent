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

"""prop_78aa241f — the 75% cold-consumer override is EVIDENCE-bound, not prose.

The 75% gate's block message demands the agent paste the exact command it ran
against the delivery artifact and the output it produced. A reason that merely
SAYS "I ran it" (prose) must be REJECTED when the trial's own recorded action
lines contain no such command — that was the observed escape (133/134 losing
trials overrode with prose). A reason quoting a command that genuinely appears
in the recorded actions is accepted.
"""

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.first_e2e_run import FirstE2eRunGuard


def make_ctx(tool_name="shell", override_reason="", messages=None):
    ctx = GuardContext()
    ctx.tool_name = tool_name
    ctx.override_reason = override_reason
    if messages is not None:
        ctx.messages = messages
    return ctx


class _Stats:
    def __init__(self):
        self.pct = 0.0

    def __call__(self):
        if self.pct is None:
            return None
        budget = 1200.0
        elapsed = budget * self.pct / 100.0
        return {
            "elapsed": elapsed,
            "budget": budget,
            "remaining": budget - elapsed,
            "pct": self.pct,
        }


_RECORDED = (
    "[{'name': 'shell', 'result': '$ python3 -m pytest tests/test_x.py -q\\n"
    "8 passed in 2.14s'}]"
)
_REASON_EVIDENCE = (
    "Ran the consumer's command against the delivered artifact: "
    "`python3 -m pytest tests/test_x.py -q` -> 8 passed in 2.14s"
)
_REASON_PROSE = (
    "I ran the end-to-end path cold and read the output myself; the "
    "deliverable is verified and this checkpoint can release."
)
_REASON_NONE_APPLY = (
    "none apply — this deliverable has no end-to-end path; the artifact is a "
    "data file consumed directly by the verifier."
)
_PROSE_PASTE = (
    "I ran it: the command and its output showed everything works. "
    "The cold consumer replay produced expected results."
)


def _messages_with_recording():
    return [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": "let me run it"},
        {"role": "user", "content": "tool_result: " + _RECORDED},
        {"role": "assistant", "content": _REASON_EVIDENCE},
    ]


class TestOverrideEvidenceBinding:
    def _gate_at_75(self, messages):
        """Real flow: pct>=75 fires the 75% block (which sets the pending
        threshold), THEN the registry calls accept_override with the reason."""
        s = _Stats()
        s.pct = 76.0
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx(messages=messages))
        assert v is not None and v.reason == "first_e2e_run_75pct"
        return g

    def test_prose_only_override_rejected(self):
        g = self._gate_at_75(_messages_with_recording())
        assert g.accept_override(_REASON_PROSE, make_ctx(
            override_reason=_REASON_PROSE,
            messages=_messages_with_recording(),
        )) is False

    def test_evidence_override_accepted(self):
        g = self._gate_at_75(_messages_with_recording())
        assert g.accept_override(_REASON_EVIDENCE, make_ctx(
            override_reason=_REASON_EVIDENCE,
            messages=_messages_with_recording(),
        )) is True

    def test_none_apply_still_allowed(self):
        # (b) "none apply" stays a legal release: no e2e path exists.
        g = self._gate_at_75(_messages_with_recording())
        assert g.accept_override(_REASON_NONE_APPLY, make_ctx(
            override_reason=_REASON_NONE_APPLY,
            messages=_messages_with_recording(),
        )) is True

    def test_command_like_prose_without_recording_rejected(self):
        # The mutation-gate variant: a reason LOOKING like evidence but whose
        # command never appears in the recorded actions must not pass.
        g = self._gate_at_75(_messages_with_recording())
        assert g.accept_override(_PROSE_PASTE, make_ctx(
            override_reason=_PROSE_PASTE,
            messages=_messages_with_recording(),
        )) is False

    def test_backticked_command_not_in_recording_rejected(self):
        # A FABRICATED command in backticks: command-shaped, but never ran —
        # the membership test against the trial's own recorded text must
        # reject it. (Without this, dropping the membership test would pass.)
        fake = (
            "Verified: `python3 -m pytest tests/never_ran.py -q` -> all green. "
            "The artifact at its delivery path reproduces."
        )
        g = self._gate_at_75(_messages_with_recording())
        assert g.accept_override(fake, make_ctx(
            override_reason=fake,
            messages=_messages_with_recording(),
        )) is False

    def test_rejected_override_rearms_75(self):
        # Rejection must not consume the checkpoint: the SAME guard must
        # re-block on its next check_pre, otherwise one rejected prose reason
        # releases forever (the fired set was already marked when the block
        # first surfaced).
        s = _Stats()
        s.pct = 76.0
        g = FirstE2eRunGuard(stats_fn=s)
        v = g.check_pre(make_ctx(messages=_messages_with_recording()))
        assert v is not None and v.reason == "first_e2e_run_75pct"
        assert g.accept_override(_REASON_PROSE, make_ctx(
            override_reason=_REASON_PROSE,
            messages=_messages_with_recording(),
        )) is False
        # Same guard, still >=75%: the checkpoint must have been re-armed so
        # the gate blocks again instead of staying silent forever.
        v2 = g.check_pre(make_ctx(messages=_messages_with_recording()))
        assert v2 is not None and v2.reason == "first_e2e_run_75pct"

    def test_accepted_override_consumes_checkpoint(self):
        # On SUCCESS the checkpoint stays spent (no re-block): the agent pasted
        # evidence and the gate released.
        g = self._gate_at_75(_messages_with_recording())
        assert g.accept_override(_REASON_EVIDENCE, make_ctx(
            override_reason=_REASON_EVIDENCE,
            messages=_messages_with_recording(),
        )) is True
        assert 75 in g._fired

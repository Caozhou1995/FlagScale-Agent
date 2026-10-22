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

"""Byte-preservation evidence binding on the 75% cold-consumer checkpoint.

When the CURRENT task's instruction text itself states a preservation
requirement (the wording must contain the task's OWN preservation language —
never a task-name list), the 75% release must carry IDENTITY evidence for the
files the instruction asks to keep unchanged: a cmp / md5sum / sha256sum-style
command whose output appears in the trial's own recorded actions.

Red line: the keyword detection reads the task text at RUNTIME — there is no
task-name table anywhere in this module. A task whose instruction never
mentions preservation is never asked for identity evidence (zero false
rejects on non-preservation tasks).
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


_TASK_PRESERVE = (
    "Edit the config files in /app/conf as requested. The backup files in "
    "/app/ref must remain unchanged. Save the result to /app/out.conf."
)
_TASK_PLAIN = (
    "Write a python file /app/solve.py that reads stdin and prints the "
    "sorted lines. The output must be correct for all inputs."
)

_REC_EVIDENCE = (
    "[{'name': 'shell', 'result': '$ md5sum /app/ref/base.conf\\n"
    "3f2a...  /app/ref/base.conf\\n"
    "$ cmp /app/ref/base.conf /app/out_ref_copy.conf && echo IDENTICAL\\n"
    "IDENTICAL'}]"
)
_REASON_MD5 = (
    "Identity verified before release: `md5sum /app/ref/base.conf` -> "
    "3f2a... and `cmp /app/ref/base.conf /app/out_ref_copy.conf` -> "
    "IDENTICAL. The preservation requirement holds on disk."
)
_REASON_PROSE_PRESERVE = (
    "I preserved the reference files untouched; nothing was modified and the "
    "task is complete, so this checkpoint can release."
)


def _messages(task_text, recorded=None):
    msgs = [{"role": "user", "content": task_text}]
    if recorded:
        msgs.append({"role": "user", "content": "tool_result: " + recorded})
    return msgs


def _gate_at_75(task_text, recorded=None):
    s = _Stats()
    s.pct = 76.0
    g = FirstE2eRunGuard(stats_fn=s, task_text_fn=lambda: task_text)
    v = g.check_pre(make_ctx(messages=_messages(task_text, recorded)))
    assert v is not None and v.reason == "first_e2e_run_75pct"
    return g


class TestPreservationEvidenceBinding:
    def test_prose_rejected_when_task_demands_preservation(self):
        # The task text contains preservation language; the override reason
        # only CLAIMS preservation with no identity command -> reject + re-arm.
        g = _gate_at_75(_TASK_PRESERVE, _REC_EVIDENCE)
        assert g.accept_override(_REASON_PROSE_PRESERVE, make_ctx(
            override_reason=_REASON_PROSE_PRESERVE,
            messages=_messages(_TASK_PRESERVE, _REC_EVIDENCE),
        )) is False

    def test_md5_evidence_accepted(self):
        # A cmp/md5 command quoted in the reason AND present in the recorded
        # actions satisfies the identity-evidence binding.
        g = _gate_at_75(_TASK_PRESERVE, _REC_EVIDENCE)
        assert g.accept_override(_REASON_MD5, make_ctx(
            override_reason=_REASON_MD5,
            messages=_messages(_TASK_PRESERVE, _REC_EVIDENCE),
        )) is True

    def test_plain_task_never_demands_identity_evidence(self):
        # Red line: a task whose instruction has NO preservation language is
        # never asked for identity evidence — the existing command-evidence
        # rule alone governs (a plausible run command suffices).
        rec = ("[{'name': 'shell', 'result': '$ python3 /app/solve.py < in.txt\\n"
               "sorted ok'}]")
        reason = "Ran it: `python3 /app/solve.py < in.txt` -> sorted ok."
        g = _gate_at_75(_TASK_PLAIN, rec)
        assert g.accept_override(reason, make_ctx(
            override_reason=reason,
            messages=_messages(_TASK_PLAIN, rec),
        )) is True

    def test_preservation_task_without_any_recording_rejected(self):
        # Even an honest-looking md5 claim cannot pass when the trial never
        # recorded such a command: membership in recorded actions still binds.
        g = _gate_at_75(_TASK_PRESERVE, None)
        assert g.accept_override(_REASON_MD5, make_ctx(
            override_reason=_REASON_MD5,
            messages=_messages(_TASK_PRESERVE, None),
        )) is False

    def test_rejected_preservation_override_rearms(self):
        g = _gate_at_75(_TASK_PRESERVE, _REC_EVIDENCE)
        assert g.accept_override(_REASON_PROSE_PRESERVE, make_ctx(
            override_reason=_REASON_PROSE_PRESERVE,
            messages=_messages(_TASK_PRESERVE, _REC_EVIDENCE),
        )) is False
        v2 = g.check_pre(make_ctx(messages=_messages(_TASK_PRESERVE, _REC_EVIDENCE)))
        assert v2 is not None and v2.reason == "first_e2e_run_75pct"

    def test_pipeline_command_alone_insufficient_for_preservation_task(self):
        # The mutation-gate discriminator: the reason quotes a REAL recorded
        # command (the pipeline ran), but on a preservation task that command
        # is not IDENTITY evidence — the gate must still reject it.
        rec = ("[{'name': 'shell', 'result': '$ python3 /app/tune.py\\n"
               "tuned ok, mse 0.01'}]")
        reason = "Ran it: `python3 /app/tune.py` -> tuned ok, mse 0.01."
        g = _gate_at_75(_TASK_PRESERVE, rec)
        assert g.accept_override(reason, make_ctx(
            override_reason=reason,
            messages=_messages(_TASK_PRESERVE, rec),
        )) is False

    def test_no_task_text_falls_back_to_base_rule(self):
        # task_text_fn absent/empty -> no preservation demand; base rule only.
        rec = ("[{'name': 'shell', 'result': '$ python3 -m pytest tests/test_x.py -q\\n"
               "8 passed in 2.14s'}]")
        reason = "Ran: `python3 -m pytest tests/test_x.py -q` -> 8 passed in 2.14s"
        g = _gate_at_75("", rec)
        assert g.accept_override(reason, make_ctx(
            override_reason=reason,
            messages=_messages("", rec),
        )) is True

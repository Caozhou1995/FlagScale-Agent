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

"""Reasoning dump written immediately BEFORE a post-edit far-end inject.

The far-end block demands the agent "point at the turn where the decision was
made". That demand was unperformable: reasoning is not part of the conversation
the agent can re-read. These tests pin the two halves of the fix:

  (1) the harness WRITES the session's reasoning to <session_dir>/thinking_log.md
      at inject time — before the reminder is emitted, not after;
  (2) the emitted inject TEXT carries that exact path, so the agent performs a
      bounded read instead of an impossible recall.

A third group pins the safety property the plan forbids violating: with no
session dir, no reasoning, or a failing write, the inject is STILL emitted with
its ordinary lines — the dump is an enhancement, never a gate.
"""

import os

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.post_edit_far_end import (
    THINKING_DUMP_FILENAME,
    PostEditFarEndGuard,
    _iter_history_thinking,
    _write_thinking_dump,
)


def _thinking_msg(text, sig="s"):
    return {"role": "assistant",
            "content": [{"type": "thinking", "thinking": text, "signature": sig}]}


def _ctx(path, messages=None, thinking_text="", result=None):
    return GuardContext(
        tool_name="edit_file",
        tool_args={"path": path},
        tool_result=result or f"Successfully edited {path}",
        messages=messages or [],
        thinking_text=thinking_text,
    )


class TestIterHistoryThinking:
    def test_anthropic_block_shape(self):
        msgs = [_thinking_msg("A-decided"), _thinking_msg("B-decided")]
        assert _iter_history_thinking(msgs) == ["A-decided", "B-decided"]

    def test_openai_flat_key_shape(self):
        # The openai provider stores a flat `thinking` key instead of blocks.
        assert _iter_history_thinking([{"role": "assistant", "thinking": "flat"}]) == ["flat"]

    def test_plain_text_is_not_reasoning(self):
        msgs = [{"role": "assistant",
                 "content": [{"type": "text", "text": "just an answer"}]}]
        assert _iter_history_thinking(msgs) == []

    def test_empty_thinking_blocks_are_skipped(self):
        assert _iter_history_thinking([_thinking_msg("   ")]) == []


class TestDumpFileContents:
    def test_dump_path_and_body(self, tmp_path):
        sess = str(tmp_path / "sess")
        ctx = _ctx("a.py", messages=[_thinking_msg("DECISION-A: three slots")])
        path = _write_thinking_dump(sess, ctx, "a.py")
        assert path == os.path.join(sess, THINKING_DUMP_FILENAME)
        body = open(path, encoding="utf-8").read()
        assert "DECISION-A: three slots" in body
        assert "a.py" in body  # names the file whose edit triggered the dump

    def test_current_iteration_appended_when_absent(self, tmp_path):
        ctx = _ctx("a.py", messages=[], thinking_text="DECISION-C: cap the head")
        path = _write_thinking_dump(str(tmp_path / "s"), ctx, "a.py")
        assert "DECISION-C: cap the head" in open(path, encoding="utf-8").read()

    def test_current_iteration_not_duplicated(self, tmp_path):
        text = "DECISION-A: once only"
        ctx = _ctx("a.py", messages=[_thinking_msg(text)], thinking_text=text)
        path = _write_thinking_dump(str(tmp_path / "s"), ctx, "a.py")
        body = open(path, encoding="utf-8").read()
        assert body.count(text) == 1, "the live iteration must not be written twice"

    def test_no_reasoning_writes_nothing(self, tmp_path):
        ctx = _ctx("a.py", messages=[], thinking_text="")
        assert _write_thinking_dump(str(tmp_path / "s"), ctx, "a.py") is None

    def test_no_session_dir_writes_nothing(self):
        ctx = _ctx("a.py", messages=[_thinking_msg("A")])
        assert _write_thinking_dump("", ctx, "a.py") is None

    def test_every_entry_is_kept(self, tmp_path):
        # No entry is dropped. A newest-N or "first N + last" window discards
        # exactly the class of block the far-end reminder asks about.
        n = 200
        msgs = [_thinking_msg(f"R{i}") for i in range(n)]
        path = _write_thinking_dump(str(tmp_path / "s"), _ctx("a.py", messages=msgs), "a.py")
        body = open(path, encoding="utf-8").read()
        for i in (0, 1, 99, 100, n - 2, n - 1):
            assert f"R{i}\n" in body, f"entry R{i} was dropped"
        # structural markers only — the header prose legitimately contains the
        # words "truncated"/"dropped" while stating that nothing is.
        assert "## reasoning" in body
        assert not [ln for ln in body.splitlines() if ln.startswith("- OMITTED")]
        assert not [ln for ln in body.splitlines()
                    if ln.startswith("## reasoning") and "truncated" in ln]
        assert f"## reasoning {n - 1}\n" in body, "the last entry must be present"

    def test_each_entry_is_written_in_full(self, tmp_path):
        # The decision sits at the END of a long reasoning block (measured on
        # real sessions), so a head-only slice destroys the very thing this file
        # exists for. The whole block must be present, tail included.
        filler = "x" * 20000
        long_text = "HEAD" + filler + "DECISION_AT_THE_END"
        path = _write_thinking_dump(str(tmp_path / "s"),
                                   _ctx("a.py", messages=[_thinking_msg(long_text)]), "a.py")
        body = open(path, encoding="utf-8").read()
        assert long_text in body, "the entry body was cut somewhere"
        assert "[truncated]" not in body and "byte ceiling" not in body

    def test_no_byte_ceiling_on_a_huge_session(self, tmp_path):
        # Storage is cheap: a pathological session must not hit a file ceiling.
        block = "y" * 5000
        msgs = [_thinking_msg(block) for _ in range(150)]  # ~750 KB of reasoning
        path = _write_thinking_dump(str(tmp_path / "s"), _ctx("a.py", messages=msgs), "a.py")
        body = open(path, encoding="utf-8").read()
        assert "byte ceiling" not in body
        assert body.count(block) == 150, "some entries were cut by a size ceiling"

    def test_header_points_at_thinking_search(self, tmp_path):
        # The file can be large, so the header must name the retrieval tool
        # instead of telling the reader to read the whole thing.
        path = _write_thinking_dump(str(tmp_path / "s"),
                                   _ctx("a.py", messages=[_thinking_msg("A")]), "a.py")
        body = open(path, encoding="utf-8").read()
        assert "thinking_search" in body
        assert "IN FULL" in body


class TestInjectCarriesThePath:
    def test_path_branch_inject_names_the_dump(self, tmp_path):
        sess = str(tmp_path / "sess")
        guard = PostEditFarEndGuard(session_dir_fn=lambda: sess)
        v = guard.check_post(_ctx("a.py", messages=[_thinking_msg("DECISION-A: 3 slots")]))
        assert v is not None and v.action == "inject"
        expected = os.path.join(sess, THINKING_DUMP_FILENAME)
        assert expected in v.message, "the inject must carry the exact dump path"
        # and the file must ALREADY exist when the inject is emitted
        assert os.path.exists(expected)

    def test_pointer_precedes_the_far_end_block(self, tmp_path):
        # Order matters: the demand to "go back further" must come AFTER the
        # pointer that makes the demand performable.
        guard = PostEditFarEndGuard(session_dir_fn=lambda: str(tmp_path / "s"))
        v = guard.check_post(_ctx("a.py", messages=[_thinking_msg("D")]))
        body = v.message
        assert body.index("YOUR REASONING, ON DISK") < body.index("THINKING↔BYTES DIFF")

    def test_effect_branch_inject_also_names_the_dump(self, tmp_path):
        # The shell-heredoc branch must not silently miss what the path branch
        # delivers (the write SHAPE does not change the reminders' truth).
        sess = str(tmp_path / "sess")
        guard = PostEditFarEndGuard(session_dir_fn=lambda: sess)
        ctx = _ctx("", messages=[_thinking_msg("DECISION-A")])
        ctx.tool_args = {}
        lines = guard._reasoning_dump_lines(ctx, "out.txt")
        assert any(os.path.join(sess, THINKING_DUMP_FILENAME) in ln for ln in lines)


class TestDegradesWithoutBreakingTheReminder:
    def test_no_session_dir_fn_still_injects(self):
        guard = PostEditFarEndGuard()  # no session_dir_fn at all
        v = guard.check_post(_ctx("a.py", messages=[_thinking_msg("A")]))
        assert v is not None and v.action == "inject"
        assert "THINKING↔BYTES DIFF" in v.message
        assert "YOUR REASONING, ON DISK" not in v.message

    def test_session_dir_fn_raising_still_injects(self):
        def boom():
            raise RuntimeError("no session")
        guard = PostEditFarEndGuard(session_dir_fn=boom)
        v = guard.check_post(_ctx("a.py", messages=[_thinking_msg("A")]))
        assert v is not None and v.action == "inject"
        assert "THINKING↔BYTES DIFF" in v.message

    def test_unwritable_session_dir_still_injects(self, tmp_path):
        # A path under a regular FILE can never be a directory -> the write
        # fails; the reminder must survive intact.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a dir")
        guard = PostEditFarEndGuard(session_dir_fn=lambda: str(blocker / "sub"))
        v = guard.check_post(_ctx("a.py", messages=[_thinking_msg("A")]))
        assert v is not None and v.action == "inject"
        assert "THINKING↔BYTES DIFF" in v.message

    def test_never_blocks(self, tmp_path):
        guard = PostEditFarEndGuard(session_dir_fn=lambda: str(tmp_path / "s"))
        v = guard.check_post(_ctx("a.py", messages=[_thinking_msg("A")]))
        assert v.action == "inject"  # never block/escalate

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

"""Tests for FirstArtifactGuard.

The guard exists because a measured failure shape (a whole run spent in design
reasoning, zero writes, verifier read "no such file") was invisible to every
existing text-only nudge. These tests pin both directions:

  * it BLOCKS the read-only forever-agent once the turn budget is spent
  * it RELEASES an agent that writes (so it can never livelock a healthy run)
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.first_artifact import (
    FirstArtifactGuard,
    _shell_writes_bytes,
)


def _ctx(tool_name="shell", tool_args=None, messages=None):
    return GuardContext(
        tool_name=tool_name,
        tool_args=tool_args or {},
        messages=messages or [],
    )


def _turns(n: int, command: str = "cat /app/check.py"):
    """n assistant turns, each reading something and writing nothing."""
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "shell", "input": {"command": command}},
            ],
        }
        for _ in range(n)
    ]


def _history_with(tool: str, args: dict):
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": tool, "input": args},
            ],
        }
    ]


# ── the failure shape: read-only forever ─────────────────────────────────


def test_blocks_read_only_shell_after_turn_budget():
    """The measured failure: N+ turns of `ls`/`cat`, no write. Must block."""
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_name="shell",
        tool_args={"command": "cd /app && ls -la && cat check.py"},
        messages=_turns(FirstArtifactGuard.TURN_BUDGET),
    )
    verdict = guard.check_pre(ctx)
    assert verdict is not None, "a design-only run past the budget must be blocked"
    assert verdict.action == "block"
    assert verdict.reason == "no_artifact_write_yet"


def test_does_not_block_before_budget():
    """Orientation is legitimate — the guard must not fire on turn 1."""
    guard = FirstArtifactGuard()
    ctx = _ctx(tool_args={"command": "ls -la /app"}, messages=_turns(1))
    assert guard.check_pre(ctx) is None


def test_does_not_block_non_artifact_tools():
    """read_file/plan_update are not delivery-surface mutations."""
    guard = FirstArtifactGuard()
    ctx = _ctx(tool_name="read_file", messages=_turns(99))
    assert guard.check_pre(ctx) is None


# ── regression: ctx.turn_count is always 0 ───────────────────────────────


def test_turn_count_field_is_not_used_as_the_clock():
    """REGRESSION: the budget must come from history, not ctx.turn_count.

    kernel.py builds GuardContext.turn_count from
    `getattr(d.config, "_turn_count", 0)`, but nothing in the package ever
    assigns that attribute (the only `_turn_count` belongs to PromptBuilder).
    A guard that trusted it could never fire. This test drives exactly that
    world: turn_count stays 0 while the history clearly holds enough turns.
    """
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_args={"command": "ls -la /app"},
        messages=_turns(FirstArtifactGuard.TURN_BUDGET + 3),
    )
    assert ctx.turn_count == 0, "precondition: the context field really is 0"
    verdict = guard.check_pre(ctx)
    assert verdict is not None, (
        "guard must count turns from history; relying on ctx.turn_count (always "
        "0) makes it unreachable"
    )


def test_byte_budget_fires_on_a_single_huge_turn():
    """REGRESSION: the target failure shape makes very FEW turns.

    Measured: the whole failing run produced 7 thinking turns and one tool
    call across 540KB. A turn-only budget can therefore fire too late — one
    turn can carry tens of KB of reasoning before its single tool call. The
    byte trigger must catch a run that is still on turn 1 with nothing on disk.
    """
    guard = FirstArtifactGuard()
    huge = "x" * (FirstArtifactGuard.REASONING_BYTES_BUDGET + 1)
    msgs = [{"role": "assistant", "content": huge}]
    ctx = _ctx(tool_args={"command": "ls -la /app"}, messages=msgs)
    verdict = guard.check_pre(ctx)
    assert verdict is not None, "a single huge reasoning turn must trigger the guard"
    assert verdict.reason == "no_artifact_write_yet"


def test_byte_budget_not_reached_early():
    """A short orientation turn must not trigger the byte trigger."""
    guard = FirstArtifactGuard()
    msgs = [{"role": "assistant", "content": "let me look around" }]
    ctx = _ctx(tool_args={"command": "ls -la /app"}, messages=msgs)
    assert guard.check_pre(ctx) is None


# ── the healthy shape: must never livelock ────────────────────────────────


def test_write_file_through_after_budget():
    """A write_file IS the fix, so it must pass even past the budget."""
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_name="write_file",
        tool_args={"path": "/app/re.json", "content": "[]"},
        messages=_turns(99),
    )
    assert guard.check_pre(ctx) is None


def test_shell_write_through_after_budget():
    """A heredoc write is how most deliverables are actually produced."""
    guard = FirstArtifactGuard()
    ctx = _ctx(
        tool_args={"command": "cat > /app/re.json <<'EOF'\n[]\nEOF"},
        messages=_turns(99),
    )
    assert guard.check_pre(ctx) is None


def test_earlier_write_disarms_guard():
    """A write in an EARLIER turn must disarm — history is authoritative."""
    guard = FirstArtifactGuard()
    msgs = _turns(99) + _history_with("write_file", {"path": "/app/re.json"})
    ctx = _ctx(tool_args={"command": "ls -la /app"}, messages=msgs)
    assert guard.check_pre(ctx) is None


def test_openai_style_history_write_disarms_guard():
    """REGRESSION: this harness emits OpenAI-style msg["tool_calls"].

    Scanning only Anthropic-style content blocks found nothing here, so the
    disarm check silently failed and the guard could block a run that had
    already written its artifact.
    """
    guard = FirstArtifactGuard()
    msgs = _turns(99) + [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "write_file", "arguments": '{"path": "/app/re.json"}'}}
            ],
        }
    ]
    ctx = _ctx(tool_args={"command": "ls -la /app"}, messages=msgs)
    assert guard.check_pre(ctx) is None


def test_openai_style_shell_write_disarms_guard():
    guard = FirstArtifactGuard()
    msgs = _turns(99) + [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "shell", "arguments": '{"command": "cat > /app/re.json <<EOF"}'}}
            ],
        }
    ]
    ctx = _ctx(tool_args={"command": "ls -la /app"}, messages=msgs)
    assert guard.check_pre(ctx) is None


def test_override_disarms_for_session():
    """Overriding must stop the guard firing again (no livelock)."""
    guard = FirstArtifactGuard()
    ctx = _ctx(tool_args={"command": "cat /app/spec.md"}, messages=_turns(99))
    verdict = guard.check_pre(ctx)
    assert verdict is not None
    assert guard.accept_override("I must read the spec format first", ctx) is True
    assert guard.check_pre(ctx) is None


# ── the shell-write classifier ───────────────────────────────────────────


@pytest.mark.parametrize(
    "command",
    [
        "ls -la /app",
        "cat /app/check.py",
        "grep -rn foo /app",
        "cd /app && ls -la && cat check.py",
        "ls -la /app 2>&1",
        "cat /app/x >/dev/null",
        # REGRESSION: this exact command disarmed the guard in a real run.
        # Pure introspection, no file created, yet inline python used to be a
        # write marker by itself.
        'ls -la /app; echo ---; cat /app/check.py 2>/dev/null; '
        'python3 -c "import chess; print(chess.__version__, chess.__file__)"',
        'python3 -c "import json; print(json.load(open(\'/app/x.json\')))"',
    ],
)
def test_read_only_shell_does_not_count_as_write(command):
    assert _shell_writes_bytes(command) is False


@pytest.mark.parametrize(
    "command",
    [
        "cat > /app/re.json <<'EOF'",
        "echo '[]' > /app/re.json",
        "cp /tmp/x /app/re.json",
        "python3 -c \"open('/app/re.json','w').write('[]')\"",
        "sed -i 's/a/b/' /app/re.json",
        "tee /app/re.json",
    ],
)
def test_write_shell_counts_as_write(command):
    assert _shell_writes_bytes(command) is True


# ── Top-of-loop consultation (tool_name == "") ─────────────────────────────
# The failure shape this guard targets (measured: 111KB reasoning, 2 thinking
# blocks, ZERO tool calls) never reaches the tool-call consultation at all.
# The top-of-loop check_pre runs BEFORE the LLM call and is the only point
# that can deliver the advisory in that shape. It must INJECT (loop proceeds
# into the LLM call, so the message is read) — a block would `continue` back
# to the same check without the LLM ever running, and the message would never
# be seen.

def _ctx_two_turns_of_reasoning(msgs=None):
    from flagscale_agent.react.guard import GuardContext
    big = "x" * 50000
    messages = msgs if msgs is not None else [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": big},
    ]
    return GuardContext(
        tool_name="", tool_args={}, tool_result=None, turn_count=0,
        recent_tool_names=[], recent_tool_history=[], context_pressure=10,
        evictable_indexes=[], messages=messages, assistant_text=big,
        llm_responded=False,
    )


def test_top_of_loop_injects_when_reasoning_budget_exceeded():
    """No tool call yet — the advisory must still be delivered, as an inject."""
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard
    v = FirstArtifactGuard().check_pre(_ctx_two_turns_of_reasoning())
    assert v is not None, "guard must speak even with zero tool calls"
    assert v.action == "inject", "block would never reach the LLM in this shape"
    assert v.reason == "reasoning_exceeded_without_artifact"


def test_top_of_loop_inject_is_rate_limited():
    """Re-injecting every iteration would spam; one per budget's worth only."""
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard
    g = FirstArtifactGuard()
    ctx = _ctx_two_turns_of_reasoning()
    assert g.check_pre(ctx) is not None
    assert g.check_pre(ctx) is None, "second identical consultation must be silent"


def test_top_of_loop_refires_after_more_reasoning():
    from flagscale_agent.react.guard import GuardContext
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard
    g = FirstArtifactGuard()
    g.check_pre(_ctx_two_turns_of_reasoning())
    big = "x" * 100000
    ctx2 = GuardContext(
        tool_name="", tool_args={}, tool_result=None, turn_count=0,
        recent_tool_names=[], recent_tool_history=[], context_pressure=10,
        evictable_indexes=[],
        messages=[{"role": "system", "content": "sys"},
                  {"role": "assistant", "content": big}],
        assistant_text=big, llm_responded=False,
    )
    v = g.check_pre(ctx2)
    assert v is not None and v.action == "inject"


def test_top_of_loop_silent_when_artifact_already_written():
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard
    big = "x" * 50000
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": big},
        {"role": "assistant", "content": "writing",
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "write_file", "arguments": "{}"}}]},
    ]
    g = FirstArtifactGuard()
    assert g.check_pre(_ctx_two_turns_of_reasoning(msgs)) is None


def test_top_of_loop_silent_below_budget():
    from flagscale_agent.react.guard import GuardContext
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard
    small = "x" * 500
    ctx = GuardContext(
        tool_name="", tool_args={}, tool_result=None, turn_count=0,
        recent_tool_names=[], recent_tool_history=[], context_pressure=10,
        evictable_indexes=[],
        messages=[{"role": "system", "content": "s"},
                  {"role": "assistant", "content": small}],
        assistant_text=small, llm_responded=False,
    )
    assert FirstArtifactGuard().check_pre(ctx) is None


def test_reasoning_bytes_reads_the_thinking_key():
    """Regression: reasoning lives in `thinking`, not `content`.

    Measured false negative: a real trial emitted >100KB of reasoning but the
    guard's advisory printed "~0KB", because provider assistant messages keep
    reasoning in a dedicated `thinking` key and only the visible answer in
    `content`. The byte budget was therefore unreachable in production while
    every unit test (which put text in `content`) passed.
    """
    from flagscale_agent.react.guard import GuardContext
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard

    msgs = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "thinking": "x" * 60000},
        {"role": "assistant", "thinking": "y" * 5000,
         "tool_calls": [{"function": {"name": "shell", "arguments": "{}"}}]},
    ]
    ctx = GuardContext(
        tool_name="", tool_args={}, tool_result=None, turn_count=0,
        recent_tool_names=[], recent_tool_history=[], context_pressure=10,
        evictable_indexes=[], messages=msgs, assistant_text="", llm_responded=False,
    )
    assert FirstArtifactGuard()._reasoning_bytes(ctx) == 65000


def test_injects_on_the_measured_failure_shape():
    """The exact shape that motivated the guard: 1 turn, 111KB, 0 tool calls."""
    from flagscale_agent.react.guard import GuardContext
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard

    msgs = [
        {"role": "system", "content": "task"},
        {"role": "assistant", "thinking": "x" * 111000},
    ]
    ctx = GuardContext(
        tool_name="", tool_args={}, tool_result=None, turn_count=0,
        recent_tool_names=[], recent_tool_history=[], context_pressure=10,
        evictable_indexes=[], messages=msgs, assistant_text="", llm_responded=False,
    )
    v = FirstArtifactGuard().check_pre(ctx)
    assert v is not None and v.action == "inject"
    assert v.reason == "reasoning_exceeded_without_artifact"


def test_reasoning_bytes_reads_anthropic_thinking_blocks():
    """Regression: the live provider stores reasoning as content blocks.

    Measured in the running trial: the advisory printed "~0KB of reasoning in"
    while the log held >70KB, because the message carries reasoning as
    {"type": "thinking"} blocks inside `content` — not as a top-level
    `thinking` string and not as a {"type": "text"} block.
    """
    from flagscale_agent.react.guard import GuardContext
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard

    msgs = [
        {"role": "system", "content": "task"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "x" * 70000},
            {"type": "text", "text": "short answer"},
        ]},
    ]
    ctx = GuardContext(
        tool_name="", tool_args={}, tool_result=None, turn_count=0,
        recent_tool_names=[], recent_tool_history=[], context_pressure=10,
        evictable_indexes=[], messages=msgs, assistant_text="", llm_responded=False,
    )
    assert FirstArtifactGuard()._reasoning_bytes(ctx) == 70012


# ── Regression: the 6th "condition silently never true" defect ────────────
# Measured on a real trial: the agent's first write was `/app/NOTES.md`, the
# write was BLOCKED by another guard ("[PostEditBlocked] ... did NOT land"),
# /app still held only check.py, and yet the guard disarmed permanently for
# the remaining 80KB of reasoning. Two independent bugs: (a) any path counted
# as the artifact, (b) the tool NAME alone counted as proof of a landed write.

def test_note_target_is_not_a_deliverable():
    from flagscale_agent.react.guard.first_artifact import _artifact_write
    assert _artifact_write("write_file", {"path": "/app/NOTES.md"}) is False
    assert _artifact_write("write_file", {"path": "/app/re.json"}) is True
    assert _artifact_write("shell", {"command": "cat > /app/NOTES.md <<'EOF'\nx\nEOF"}) is False
    assert _artifact_write("shell", {"command": "cat > /app/re.json <<'EOF'\n[]\nEOF"}) is True


def test_blocked_write_does_not_disarm():
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard
    from flagscale_agent.react.guard import GuardContext
    g = FirstArtifactGuard()
    blocked = ("[PostEditBlocked] This edit did NOT land - a guard blocked it and "
               "NOTHING changed at /app/NOTES.md.")
    g.check_post(GuardContext(tool_name="write_file",
                              tool_args={"path": "/app/re.json"},
                              tool_result=blocked))
    assert g._wrote_artifact is False, "a blocked write must not disarm the guard"
    g.check_post(GuardContext(tool_name="write_file",
                              tool_args={"path": "/app/re.json"},
                              tool_result="Wrote 12 chars to /app/re.json (total file size: 12 bytes)"))
    assert g._wrote_artifact is True


def test_note_write_in_history_does_not_disarm_top_loop():
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard
    from flagscale_agent.react.guard import GuardContext
    hist = [{"role": "assistant", "thinking": "x" * 80000, "tool_calls": [
        {"function": {"name": "write_file", "arguments": '{"path": "/app/NOTES.md"}'}}]}]
    for _ in range(4):
        hist.append({"role": "assistant", "thinking": "y" * 1000})
    v = FirstArtifactGuard().check_pre(GuardContext(tool_name="", messages=hist))
    assert v is not None and v.action == "inject", (
        "a note-only history must NOT disarm the guard; the artifact is missing"
    )


# ── Regression: the 7th "condition silently never true" defect ─────────────
# Measured on a real trial: the advisory read "~0KB of reasoning" while the log
# held 90KB. Cause (read from kernel.py): the provider DROPS assistant thinking
# blocks before writing the message back into history, so reasoning that has
# already been written back is not there to measure. A synthetic history that
# still carries thinking blocks measured ~51KB and hid this completely.

def test_reasoning_bytes_uses_inflight_response_not_only_history():
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard
    from flagscale_agent.react.guard import GuardContext
    g = FirstArtifactGuard()
    # REAL shape: thinking already stripped, only a short text block remains.
    real = GuardContext(
        tool_name="",
        messages=[{"role": "assistant", "content": [{"type": "text", "text": "plan"}]}],
        assistant_text="z" * 50000,
        llm_responded=True,
    )
    assert g._reasoning_bytes(real) >= 50000, (
        "the in-flight response must count; history alone measures ~0 because "
        "the provider drops thinking blocks on write-back"
    )
    v = g.check_pre(real)
    assert v is not None and v.action == "inject"


def test_turn_budget_still_fires_without_any_reasoning_bytes():
    from flagscale_agent.react.guard.first_artifact import FirstArtifactGuard
    from flagscale_agent.react.guard import GuardContext
    hist = [{"role": "assistant", "content": [{"type": "text", "text": "plan"}]}
            for _ in range(5)]
    c = GuardContext(tool_name="", messages=hist)
    assert FirstArtifactGuard()._reasoning_bytes(c) < 100  # bytes are useless here
    v = FirstArtifactGuard().check_pre(c)
    assert v is not None and v.action == "inject", "turn budget is the primary clock"

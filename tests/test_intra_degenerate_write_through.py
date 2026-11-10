"""Falsifier for the degenerate-staircase WRITE-THROUGH rung.

Live evidence this pins (polyglot-c-py, job 2026-11-10__00-11-22):
the intra-stream degenerate detector fired four times (recovery 1/4..4/4)
inside 8 minutes of a 900s wall, and the turn then ended at
`intra_degenerate_max_retries` with THREE tool calls total and the
deliverable never written. Historical win rate for that task is 0.955,
so the loss was the harness ending the turn instead of landing what the
model had already derived.

The fix under test: when the 4th rung is reached, the kernel must NOT end
the turn. It arms exactly ONE write-through round (`_force_tool_next=True`
+ a warning), and only ends the turn if THAT round degenerates too.

Three directions:
  (a) 4 taps then a real tool call -> the turn survives and the tool runs;
  (b) 4 taps then another degenerate -> the turn ends, bounded (no infinite
      loop), i.e. the write-through rung is ONE round, not a reset-to-zero;
  (c) a normal non-degenerate turn is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass

from flagscale_agent.react.kernel import AgentKernel, KernelDeps

FILLER = "Let me write.\nLet me go.\nOK.\nWriting."


class _Display:
    def __init__(self):
        self.warnings = []

    def warn(self, *a, **k):
        self.warnings.append(" ".join(str(x) for x in a))

    def __getattr__(self, _name):
        return lambda *a, **k: None


class _History:
    def __init__(self):
        self.messages = []

    def get_messages(self):
        return list(self.messages)

    def append(self, msg):
        self.messages.append(msg)

    def pop_last_assistant(self):
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                self.messages.pop(i)
                return

    def get_context_pressure(self):
        return 0.0

    def get_evictable_indexes(self):
        return []

    def get_active(self):
        return None

    def report_actual_tokens(self, n):
        pass


@dataclass
class _Config:
    max_iterations: int = 40
    max_continuations: int = 30
    max_output_tokens: int = 0
    context_limit: int = 100000


class _Guards:
    def check_pre(self, *a, **k):
        return None

    def check_post(self, *a, **k):
        return None

    def __getattr__(self, _n):
        return lambda *a, **k: None


class _Judge:
    def reset_turn(self):
        pass

    def __getattr__(self, _n):
        return lambda *a, **k: None


class _ScriptedProvider:
    def __init__(self, script):
        self.script = script
        self.i = 0

    def format_assistant_message(self, response):
        return response

    def format_tool_calls(self, response):
        return []

    def __getattr__(self, _n):
        return lambda *a, **k: None


def _make_deps(script):
    provider = _ScriptedProvider(script)
    display = _Display()
    history = _History()
    executed = []
    force_flags = []

    def call_llm(messages, schemas):
        idx = min(provider.i, len(script) - 1)
        provider.i += 1
        content, tool_calls, thinking, intra = script[idx]
        history.append({"role": "assistant", "content": content})
        usage = {"input_tokens": 10, "output_tokens": 10}
        return (
            {
                "content": content,
                "tool_calls": tool_calls,
                "thinking": thinking,
                "intra_degenerate": intra,
            },
            usage,
        )

    def execute_tools(calls):
        executed.append([c.get("name") for c in calls])
        return ["ok"] * len(calls)

    deps = KernelDeps(
        provider=provider,
        history=history,
        tool_registry=None,
        judge=_Judge(),
        guard_registry=_Guards(),
        config=_Config(),
        display=display,
        get_schemas_fn=lambda: [],
        inject_message_fn=lambda m: None,
        append_tool_results_fn=lambda r: None,
        format_tool_result_fn=lambda i, r: {},
        execute_tools_fn=execute_tools,
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=call_llm,
    )
    return deps, history, display, executed, force_flags


# ── direction (a): staircase exhausted, then a real tool call → turn survives
def test_degenerate_staircase_forces_write_through_and_survives():
    tool_call = [{"id": "t1", "name": "write_file",
                  "arguments": {"path": "/app/main.py.c", "content": "x"}}]
    # Five degenerate taps exhaust the staircase (the branch reads
    # _intra_degen_retries BEFORE incrementing, so >=4 is reached on the 5th
    # entry, which is exactly when the live trajectory printed recovery 4/4);
    # the 6th response is the write-through the kernel must have FORCED
    # rather than ending the turn.
    script = [
        ("", None, FILLER, True),
        ("", None, FILLER, True),
        ("", None, FILLER, True),
        ("", None, FILLER, True),
        ("", None, FILLER, True),
        ("Writing the deliverable.", tool_call, "", False),
        ("Done. [TASK_COMPLETE]", None, "", False),
    ]
    deps, history, display, executed, _ = _make_deps(script)
    kernel = AgentKernel(deps)
    result = kernel.run_turn()

    assert any("forcing" in w and "write-through" in w for w in display.warnings), (
        "the write-through rung never armed — the kernel still ends the turn "
        f"at the 4th degenerate tap. warnings={display.warnings!r}"
    )
    assert executed, (
        "no tool ever executed: the turn died at the degenerate staircase "
        f"instead of landing a deliverable. stop_reason={result.stop_reason!r}"
    )
    assert result.stop_reason == "explicit_signal", (
        f"turn did not reach completion; stop_reason={result.stop_reason!r}"
    )


# ── direction (b): staircase exhausted, then MORE degeneration → bounded end
def test_write_through_rung_is_one_round_not_unbounded():
    script = [("", None, FILLER, True)]  # degenerates forever
    deps, history, display, executed, _ = _make_deps(script)
    kernel = AgentKernel(deps)
    result = kernel.run_turn()

    assert result.stop_reason == "intra_degenerate_max_retries", (
        "an endless-degenerate model must still end the turn; got "
        f"{result.stop_reason!r}"
    )
    assert not executed, "no tool should have run in the all-degenerate script"
    # The rung must have armed exactly once before giving up — a second
    # arm would mean the staircase reset and the turn could loop forever.
    armed = [w for w in display.warnings if "forcing" in w and "write-through" in w]
    assert len(armed) == 1, (
        f"the write-through rung armed {len(armed)} times; it must be ONE round"
    )


# ── direction (c): a normal turn is unaffected
def test_normal_turn_unaffected_by_write_through_rung():
    tool_call = [{"id": "t1", "name": "shell", "arguments": {"command": "ls"}}]
    script = [
        ("Inspecting.", tool_call, "thinking", False),
        ("Finished. [TASK_COMPLETE]", None, "", False),
    ]
    deps, history, display, executed, _ = _make_deps(script)
    kernel = AgentKernel(deps)
    result = kernel.run_turn()

    assert result.stop_reason == "explicit_signal"
    assert executed == [["shell"]]
    assert not any("write-through" in w for w in display.warnings), (
        "the write-through rung fired on a healthy turn"
    )

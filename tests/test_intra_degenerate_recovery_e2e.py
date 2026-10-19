"""END-TO-END falsifier: an intra-stream degenerate abort must RECOVER, not end the turn.

Live evidence this pins (jobs/four_m23/dna-insert__PUzb7eY): the detector fired
4 times, each abort produced an empty assistant_text plus the degenerate thinking,
and the turn then died with empty_output_max_retries after only 489s of an 1800s
wall — no deliverable was ever written, so the grader failed with
"File /app/primers.fasta does not exist".

The fix under test: a response carrying intra_degenerate=True gets its OWN
bounded recovery path (hard-stop nudge demanding a tool call, without
re-injecting the filler), and a subsequent real tool call completes the turn.

Two directions:
  (a) a degenerate abort followed by a tool call must NOT end the turn, and the
      nudge must not reproduce the degenerate text;
  (b) a model that degenerates forever must still end, via
      intra_degenerate_max_retries (bounded, not an infinite loop).
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
    """Scripted responses; each entry is (content, tool_calls, thinking, intra_degen)."""

    def __init__(self, script):
        self.script = script
        self.i = 0
        self.calls = 0

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

    def call_llm(messages, schemas):
        idx = min(provider.i, len(script) - 1)
        provider.i += 1
        provider.calls += 1
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
    return deps, history, display, executed


# ── direction (a): abort then a real tool call → turn survives, filler not re-fed
def test_degenerate_abort_recovers_via_tool_call():
    tool_call = [{"id": "t1", "name": "write_file",
                  "arguments": {"path": "/app/x.txt", "content": "hi"}}]
    script = [
        ("", None, FILLER, True),                       # degenerate abort
        ("Writing the deliverable.", tool_call, "", False),  # recovery: real tool
        ("Done. [TASK_COMPLETE]", None, "", False),     # normal completion
    ]
    deps, history, display, executed = _make_deps(script)
    kernel = AgentKernel(deps)
    result = kernel.run_turn()

    assert executed, "the recovery tool call never executed — turn died after the abort"
    assert result.stop_reason != "empty_output_max_retries", (
        f"abort still consumed the empty-output budget; stop_reason={result.stop_reason!r}"
    )
    assert result.stop_reason == "explicit_signal", (
        f"turn did not reach completion; stop_reason={result.stop_reason!r}"
    )
    # The degenerate text must NOT be re-injected into the conversation.
    dumped = "\n".join(str(m.get("content", "")) for m in history.messages)
    assert "Let me go." not in dumped, (
        "the degenerate filler was re-injected into history — it re-feeds the loop"
    )
    assert any("Degenerate repeat inside the reasoning stream" in w for w in display.warnings)


# ── direction (b): endless degeneration still ends, bounded
def test_endless_degeneration_ends_with_its_own_reason():
    script = [("", None, FILLER, True)]  # always degenerate
    deps, history, display, executed = _make_deps(script)
    kernel = AgentKernel(deps)
    result = kernel.run_turn()

    assert result.stop_reason == "intra_degenerate_max_retries", (
        f"expected the bounded intra-degenerate stop, got {result.stop_reason!r}"
    )
    assert not executed, "no tool should have run in the all-degenerate script"


# ── falsifier: a normal (non-degenerate) turn must be unaffected
def test_normal_turn_unaffected_by_the_new_branch():
    tool_call = [{"id": "t1", "name": "shell", "arguments": {"command": "ls"}}]
    script = [
        ("Inspecting.", tool_call, "thinking about layout", False),
        ("Finished. [TASK_COMPLETE]", None, "", False),
    ]
    deps, history, display, executed = _make_deps(script)
    kernel = AgentKernel(deps)
    result = kernel.run_turn()
    assert result.stop_reason == "explicit_signal"
    assert executed == [["shell"]]

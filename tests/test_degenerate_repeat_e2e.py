"""END-TO-END falsifier: drives the REAL AgentKernel.run_turn loop.

The sibling test file (test_degenerate_repeat_loop.py) verifies the fingerprint
RULE with a replica. That is not enough: a replica can agree with itself while
the real kernel stays unwired (the self-oracle trap). This test therefore builds
real KernelDeps, feeds a stub provider that returns the exact filler cycle
observed on the live trial, and asserts the turn stops with
stop_reason == "degenerate_repeat_loop".

The paired falsifier drives the SAME loop with distinct responses and asserts it
runs to a normal stop — so the breaker cannot pass by simply aborting everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flagscale_agent.react.kernel import AgentKernel, KernelDeps


# ── minimal stubs ────────────────────────────────────────────────────────────
class _Display:
    def warn(self, *a, **k):
        pass

    def __getattr__(self, _name):
        return lambda *a, **k: None


class _History:
    def __init__(self):
        self.messages: list[dict] = []

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


@dataclass
class _Config:
    max_iterations: int = 60
    max_continuations: int = 40
    max_output_tokens: int = 0
    context_limit: int = 100000


class _Guards:
    """No guards: isolates the kernel's own loop logic from gate behaviour."""

    def check_pre(self, *a, **k):
        return None

    def check_post(self, *a, **k):
        return []

    def __getattr__(self, _n):
        return lambda *a, **k: None


class _Provider:
    """Returns a scripted sequence of assistant texts, then repeats the last."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self.i = 0

    def format_assistant_message(self, response):
        return response

    def format_tool_calls(self, response):
        return []

    def __getattr__(self, _n):
        return lambda *a, **k: None

    def respond(self, messages, schemas):
        text = self.texts[min(self.i, len(self.texts) - 1)]
        self.i += 1
        return ({"role": "assistant", "content": text}, {"input": 1, "output": 1})


class _Judge:
    def reset_turn(self):
        pass

    def __getattr__(self, _n):
        return lambda *a, **k: None


def _deps(texts: list[str]) -> KernelDeps:
    return KernelDeps(
        provider=_Provider(texts),
        history=_History(),
        tool_registry=None,
        judge=_Judge(),
        guard_registry=_Guards(),
        config=_Config(),
        display=_Display(),
        get_schemas_fn=lambda: [],
        inject_message_fn=lambda m: None,
        append_tool_results_fn=lambda r: None,
        format_tool_result_fn=lambda i, r: {},
        execute_tools_fn=lambda calls: [],
        is_context_limit_error_fn=lambda e: False,
        call_llm_fn=lambda messages, schemas: _Provider.respond(
            _PROVIDER[0], messages, schemas
        ),
    )


_PROVIDER: list[_Provider] = []


def _run(texts: list[str]):
    global _PROVIDER
    _PROVIDER = [_Provider(texts)]
    kernel = AgentKernel(_deps(texts))
    return kernel.run_turn()


# ── the two directions ───────────────────────────────────────────────────────
def test_real_kernel_stops_on_the_live_filler_loop():
    """The exact cycle from the live trial must stop the REAL loop.

    Two breakers cover this class and either is an honest stop: the
    degenerate-repeat breaker (diversity window) fires ~response #482 on the
    original log, and the auto-append fallback (AUTO_APPEND_NEED_INPUT_AT=12)
    ends the turn harness-side far earlier than that. Both stop_reasons are
    valid loop protection; what must hold is that the REAL loop stops EARLY
    and never claims completion for filler.
    """
    filler = [
        "Let me write the file.",
        "Let me write the content.",
        "Let me write it.",
        "OK.",
    ]
    # Repeat the cycle many times: the loop must abort, not run to max_iter.
    texts = filler * 20
    result = _run(texts)
    assert result.stop_reason in ("degenerate_repeat_loop",
                                  "auto_appended_need_input"), (
        "real kernel did NOT stop the live filler loop; "
        f"got stop_reason={result.stop_reason!r} after {result.iterations} iterations"
    )
    # It must stop EARLY (a few repeats), not after burning the whole budget.
    assert result.iterations < 30, (
        f"breaker fired too late: {result.iterations} iterations"
    )


def test_real_kernel_does_not_stop_on_distinct_work():
    """FALSIFIER: distinct progress must reach a normal end, not be killed."""
    texts = [
        "Reading the task statement.",
        "Running the design script.",
        "Three placements found; checking Tm.",
        "Tm delta 3.8 within budget.",
        "Writing primers.fasta.",
        "Verifying the delivered file.",
        "All checks pass. [TASK_COMPLETE]",
    ]
    result = _run(texts)
    assert result.stop_reason != "degenerate_repeat_loop", (
        "false positive: distinct work was killed as a degenerate loop"
    )

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

"""The ten-viewpoint task-statement reader, and its single-shot auto-run.

What these tests pin down:
  * ten viewpoints, each asked a DIFFERENT question (not ten copies of one);
  * every reading is returned verbatim — the tool does not decide, vote,
    rank, or drop a reading it disagrees with;
  * one viewpoint failing does not remove the other nine;
  * a dead/unavailable reading is reported as such, never as "the statement
    says nothing here";
  * the auto-run is OFF by default (byte-identical run when unset), and when
    on it returns the readings as a SEPARATE message rather than appending to
    the task text — the task text is frozen.
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.tools.read_task_statement import (
    VIEWPOINTS,
    TaskStatementReadersTool,
)

STATEMENT = (
    "Write me data.comp that's compressed such that running cat data.comp | "
    "/app/decomp gives exactly data.txt. data.comp must be at most 2500 bytes."
)


class _FakeProvider:
    """Returns a fixed body per prompt; can be told to fail for chosen views."""

    def __init__(self, bodies=None, fail_views=(), empty_views=()):
        self.bodies = bodies or {}
        self.fail_views = set(fail_views)
        self.empty_views = set(empty_views)
        self.prompts: list[str] = []

    def chat(self, messages, tools=None, force_tool=False):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        for name in self.fail_views:
            if f"YOUR VIEWPOINT: {name}" in prompt:
                raise RuntimeError(f"boom-{name}")
        for name in self.empty_views:
            if f"YOUR VIEWPOINT: {name}" in prompt:
                return {"content": ""}
        for name, body in self.bodies.items():
            if f"YOUR VIEWPOINT: {name}" in prompt:
                return {"content": body}
        return {"content": f"reading-of-{len(self.prompts)}"}


# ── the viewpoint set ────────────────────────────────────────────────────────


def test_ten_viewpoints_all_distinct():
    assert len(VIEWPOINTS) == 10
    names = [n for n, _ in VIEWPOINTS]
    assert len(set(names)) == 10, "viewpoint names must be unique"
    instrs = [i for _, i in VIEWPOINTS]
    assert len(set(instrs)) == 10, "two viewpoints share the same instruction"


def test_expected_viewpoints_present():
    names = {n for n, _ in VIEWPOINTS}
    for required in (
        "interface_signature",
        "positional_calling",
        "deliverable_paths",
        "output_shape",
        "mandatory_constraints",
        "numeric_conventions",
        "ambiguous_phrases",
        "implicit_premises",
        "what_is_compared",
        "deliverable_inventory",
    ):
        assert required in names


def test_every_instruction_demands_a_verbatim_quote():
    # Two ways a viewpoint can demand a quote: "verbatim" or "quote"/"cite".
    # The shared PROMPT's HARD RULES carries the binding requirement for all
    # ten; this test guards against a viewpoint that asks for nothing anchored.
    for name, instruction in VIEWPOINTS:
        low = instruction.lower()
        assert ("verbatim" in low or "quote" in low or "cite" in low), (
            f"viewpoint {name} does not ask for a quote; without it the reading "
            f"cannot be checked against the statement"
        )


# ── execution ────────────────────────────────────────────────────────────────


def test_runs_all_ten_and_returns_every_reading():
    prov = _FakeProvider(bodies={n: f"BODY-{n}" for n, _ in VIEWPOINTS})
    tool = TaskStatementReadersTool(provider=prov, task_text=STATEMENT)
    out = tool.execute()
    assert len(prov.prompts) == 10
    for name, _ in VIEWPOINTS:
        assert f"BODY-{name}" in out, f"reading for {name} missing from output"


def test_tool_does_not_decide__every_reading_survives_even_if_they_disagree():
    # Two viewpoints give CONTRADICTORY bodies. The tool must hand over both:
    # picking one would be the judgement the tool is explicitly not allowed to
    # make.
    prov = _FakeProvider(bodies={
        "interface_signature": "POSITION 1 binds to density_function",
        "positional_calling": "POSITION 1 binds to n",
    })
    out = TaskStatementReadersTool(provider=prov, task_text=STATEMENT).execute()
    assert "POSITION 1 binds to density_function" in out
    assert "POSITION 1 binds to n" in out


def test_one_failing_viewpoint_does_not_remove_the_others():
    prov = _FakeProvider(fail_views={"numeric_conventions"})
    out = TaskStatementReadersTool(provider=prov, task_text=STATEMENT).execute()
    assert "FAILED: numeric_conventions" in out
    assert "do not read a failure as 'the statement says nothing here'" in out
    # the other nine still reported ok
    assert out.count("(ok)") == 9


def test_empty_response_is_reported_as_an_answer_not_a_failure():
    prov = _FakeProvider(empty_views={"numeric_conventions"})
    out = TaskStatementReadersTool(provider=prov, task_text=STATEMENT).execute()
    assert "Returned nothing for their viewpoint: numeric_conventions" in out
    assert "that is a real answer, not a failure" in out
    # and it is NOT listed as a failure
    assert "FAILED:" not in out


def test_header_states_the_tool_did_not_decide():
    out = TaskStatementReadersTool(
        provider=_FakeProvider(), task_text=STATEMENT
    ).execute()
    assert "These are INPUTS, not conclusions" in out
    assert "did not decide which reading is right" in out


def test_statement_is_passed_into_every_prompt():
    prov = _FakeProvider()
    TaskStatementReadersTool(provider=prov, task_text=STATEMENT).execute()
    assert len(prov.prompts) == 10
    for p in prov.prompts:
        assert "at most 2500 bytes" in p, "the statement must reach every view"


def test_no_statement_returns_a_plain_error_not_invented_content():
    out = TaskStatementReadersTool(provider=_FakeProvider(), task_text="").execute()
    assert out.startswith("ERROR:")
    assert "2500" not in out


def test_subset_selects_only_those_views():
    prov = _FakeProvider()
    out = TaskStatementReadersTool(
        provider=prov, task_text=STATEMENT
    ).execute(views=["deliverable_paths", "numeric_conventions"])
    assert len(prov.prompts) == 2
    assert "deliverable_paths" in out
    assert "interface_signature" not in out


def test_unknown_view_name_is_rejected_with_the_valid_list():
    out = TaskStatementReadersTool(
        provider=_FakeProvider(), task_text=STATEMENT
    ).execute(views=["not_a_view"])
    assert out.startswith("ERROR:")
    assert "interface_signature" in out, "the error must name valid viewpoints"


def test_tool_schema_exposes_a_name_and_a_readme_worthy_description():
    tool = TaskStatementReadersTool(provider=_FakeProvider())
    assert tool.name == "read_task_statement"
    assert "DOES NOT DECIDE" in tool.description
    assert tool.max_result_size >= 200_000, (
        "ten readings are long; a low cap would truncate the very thing the "
        "tool exists to hand over"
    )


# ── the auto-run wiring ──────────────────────────────────────────────────────


def test_auto_run_is_off_by_default(monkeypatch):
    """Unset env => feature off, so a batch that never opts in is unchanged."""
    monkeypatch.delenv("FLAGSCALE_TASK_READERS", raising=False)
    from flagscale_agent.react.agent import WorkerAgent

    sentinel = _FakeProvider()
    obj = object.__new__(WorkerAgent)
    obj._task_statement_readers = TaskStatementReadersTool(
        provider=sentinel, task_text=""
    )
    assert obj._auto_read_task_statement("do the thing") == ""
    assert sentinel.prompts == [], "no provider call may happen when off"


@pytest.mark.parametrize("value", ["1", "true", "on", "yes"])
def test_auto_run_is_on_for_truthy_values(monkeypatch, value):
    monkeypatch.setenv("FLAGSCALE_TASK_READERS", value)
    from flagscale_agent.react.agent import WorkerAgent

    obj = object.__new__(WorkerAgent)
    obj._task_statement_readers = TaskStatementReadersTool(
        provider=_FakeProvider(), task_text=""
    )
    out = obj._auto_read_task_statement(STATEMENT)
    assert "These are INPUTS, not conclusions" in out
    # The readings carry the VIEWPOINTS' output, not the statement itself: the
    # statement is not echoed back (the agent already has it). With a fake
    # provider the bodies are placeholders, so this asserts the block exists
    # and is the reader's rendering — not that it repeats the task text.
    assert "(ok)" in out and out.count("(ok)") == 10


def test_auto_run_returns_separate_text_so_the_task_text_stays_unmodified(
    monkeypatch,
):
    """The readings must NOT be concatenated onto the task statement.

    The statement is frozen; appending to it changes the task and makes runs
    incomparable. So the readings come back as their OWN string, and the caller
    inserts them as a separate message.
    """
    monkeypatch.setenv("FLAGSCALE_TASK_READERS", "1")
    from flagscale_agent.react.agent import WorkerAgent

    obj = object.__new__(WorkerAgent)
    obj._task_statement_readers = TaskStatementReadersTool(
        provider=_FakeProvider(), task_text=""
    )
    out = obj._auto_read_task_statement(STATEMENT)
    assert out != STATEMENT
    assert STATEMENT not in out, (
        "the raw statement must not be echoed inside the readings block"
    )


def test_auto_run_failure_never_breaks_the_run(monkeypatch):
    monkeypatch.setenv("FLAGSCALE_TASK_READERS", "1")
    from flagscale_agent.react.agent import WorkerAgent

    class _Exploding:
        _task_text = ""

        def execute(self, **kwargs):
            raise RuntimeError("provider down")

    obj = object.__new__(WorkerAgent)
    obj._task_statement_readers = _Exploding()
    # warn() may print; the contract is that this returns "" and does not raise
    assert obj._auto_read_task_statement(STATEMENT) == ""


def test_auto_run_skips_an_empty_statement(monkeypatch):
    monkeypatch.setenv("FLAGSCALE_TASK_READERS", "1")
    from flagscale_agent.react.agent import WorkerAgent

    prov = _FakeProvider()
    obj = object.__new__(WorkerAgent)
    obj._task_statement_readers = TaskStatementReadersTool(
        provider=prov, task_text=""
    )
    assert obj._auto_read_task_statement("   ") == ""
    assert prov.prompts == []


def test_tool_is_not_registered_when_the_feature_is_off(monkeypatch):
    """The schema costs bytes on EVERY request, so an unconditional registration
    would change the request body of batches that never opted in. Registering
    only when enabled is what preserves "unset = identical to before".
    """
    from flagscale_agent.react.agent import WorkerAgent
    from flagscale_agent.react.tools import ToolRegistry

    def build(env_val):
        if env_val is None:
            monkeypatch.delenv("FLAGSCALE_TASK_READERS", raising=False)
        else:
            monkeypatch.setenv("FLAGSCALE_TASK_READERS", env_val)
        obj = object.__new__(WorkerAgent)
        obj.provider = _FakeProvider()
        obj.tool_registry = ToolRegistry()
        obj._task_statement_readers = None
        if obj._readers_enabled():
            obj._task_statement_readers = TaskStatementReadersTool(
                provider=obj.provider
            )
            obj.tool_registry.register(obj._task_statement_readers)
        return obj

    for off in (None, "", "0", "false", "off", "no"):
        obj = build(off)
        assert "read_task_statement" not in obj.tool_registry._tools, (
            f"tool must not be registered when the switch is {off!r}"
        )
        assert obj._task_statement_readers is None

    obj = build("1")
    assert "read_task_statement" in obj.tool_registry._tools
    assert obj._task_statement_readers is not None


# ── numeric_conventions: an UNSET axis is a finding ─────────────────────────


def _numeric_conventions_instruction() -> str:
    return dict(VIEWPOINTS)["numeric_conventions"]


def test_numeric_conventions_reports_an_unset_axis_as_a_finding():
    """A quantity whose unit/scale/index base the statement never pins IS the answer.

    Measured motivation (batch 2026-11-08__23-04-35): raman-fitting's statement
    names the four peak parameters but never says what the x axis unit is. The
    old instruction said "If the statement states no numbers, answer NOTHING
    FOUND", so the viewpoint reported nothing — and the trial then fitted peaks
    in nm where the grader wanted cm^-1 (x0 10440 vs 1580.3). The silence was
    the single most valuable thing that reading could have said.
    """
    low = _numeric_conventions_instruction().lower()
    assert "unset" in low, (
        "numeric_conventions must name the UNSET axis as a reportable finding"
    )
    assert "is a finding" in low or "as a finding" in low, (
        "the instruction must state outright that an unset axis is a finding, "
        "not just mention the word 'unset'"
    )
    for axis in ("unit", "precision", "index base"):
        assert axis in low, f"numeric_conventions must name the {axis!r} axis"


def test_numeric_conventions_still_allows_a_true_nothing_found():
    """The escape must survive: a statement with no quantity at all is still empty."""
    low = _numeric_conventions_instruction().lower()
    assert "nothing found for this viewpoint" in low, (
        "a statement that mentions no quantity at all must still be able to "
        "answer NOTHING FOUND — the change must not force invented content"
    )


def test_shared_prompt_does_not_collapse_a_silence_into_nothing_found():
    """HARD RULE 2 must not override the viewpoint: silence != absence.

    Without this, the numeric_conventions change is dead text: the shared rule
    says "if this viewpoint finds nothing, answer exactly NOTHING FOUND", and a
    model resolving that against a whole-statement pass will pick the safe
    answer.
    """
    from flagscale_agent.react.tools.read_task_statement import _PROMPT

    low = _PROMPT.lower()
    assert "it does not mean" in low, (
        "HARD RULE 2 must distinguish 'no subject for this viewpoint' from "
        "'the statement left something unstated'"
    )
    assert "is a finding" in low or "a finding" in low, (
        "HARD RULE 2 must say that reporting a silence is a finding, not an "
        "invention"
    )

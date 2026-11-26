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

"""Order of the two start-up user turns: STATEMENT first, READINGS second.

Why this file exists (measured 2026-11-09). In batch 2026-11-08__23-04-35 all
86 sessions with the readers feature enabled had

    messages[1] = "Ten independent readings of the task statement (10 ran)..."
    messages[2] = the task statement itself

i.e. the model's FIRST exposure to the task was a derived document. Three
independent statements in the tree say that is wrong:

  * agent.py's own comment: "the extra readings arrive as a SECOND user turn";
  * the tool's docstring: the readings arrive as a tool result, i.e. after the
    agent already has the statement;
  * tests/test_read_task_statement.py: "the statement is not echoed back (the
    agent already has it)".

The concrete damage is not cosmetic. `_mech_original_task()` defines "the
original task" as the FIRST user turn and hard reset re-injects it verbatim as
"### (1) ORIGINAL TASK". With readings first, a hard reset handed back the
readings block and the statement was lost.

These tests drive the REAL `_run_single_shot` append path, not the helper.
The pre-existing tests in tests/test_read_task_statement.py all call
`_auto_read_task_statement()` directly and never observe the history, which is
exactly why the ordering bug survived them.
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.tools.read_task_statement import TaskStatementReadersTool

STATEMENT = (
    'You are given the output file of a Raman Setup. We used it to measure '
    'some graphene sample.\n\nFit the G and 2D Peak of the spectrum and return '
    'the x0, gamma, amplitude and offset of the peaks and write them to a file '
    'called "/app/results.json".'
)


class _StubProvider:
    """Minimal stand-in so the readers block renders without a real model."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, messages, tools=None, force_tool=False):
        self.prompts.append(messages[0]["content"])
        return {"content": "stub-reading-body"}


class _Recorder:
    """Records history appends so the ORDER is observable without a react loop."""

    def __init__(self) -> None:
        self._full_log: list[dict] = []
        self.appends: list[dict] = []

    def append(self, msg: dict) -> None:
        self.appends.append(msg)
        self._full_log.append(msg)


def _make_agent(monkeypatch, *, readers_on: bool):
    """A WorkerAgent wired just enough to run the real append block.

    `_run_single_shot` is not re-implemented here: we bind the real method to a
    stand-in object whose collaborators are stubs, then call it. That keeps the
    assertion on the PRODUCTION append bytes rather than on a copy of them.
    """
    from flagscale_agent.react.agent import WorkerAgent

    agent = object.__new__(WorkerAgent)
    agent.history = _Recorder()
    agent._task_statement_readers = TaskStatementReadersTool(
        provider=_StubProvider(), task_text=""
    )
    agent._plan_guard = None
    agent._startup_guard = None
    agent._interrupted = False
    agent._session_dir = "/tmp/order-test"

    # Collaborators the real _run_single_shot touches, replaced by no-ops.
    agent._inject_context = lambda: None
    agent._react_loop = lambda: None
    agent._finalize_single_shot = lambda: None
    agent._auto_save = lambda: None

    monkeypatch.setattr(
        "flagscale_agent.react.agent.resolve_resume_query", lambda: None
    )
    monkeypatch.setattr(
        "flagscale_agent.react.agent.resolve_worker_query", lambda q: q
    )
    if readers_on:
        monkeypatch.setenv("FLAGSCALE_TASK_READERS", "1")
    else:
        monkeypatch.delenv("FLAGSCALE_TASK_READERS", raising=False)
    return agent


def test_statement_is_the_only_user_turn_in_slim_reader_mode(monkeypatch):
    """THE regression test (slim-reader redesign, 2026-11-24 user ruling).

    The twelve-viewpoint readings used to arrive as a SECOND pinned user turn.
    Measured on dna-assembly job 2026-11-17__20-50-10 that block was delivered
    and NEVER opened (zero of its unique eight-word sequences appeared in the
    assistant output). The slim mode removes that user turn: the blind-spot
    signal now rides the SYSTEM PROMPT (a resident region), and the only
    start-up user turn is the task statement itself at turn[0].
    """
    agent = _make_agent(monkeypatch, readers_on=True)
    agent._blindspot_block = ""
    agent._run_single_shot(STATEMENT)

    user_appends = [m for m in agent.history.appends if m.get("role") == "user"]
    assert len(user_appends) == 1, (
        "slim-reader mode appends exactly ONE user turn (the statement); the "
        f"readings are no longer a user turn. got {len(user_appends)}"
    )
    first = user_appends[0]
    assert first["content"] == STATEMENT, (
        "the ONLY user turn must be the task statement verbatim; got: "
        f"{str(first['content'])[:120]!r}"
    )
    # The statement must not have been rewritten on its way in.
    assert "Fit the G and 2D Peak" in first["content"]


def test_statement_turn_is_pinned(monkeypatch):
    """The single statement user turn is pinned so it is never evicted.

    In slim-reader mode there is no second readings turn to pin; the blind-spot
    region lives in the system prompt (rebuilt every turn), not the history.
    """
    agent = _make_agent(monkeypatch, readers_on=True)
    agent._blindspot_block = ""
    agent._run_single_shot(STATEMENT)

    user_appends = [m for m in agent.history.appends if m.get("role") == "user"]
    assert len(user_appends) == 1
    assert user_appends[0].get("_pin") is True, (
        "the task statement is a genuine user turn and must never be evicted"
    )


def test_mech_original_task_returns_the_statement_not_the_readings(monkeypatch):
    """Hard reset re-injects `_mech_original_task()` verbatim as ORIGINAL TASK.

    Before the fix this returned the readings block, so a hard reset would hand
    the agent a derived document where its task should be.
    """
    agent = _make_agent(monkeypatch, readers_on=True)
    agent._run_single_shot(STATEMENT)

    recovered = agent._mech_original_task()
    assert "Ten independent readings" not in recovered, (
        "_mech_original_task() returned the READINGS block — a hard reset would "
        f"lose the task statement. Got: {recovered[:160]!r}"
    )
    assert "Fit the G and 2D Peak" in recovered, (
        f"expected the task statement, got: {recovered[:160]!r}"
    )


def test_feature_off_leaves_a_single_user_turn_and_no_readings(monkeypatch):
    """A batch that does not opt in must be byte-identical to before."""
    agent = _make_agent(monkeypatch, readers_on=False)
    agent._run_single_shot(STATEMENT)

    user_appends = [m for m in agent.history.appends if m.get("role") == "user"]
    assert len(user_appends) == 1, (
        "with the feature off there must be exactly one user turn, "
        f"got {len(user_appends)}"
    )
    assert user_appends[0]["content"] == STATEMENT
    assert user_appends[0].get("_pin") is True

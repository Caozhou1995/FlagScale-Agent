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

"""The multi-viewpoint task-statement reader, and its single-shot auto-run.

What these tests pin down:
  * N viewpoints, each asked a DIFFERENT question (not N copies of one);
  * every reading is returned verbatim — the tool does not decide, vote,
    rank, or drop a reading it disagrees with;
  * one viewpoint failing does not remove the others;
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


def test_viewpoints_all_distinct():
    # The COUNT is deliberately not pinned here: the set grew from ten to
    # eleven when world_anchor was added, and pinning a literal turns every
    # future addition into a test edit that says nothing about correctness.
    # What must hold is that each viewpoint is a distinct question.
    assert len(VIEWPOINTS) >= 10
    names = [n for n, _ in VIEWPOINTS]
    assert len(set(names)) == len(VIEWPOINTS), "viewpoint names must be unique"
    instrs = [i for _, i in VIEWPOINTS]
    assert len(set(instrs)) == len(VIEWPOINTS), "two viewpoints share the same instruction"


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
        "world_anchor",
    ):
        assert required in names


def test_world_anchor_asks_for_a_canonical_value_and_the_field_it_pins():
    """world_anchor exists because two failed trials had the statement name a
    real-world object whose public spec settled the disputed value, and the
    readings never asked for it:

      * raman-fitting names "some graphene sample" -> the grader expects the
        graphene G peak at 1580.3 cm^-1, which pins the axis scale; the
        delivered x0 was off by 1148%.
      * make-doom-for-mips names doomgeneric -> Doom's native frame is
        320x200; the delivered frame was 640x400.

    In both readings files the entity appeared ONLY as an ambiguity source
    ("unspecified sample") and the canonical number appeared ZERO times
    (measured: 1580 -> 0 hits in a 19785-char readings block; 320/200/
    resolution -> 0 hits each in a 20852-char block). So the instruction must
    ask for the value AND for the field it pins -- a viewpoint that returns
    "graphene" without a number would reproduce the original failure.
    """
    names = dict(VIEWPOINTS)
    assert "world_anchor" in names
    low = names["world_anchor"].lower()
    # (a) it must ask what the authoritative value IS
    assert "canonical" in low, "must ask for the canonical value"
    # (b) it must bind that value to an UNSET axis or deliverable field
    assert "pin" in low, "must ask which UNSET axis / deliverable field it pins"
    # (c) the statement's naming must be declared to outrank a self-inferred
    #     range -- this is the exact sentence the raman trial overrode by
    #     rejecting 1580 as "below the minimum 1648".
    assert "outrank" in low, "must state that the named authority outranks self-inference"
    # (d) it must not license invention when no canonical value is known
    assert "say which" in low or "not something you can state" in low


def test_every_instruction_demands_a_verbatim_quote():
    # Two ways a viewpoint can demand a quote: "verbatim" or "quote"/"cite".
    # The shared PROMPT's HARD RULES carries the binding requirement for all
    # viewpoints; this test guards against a viewpoint that asks for nothing anchored.
    for name, instruction in VIEWPOINTS:
        low = instruction.lower()
        assert ("verbatim" in low or "quote" in low or "cite" in low), (
            f"viewpoint {name} does not ask for a quote; without it the reading "
            f"cannot be checked against the statement"
        )


# ── execution ────────────────────────────────────────────────────────────────


def test_runs_every_viewpoint_and_returns_every_reading():
    prov = _FakeProvider(bodies={n: f"BODY-{n}" for n, _ in VIEWPOINTS})
    tool = TaskStatementReadersTool(provider=prov, task_text=STATEMENT)
    out = tool.execute()
    assert len(prov.prompts) == len(VIEWPOINTS)
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
    # every other viewpoint still reported ok
    assert out.count("(ok)") == len(VIEWPOINTS) - 1


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
    assert len(prov.prompts) == len(VIEWPOINTS)
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
    #
    # `_READERS_ROUNDS` passes are run and EVERY pass renders all
    # viewpoints, so the count is rounds x len(VIEWPOINTS). Read the constant
    # rather than pinning a literal: the number of passes is a deliberate,
    # separately-argued design value (see its docstring in agent.py).
    rounds = obj._READERS_ROUNDS
    assert "(ok)" in out and out.count("(ok)") == rounds * len(VIEWPOINTS)


def test_each_pass_is_rendered_verbatim_under_its_own_header(monkeypatch):
    """Three passes => three separate, unmerged blocks.

    This is the anti-aggregation test. The harness must NOT merge, dedupe,
    rank or vote on the passes: any of those would bake the harness's
    judgement in where the agent can no longer correct it. So each pass must
    appear as its own block, in run order, with its full ten-viewpoint text
    intact.

    Ablation: aggregating the passes (e.g. emitting only the first, or
    concatenating the viewpoint bodies without the per-pass headers) turns
    this red.
    """
    monkeypatch.setenv("FLAGSCALE_TASK_READERS", "1")
    from flagscale_agent.react.agent import WorkerAgent

    obj = object.__new__(WorkerAgent)
    obj._task_statement_readers = TaskStatementReadersTool(
        provider=_FakeProvider(), task_text=""
    )
    out = obj._auto_read_task_statement(STATEMENT)
    rounds = obj._READERS_ROUNDS
    for i in range(1, rounds + 1):
        assert f"# PASS {i} of {rounds}" in out, (
            f"pass {i} has no header of its own; the passes were merged or "
            f"dropped. Got headers: "
            f"{[ln for ln in out.splitlines() if ln.startswith('# PASS')]}"
        )
    # Order must be run order: PASS 1 before PASS 2 before PASS 3.
    positions = [out.index(f"# PASS {i} of {rounds}") for i in range(1, rounds + 1)]
    assert positions == sorted(positions), "passes are not in run order"


def test_the_preamble_forbids_reading_the_passes_as_a_vote(monkeypatch):
    """The layout is round-by-round, which LOOKS like several ballots.

    The preamble is the only thing the harness adds, and its load-bearing
    clauses are (a) not N votes / minority does not lose, (b) disagreement
    is a finding not a ballot count, (c) unverified raw material, the statement
    wins, (d) say where they disagree before building, (e) an UNSET is a
    decision owed, settled from a DERIVED reading rather than by passing the
    raw value through. Each answers a measured failure mode, so each is
    asserted verbatim here.
    """
    monkeypatch.setenv("FLAGSCALE_TASK_READERS", "1")
    from flagscale_agent.react.agent import WorkerAgent

    obj = object.__new__(WorkerAgent)
    obj._task_statement_readers = TaskStatementReadersTool(
        provider=_FakeProvider(), task_text=""
    )
    out = obj._auto_read_task_statement(STATEMENT)
    # The vote clause names the ACTUAL pass count: hard-coding "three" made
    # this assertion fail the moment _READERS_ROUNDS became 2, and a test that
    # pins an obsolete default turns a correct render red.
    rounds = max(1, int(getattr(WorkerAgent, "_READERS_ROUNDS", 1)))
    _w = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    word = _w.get(rounds, str(rounds))
    vote_clause = "NOT a vote" if rounds == 1 else f"NOT {word} votes"
    for clause in (
        vote_clause,
        "minority does not lose",
        "FINDING to investigate",
        "UNVERIFIED RAW MATERIAL",
        "itself is authoritative",
        "where the passes do not agree",
        # Clause 5 — the UNSET-settlement rule. Load-bearing sub-parts:
        "UNSET is a DECISION you owe, not a note",
        "Whatever the statement DOES pin is authoritative",
        "a reading DERIVED from the material",
        "is itself a choice",
        "A convention you merely recall is NOT a derivation",
        "deliver the DERIVED value",
        "name it as a bet",
    ):
        assert clause in out, (
            f"the preamble lost a load-bearing clause: {clause!r}"
        )


def test_unset_clause_gives_derivation_priority_but_keeps_statement_supreme():
    """Clause 5 must not become an excuse to override the task's own wording.

    Measured failure (raman-fitting, 2026-11-15): the agent derived
    x0 = 1e7/6329 = 1580 cm-1 — exactly the graded value — then delivered the
    raw 6329 nm instead. The fix is a rule that says a DERIVED reading outranks
    passing the raw value through. The trap on the other side is a rule that
    says "pick the physically nicer one", which would license overriding what
    the statement explicitly fixes. Both bounds are asserted here.
    """
    from flagscale_agent.react.agent import WorkerAgent

    p = WorkerAgent._READERS_PREAMBLE
    # (a) must subordinate derivation to what the statement pins, and must come
    # BEFORE (b) — order carries the precedence, a substring test cannot see it.
    i_a = p.index("(a) Whatever the statement DOES pin is authoritative")
    i_b = p.index("(b) Where the statement pins nothing")
    assert i_a < i_b, "clause (a) must precede (b): statement outranks derivation"
    # (b) must say the derived reading wins over passing the raw value through.
    assert "outranks merely passing the raw number through" in p
    # (c) must deny remembered convention the status of a derivation — the
    # dominance test admits SOURCED readings, not recalled ones.
    assert "merely recall is NOT a derivation" in p



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


def test_tool_description_and_render_speak_for_the_SHIPPED_viewpoints():
    """Every sentence that describes the tool must match what it ships.

    Measured drift (2026-11-16): after `world_anchor` was added (10 -> 11), three
    separate descriptions still said "ten" and the tool's own `description`
    listed the ten old names and omitted `world_anchor`. The description is the
    model's only map of the tool, so a stale list advertises a capability the
    run does not have -- and hides the one it just gained.
    """
    from flagscale_agent.react.tools.read_task_statement import (
        VIEWPOINTS,
        TaskStatementReadersTool,
    )

    n = len(VIEWPOINTS)
    ids = [v[0] for v in VIEWPOINTS]

    d = TaskStatementReadersTool.description
    assert f"{n} independent viewpoints" in d
    assert f"all {n} readings" in d
    # EVERY shipped viewpoint must be advertised, by name
    for vid in ids:
        assert vid in d, f"tool description omits shipped viewpoint {vid!r}"
    assert "ten independent" not in d

    # the render header must not claim a stale count either
    tool = TaskStatementReadersTool(provider=None, task_text="")
    rendered = tool._render([(v, "ok", "body") for v in ids], [])
    assert f"{n} independent readings" in rendered
    assert "Ten independent" not in rendered


# ── blind-spot reader: deadline, non-blocking shutdown, outcome stats ─────────
#
# These lock the 2026-11-24 fix for the two defects measured on a real dna run
# (job 2026-11-24__12-11-41): per-pass failures were swallowed silently (empty
# union was byte-identical to "no blind spot"), and the pool's shutdown(wait=
# True) blocked the pre-loop step on the slowest straggler. The reader must now
# (a) record per-call outcome counts, and (b) not block past its wall-clock
# deadline when a pass hangs.

import threading
import time

from flagscale_agent.react.tools.read_task_statement import (
    _BLINDSPOT_SENTINEL,
)


class _BlindspotProvider:
    """Provider whose chat() can return a body, the sentinel, raise, or hang.

    mode is chosen per-call-index from `sequence` (cycled). 'hang' blocks on an
    Event that the test never sets, simulating a straggler the deadline must cut.
    """

    def __init__(self, sequence, body="- a blind spot line"):
        self._sequence = list(sequence)
        self._body = body
        self._i = 0
        self._lock = threading.Lock()
        self._gate = threading.Event()  # never set -> 'hang' blocks forever

    def chat(self, messages, tools=None, force_tool=False):
        with self._lock:
            mode = self._sequence[self._i % len(self._sequence)]
            self._i += 1
        if mode == "body":
            return {"content": self._body}
        if mode == "sentinel":
            return {"content": _BLINDSPOT_SENTINEL}
        if mode == "empty":
            return {"content": ""}
        if mode == "raise":
            raise RuntimeError("boom")
        if mode == "hang":
            # Long enough that the reader's deadline cuts first, short enough
            # that the lingering worker thread does not slow process teardown
            # (concurrent.futures joins non-daemon workers at interpreter exit).
            self._gate.wait(timeout=3)
            return {"content": self._body}
        raise AssertionError(f"unknown mode {mode!r}")


def test_blindspot_stats_recorded_on_normal_pass():
    prov = _BlindspotProvider(sequence=["body", "sentinel", "body"])
    tool = TaskStatementReadersTool(provider=prov, task_text=STATEMENT)
    lines = tool.read_blindspots(rounds=3)
    s = tool._last_blindspot_stats
    assert s["rounds"] == 3
    assert s["contributed"] == 2       # two 'body' passes
    assert s["empty"] == 1             # the sentinel pass contributes no line
    assert s["errored"] == 0
    assert s["timed_out"] == 0
    assert s["lines"] == len(lines) == 1  # both bodies identical -> union de-dups


def test_blindspot_errored_passes_are_counted_not_silent():
    prov = _BlindspotProvider(sequence=["raise", "body", "raise"])
    tool = TaskStatementReadersTool(provider=prov, task_text=STATEMENT)
    lines = tool.read_blindspots(rounds=3)
    s = tool._last_blindspot_stats
    assert s["errored"] == 2
    assert s["contributed"] == 1
    assert len(lines) == 1


def test_blindspot_all_sentinel_is_distinguishable_from_all_errored():
    all_sentinel = _BlindspotProvider(sequence=["sentinel"])
    t1 = TaskStatementReadersTool(provider=all_sentinel, task_text=STATEMENT)
    assert t1.read_blindspots(rounds=3) == []
    assert t1._last_blindspot_stats["empty"] == 3
    assert t1._last_blindspot_stats["errored"] == 0

    all_err = _BlindspotProvider(sequence=["raise"])
    t2 = TaskStatementReadersTool(provider=all_err, task_text=STATEMENT)
    assert t2.read_blindspots(rounds=3) == []
    assert t2._last_blindspot_stats["errored"] == 3
    assert t2._last_blindspot_stats["empty"] == 0
    # same empty result, different stats -> the caller can tell them apart


def test_blindspot_deadline_does_not_block_on_a_straggler():
    # One pass hangs; with a tiny deadline the call must return promptly and
    # count the hung pass as timed_out rather than waiting on it.
    prov = _BlindspotProvider(sequence=["body", "hang", "body"])
    tool = TaskStatementReadersTool(provider=prov, task_text=STATEMENT)
    tool._blindspot_deadline_s = 0.5
    start = time.monotonic()
    lines = tool.read_blindspots(rounds=3)
    elapsed = time.monotonic() - start
    assert elapsed < 10.0, f"reader blocked on straggler: {elapsed:.1f}s"
    s = tool._last_blindspot_stats
    assert s["timed_out"] >= 1
    assert s["contributed"] >= 1
    assert len(lines) >= 1


def test_blindspot_no_task_text_leaves_defined_stats():
    tool = TaskStatementReadersTool(provider=_BlindspotProvider(["body"]),
                                    task_text="")
    assert tool.read_blindspots(rounds=3) == []
    s = tool._last_blindspot_stats
    assert s == {"rounds": 0, "contributed": 0, "empty": 0,
                 "errored": 0, "timed_out": 0, "lines": 0}

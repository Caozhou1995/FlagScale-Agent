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

"""Oracle-at-frame: force the oracle's PREMISE to be external to the agent.

The loss signature: a self-authored oracle shares the agent's own wrong premise
and green-washes the error (recurring base/unit/form/direction/signature/type
failures). This mechanism (a) injects an acceptance item into the fixed FRAME
step demanding an external-premise oracle bound to a first runnable exercise,
and (b) when an `oracle` is supplied to plan_create, refuses a self-authored
premise and binds the first exercise into the FIRST load-bearing step.

Every test here is a FALSIFIER: the ones asserting new behavior fail on the
pre-change code.
"""
import shutil
import tempfile

import pytest

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.tools.plan_create import (
    PlanCreateTool,
    _FRAME_STEP_ACCEPTANCE,
)
from flagscale_agent.react.oracle import (
    PREMISE_SOURCES,
    ORACLE_ACCEPTANCE_ITEM,
    STATED_CONSTRAINT_ITEM,
    AMBIGUITY_ANCHOR_ITEM,
    build_oracle_binding_line,
    declares_external_premise,
)
from flagscale_agent.react.guard.verification import (
    VerificationGuard,
    _oracle_exercised,
    _normalize_for_grounding,
)
from flagscale_agent.react.guard import GuardContext


@pytest.fixture
def plan_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestFrameAcceptanceCarriesOracleItem:
    def test_oracle_item_is_in_frame_acceptance(self):
        assert ORACLE_ACCEPTANCE_ITEM in _FRAME_STEP_ACCEPTANCE

    def test_oracle_item_names_external_premise_sources(self):
        low = ORACLE_ACCEPTANCE_ITEM.lower()
        # all three external premise sources are named
        for src in PREMISE_SOURCES:
            assert src in low
        # the self-authored failure mode is named
        assert "self-authored" in low
        # the first-step binding is demanded
        assert "first load-bearing step" in low
        assert "first runnable exercise" in low

    def test_oracle_item_is_task_agnostic(self):
        # No concrete parser/library/corpus may be named -- that would hand the
        # agent the checker (see tests/test_no_harness_grader_leak.py).
        low = ORACLE_ACCEPTANCE_ITEM.lower()
        for tok in ("bs4", "beautifulsoup", "lxml", "html.parser"):
            assert tok not in low


class TestIsPremiseExternal:
    def test_valid_external_premise(self):
        for src in PREMISE_SOURCES:
            assert declares_external_premise(
                {"oracle": "x", "premise_source": src, "first_exercise": "cmd"}
            )

    def test_self_authored_premise_rejected(self):
        # any premise_source outside the external set is self-authored -> False
        assert not declares_external_premise(
            {"oracle": "x", "premise_source": "self", "first_exercise": "cmd"}
        )
        assert not declares_external_premise(
            {"oracle": "x", "premise_source": "my_own_pipeline", "first_exercise": "cmd"}
        )

    def test_missing_legs_rejected(self):
        # a missing first exercise is not evidence of independence
        assert not declares_external_premise(
            {"oracle": "x", "premise_source": "provided", "first_exercise": ""}
        )
        assert not declares_external_premise(
            {"oracle": "", "premise_source": "provided", "first_exercise": "cmd"}
        )
        assert not declares_external_premise({"premise_source": "provided"})

    def test_non_dict_rejected(self):
        assert not declares_external_premise(None)
        assert not declares_external_premise("provided")
        assert not declares_external_premise([])


class TestPlanCreateOracleBinding:
    def test_self_authored_oracle_refused(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="T", steps=["A"],
            oracle={"oracle": "x", "premise_source": "self", "first_exercise": "cmd"},
        )
        assert "ERROR" in result
        # the plan must NOT have been created
        assert tp.get_active() is None

    def test_valid_oracle_binds_into_first_load_bearing_step(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="T",
            steps=[{"title": "A", "acceptance": ["A1"]}, "B"],
            oracle={
                "oracle": "second impl diff",
                "premise_source": "differential",
                "first_exercise": "python diff_two.py --min",
            },
        )
        assert "Plan created" in result
        plan = tp.get_active()
        # FRAME step is index 0; first agent step is index 1.
        a_accept = plan["steps"][1]["acceptance"]
        assert a_accept[0] == build_oracle_binding_line(
            "second impl diff", "python diff_two.py --min"
        )
        assert "A1" in a_accept
        # the binding must NOT leak onto the SECOND agent step
        assert plan["steps"][2]["acceptance"] == []

    def test_oracle_absent_is_backward_compatible(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(title="T", steps=["A"])
        assert "Plan created" in result
        # no oracle -> first step acceptance stays empty
        assert tp.get_active()["steps"][1]["acceptance"] == []

    def test_oracle_on_string_step_promotes_it(self, plan_dir):
        # a plain string first step is promoted to a structured step carrying
        # the binding, so the acceptance channel still fires.
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="T", steps=["A", "B"],
            oracle={
                "oracle": "golden file",
                "premise_source": "provided",
                "first_exercise": "python cmp.py golden.json out.json",
            },
        )
        assert "Plan created" in result
        a = tp.get_active()["steps"][1]
        assert "golden file" in a["acceptance"][0]


class TestReviewRegressionFixes:
    """Regression tests for the adversarial-review findings (F3/F4/F6/F1-doc)."""

    def test_F4_json_string_oracle_is_decoded(self, plan_dir):
        # A model that JSON-encodes the oracle object must be DEGRADED, not
        # refused -- same normalization `steps` already gets.
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="T", steps=["A"],
            oracle='{"oracle": "golden", "premise_source": "provided", '
                   '"first_exercise": "cmp"}',
        )
        assert "Plan created" in result
        assert "golden" in tp.get_active()["steps"][1]["acceptance"][0]

    def test_F3_oracle_step_id_targets_the_load_bearing_step(self, plan_dir):
        # Bind to the step the agent marks load-bearing, NOT silently the first
        # listed (here step 1 is a trivial preamble).
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="T", steps=["read task", "do the real work"],
            oracle={"oracle": "o", "premise_source": "provided",
                    "first_exercise": "c"},
            oracle_step_id=2,
        )
        assert "Plan created" in result
        plan = tp.get_active()
        assert plan["steps"][1]["acceptance"] == []           # preamble untouched
        assert "Oracle:" in plan["steps"][2]["acceptance"][0]  # load-bearing bound

    def test_F3_oracle_step_id_out_of_range_falls_back_to_first(self, plan_dir):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        result = tool.execute(
            title="T", steps=["A", "B"],
            oracle={"oracle": "o", "premise_source": "provided",
                    "first_exercise": "c"},
            oracle_step_id=99,
        )
        assert "Plan created" in result
        assert "Oracle:" in tp.get_active()["steps"][1]["acceptance"][0]

    def test_F6_unknown_key_refused(self):
        # An incriminating annex key must not ride along a passing declaration.
        assert not declares_external_premise(
            {"oracle": "x", "premise_source": "provided",
             "first_exercise": "c", "note": "my own script"}
        )

    def test_F1_predicate_validates_shape_not_externalness(self):
        # Documented limitation: a mislabeled self-pipeline passes the SHAPE
        # check -- a string label is not evidence. This test pins the honesty of
        # the scope so a future reader is not misled into trusting it as proof;
        # if someone later adds real provenance verification, this test should be
        # updated to assert rejection instead.
        assert declares_external_premise(
            {"oracle": "my own verify.py rerun", "premise_source": "provided",
             "first_exercise": "python verify.py"}
        )


class TestFrameAcceptanceCarriesConstraintCoverageItem:
    """The constraint-coverage item closes the COMPLEMENT of the oracle gate.

    Loss it targets: an agent guards the ONE axis its oracle covers and ships
    with every OTHER stated constraint unverified. Its own report says so
    verbatim ("NOT VERIFIED: ... cannot be checked against a reference") -- yet
    the task STATED the constraint, so it was checkable in form if not in value.
    """

    def test_constraint_item_is_in_frame_acceptance(self):
        assert STATED_CONSTRAINT_ITEM in _FRAME_STEP_ACCEPTANCE

    def test_constraint_item_demands_compiled_runnable_assertion(self):
        low = STATED_CONSTRAINT_ITEM.lower()
        # enumerate EVERY stated constraint, compile each to a runnable check
        assert "enumerate" in low
        assert "compile" in low
        assert "runnable assertion" in low
        assert "run it before you complete" in low

    def test_constraint_item_resists_semantic_inversion(self):
        # F1 (reviewer, reproduced): token-PRESENCE alone is vacuous -- an
        # inversion ("do NOT enumerate", "do NOT compile", "optional") keeps
        # every asserted substring and would still pass. Pin the demand against
        # an actual inversion by asserting the inversion's distinctive tokens are
        # ABSENT. This is the mutation that must fail.
        inv = (
            STATED_CONSTRAINT_ITEM
            .replace("ENUMERATE ", "do NOT enumerate ")
            .replace("COMPILE ", "do NOT compile ")
            .replace("before you complete", "before you complete (optional)")
        )
        low_inv = inv.lower()
        # the inverted text is what a broken item would look like; the guard
        # below must reject it
        assert "do not enumerate" in low_inv    # the inversion really changed it
        assert "do not compile" in low_inv
        # and the REAL item must not carry those negation tokens
        real = STATED_CONSTRAINT_ITEM.lower()
        assert "do not enumerate" not in real
        assert "do not compile" not in real
        assert "(optional)" not in real

    def test_constraint_item_refuses_prose_not_verified(self):
        # The exact escape hatch the observed agent used to ship unchecked.
        low = STATED_CONSTRAINT_ITEM.lower()
        assert "not verified" in low
        assert "prose" in low and "is not" in low

    def test_constraint_item_covers_non_value_form(self):
        # When the task gives no expected VALUE it still gives FORM.
        low = STATED_CONSTRAINT_ITEM.lower()
        assert "form" in low
        assert "bound" in low or "range" in low
        assert "tolerance" in low

    def test_constraint_item_is_distinct_from_oracle_item(self):
        # It must name the oracle item as the ONE-axis fix, not duplicate it.
        assert STATED_CONSTRAINT_ITEM != ORACLE_ACCEPTANCE_ITEM
        low = STATED_CONSTRAINT_ITEM.lower()
        assert "one axis" in low or "one  axis" in low

    def test_constraint_item_is_task_agnostic(self):
        # No concrete unit/threshold/task may be named (grader-leak constraint).
        low = STATED_CONSTRAINT_ITEM.lower()
        for banned in ("raman", "dna", "filter-js", "gpt2", "fasttext",
                       "2670", "0.62", "bs4", "beautifulsoup"):
            assert banned not in low


class TestOracleExerciseActionLeg:
    """The oracle gate's ACTION half: a bound oracle must be EXERCISED, not just
    named. Framing validates the declaration SHAPE; this leg forces the declared
    first_exercise to have been RUN -- a corpus-backed `oracle_act`.

    Every falsifier here fails on the pre-change code (no _oracle_exercised, no
    completion_without_oracle_exercise arm).
    """

    def test_ambiguity_item_is_in_frame_acceptance(self):
        assert AMBIGUITY_ANCHOR_ITEM in _FRAME_STEP_ACCEPTANCE

    def test_ambiguity_item_demands_external_anchor_and_probe(self):
        low = AMBIGUITY_ANCHOR_ITEM.lower()
        assert "external" in low
        assert "probe" in low
        assert "ambiguity axis" in low
        # the self-authored failure mode is named
        assert "self" in low and "reasoning" in low

    def test_ambiguity_item_resists_semantic_inversion(self):
        # token-PRESENCE alone is vacuous; pin the demand against an inversion.
        inv = (
            AMBIGUITY_ANCHOR_ITEM
            .replace("do NOT resolve", "DO resolve")
            .replace("ANCHOR ", "ignore ")
        ).lower()
        assert "do resolve" in inv  # the inversion really changed it
        real = AMBIGUITY_ANCHOR_ITEM.lower()
        assert "do not resolve" in real
        assert "anchor the choice on the strongest external" in real

    def test_ambiguity_item_is_task_agnostic(self):
        low = AMBIGUITY_ANCHOR_ITEM.lower()
        for banned in ("raman", "dna", "fasttext", "gpt2", "2670", "0.62",
                       "bs4", "beautifulsoup", "lxml"):
            assert banned not in low

    def test_oracle_exercised_requires_corpus_backed_output(self):
        # a claimed-but-never-run exercise (observed absent from corpus) is False
        corpus = "real output line 42"
        assert _oracle_exercised(
            {"command": "run oracle", "observed": "real output line 42"}, corpus)
        assert not _oracle_exercised(
            {"command": "run oracle", "observed": "NEVER RAN THIS"}, corpus)

    def test_oracle_exercised_rejects_declaration_only(self):
        # a bare declaration (no command / no observed) is not an exercise
        assert not _oracle_exercised({"oracle": "named"}, "corpus")
        assert not _oracle_exercised({"command": "c"}, "corpus")
        assert not _oracle_exercised({"observed": "o"}, "corpus")
        assert not _oracle_exercised("ran it", "corpus")

    def test_oracle_exercised_fails_open_when_corpus_unavailable(self):
        # shape OK but corpus unknown (None) -> fail-open, never spurious block
        assert _oracle_exercised(
            {"command": "c", "observed": "o"}, None)


class TestOracleExerciseCompletionArm:
    """End-to-end: a plan with a BOUND oracle cannot [TASK_COMPLETE] until the
    oracle's exercise is recorded as a corpus-backed action."""

    def _plan_with_oracle(self, plan_dir, oracle):
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        tool.execute(title="T", steps=["A"], oracle=oracle)
        return tp

    @staticmethod
    def _close_all_steps(tp):
        """Mark every step done/skipped so the unfinished-steps arm is inert and
        the completion path reaches the oracle-exercise arm."""
        plan = tp.get_active()
        for s in plan.get("steps", []):
            tp.update_step(s["id"], "skipped")

    def _ctx(self, corpus_obs=None):
        msgs = []
        if corpus_obs is not None:
            msgs = [{"role": "tool", "content": corpus_obs}]
        return GuardContext(tool_name="", tool_args={}, tool_result=None,
                            assistant_text="[TASK_COMPLETE]", override_reason="",
                            llm_responded=True, messages=msgs)

    def test_bound_oracle_without_exercise_blocks(self, plan_dir):
        tp = self._plan_with_oracle(plan_dir, {
            "oracle": "ref impl", "premise_source": "differential",
            "first_exercise": "python diff.py",
        })
        self._close_all_steps(tp)
        g = VerificationGuard(plan=tp)
        v = g.check_pre(self._ctx())
        assert v is not None and v.reason == "completion_without_oracle_exercise"

    def test_no_bound_oracle_does_not_block_on_this_arm(self, plan_dir):
        # no oracle bound -> this arm is inert (must not regress normal tasks)
        tp = TaskPlan(plan_dir)
        tool = PlanCreateTool(tp)
        tool.execute(title="T", steps=["A"])
        self._close_all_steps(tp)
        g = VerificationGuard(plan=tp)
        v = g.check_pre(self._ctx())
        assert v is None or v.reason != "completion_without_oracle_exercise"

    def test_override_does_not_release_the_arm(self, plan_dir):
        """M2: the oracle-exercise arm is NON-overridable -- the exit is the
        ACTION of running the declared exercise and recording oracle_act, not
        prose."""
        tp = self._plan_with_oracle(plan_dir, {
            "oracle": "ref impl", "premise_source": "provided",
            "first_exercise": "cmp golden",
        })
        self._close_all_steps(tp)
        g = VerificationGuard(plan=tp)
        ctx = self._ctx()
        ctx.override_reason = "oracle unrunnable in this environment"
        v = g.check_pre(ctx)
        assert v is not None and v.reason == "completion_without_oracle_exercise", v
        assert v.overridable is False

    def test_exercised_oracle_releases_the_arm(self, plan_dir):
        # simulate the step_done that RUNS the oracle: latch via oracle_act with
        # a corpus-backed observed output, then completion passes this arm.
        tp = self._plan_with_oracle(plan_dir, {
            "oracle": "ref impl", "premise_source": "differential",
            "first_exercise": "python diff.py",
        })
        g = VerificationGuard(plan=tp)
        obs = "diff.py: 0 differing lines"
        ctx = self._ctx(corpus_obs=obs)
        # the topical relation is read by the judge (no token overlap); stub it
        # so the exercise is affirmable without an LLM, exactly as production
        # passes ctx.judge.
        ctx.judge = _StubJudge(same=True)
        ctx.tool_name = "plan_update"
        ctx.tool_args = {
            "action": "step_done", "step_id": 1,
            "verification": ["ran the oracle"],
            "convention": {"name": "none"},
            "deliverable_reading": {"deliverable": "none"},
            "assumptions": {"unverified": "none"},
            "oracle_act": {"command": "python diff.py", "observed": obs},
        }
        # step_id 1 is the fixed FRAME step; its acceptance is non-empty, so the
        # Mode-1 evidence gate is satisfied by the verification list. The
        # premise-recheck gate fires once on the first step_done.
        v1 = g.check_pre(ctx)
        assert v1 is None or v1.reason == "step_done_premise_recheck"
        ctx.override_reason = "re-checked"
        v2 = g.check_pre(ctx)
        assert v2 is None  # exercise recorded
        # now completion clears the oracle arm
        self._close_all_steps(tp)
        done_ctx = self._ctx(corpus_obs=obs)
        v = g.check_pre(done_ctx)
        assert v is None or v.reason != "completion_without_oracle_exercise"


class TestOracleArmNotDisarmableByPlanMutation:
    """Regression (review F1/F3): a bound oracle must survive the tool completion
    route and plan mutation. The arm previously read the oracle from the ACTIVE
    plan, so `plan_update(action='complete')` (which clears the active pointer),
    a re-frame, or a deactivate() silently disarmed the obligation."""

    def _bind(self, plan_dir, oracle):
        tp = TaskPlan(plan_dir)
        g = VerificationGuard(plan=tp)
        # the guard must LATCH the declaration when it sees plan_create
        g.check_pre(GuardContext(tool_name="plan_create",
                                 tool_args={"title": "T", "oracle": oracle}))
        PlanCreateTool(tp).execute(title="T", steps=["A"], oracle=oracle)
        return tp, g

    @staticmethod
    def _close(tp):
        for s in tp.get_active().get("steps", []):
            tp.update_step(s["id"], "skipped")

    def _complete_ctx(self):
        return GuardContext(tool_name="", tool_args={}, assistant_text="[TASK_COMPLETE]",
                            llm_responded=True, messages=[])

    def test_tool_complete_route_does_not_disarm(self, plan_dir):
        # F1: close via the TOOL route (ungated), then the text sentinel must
        # STILL block -- the active plan is gone, but the run latched the oracle.
        tp, g = self._bind(plan_dir, {
            "oracle": "ref impl", "premise_source": "provided",
            "first_exercise": "python x.py"})
        self._close(tp)
        tp.complete()  # clears the active pointer
        assert g._plan.get_active() in (None, False, {})
        v = g.check_pre(self._complete_ctx())
        assert v is not None and v.reason == "completion_without_oracle_exercise"

    def test_reframe_without_oracle_does_not_disarm(self, plan_dir):
        # F3a: a second plan_create with NO oracle must not drop the obligation.
        tp, g = self._bind(plan_dir, {
            "oracle": "ref impl", "premise_source": "provided",
            "first_exercise": "python x.py"})
        g.check_pre(GuardContext(tool_name="plan_create", tool_args={"title": "T2"}))
        PlanCreateTool(tp).execute(title="T2", steps=["B"])  # no oracle
        self._close(tp)
        v = g.check_pre(self._complete_ctx())
        assert v is not None and v.reason == "completion_without_oracle_exercise"

    def test_deactivate_does_not_disarm(self, plan_dir):
        # F3b: deactivate() must not drop the obligation either.
        tp, g = self._bind(plan_dir, {
            "oracle": "ref impl", "premise_source": "provided",
            "first_exercise": "python x.py"})
        tp.deactivate()
        v = g.check_pre(self._complete_ctx())
        assert v is not None and v.reason == "completion_without_oracle_exercise"


class _StubJudge:
    """Affirms/refuses the prose relation without an LLM (semantic half stub)."""

    def __init__(self, same: bool):
        self._same = same

    def classify(self, category, payload, default=None):
        return self._same


class TestOracleExerciseTopicalRelation:
    """Regression (review F2): the recorded command must relate to the declared
    first_exercise -- an unrelated command with any corpus-backed output must not
    latch the "exercise" (which would make the label meaningless).

    The relation is SEMANTIC, so it is read by the judge (no token overlap -- that
    was the earlier implementation the user ruled out). These tests therefore
    inject a stub judge; the no-judge path is fail-CLOSED and covered explicitly
    below.
    """

    def test_unrelated_command_is_refused(self):
        corpus = _normalize_for_grounding("Plan created.\n\nFOUND: 42")
        assert not _oracle_exercised(
            {"command": "python diff.py", "observed": "Plan created."},
            corpus, "pytest tests/test_x.py", _StubJudge(same=False))

    def test_related_command_is_allowed(self):
        corpus = _normalize_for_grounding("ran ok\nFOUND: 42")
        assert _oracle_exercised(
            {"command": "pytest tests/test_x.py -q", "observed": "FOUND: 42"},
            corpus, "pytest tests/test_x.py -q", _StubJudge(same=True))

    def test_no_judge_fails_closed(self):
        """A CERTIFYING predicate must not pass when the relation is unreadable.

        Accepting here would let any corpus-backed command latch the exercise
        whenever no judge is configured, i.e. the gate disappears exactly when
        the machinery is degraded. Fail-closed is the only direction that cannot
        produce a false PASS.
        """
        corpus = _normalize_for_grounding("ran ok\nFOUND: 42")
        assert not _oracle_exercised(
            {"command": "pytest tests/test_x.py -q", "observed": "FOUND: 42"},
            corpus, "pytest tests/test_x.py -q", None)

    def test_dead_judge_fails_open(self):
        """The JUDGE dying is not the agent's fault -> keep the arm open."""
        class _Boom:
            def classify(self, *a, **k):
                raise RuntimeError("provider down")

        corpus = _normalize_for_grounding("ran ok\nFOUND: 42")
        assert _oracle_exercised(
            {"command": "pytest tests/test_x.py -q", "observed": "FOUND: 42"},
            corpus, "pytest tests/test_x.py -q", _Boom())

    def test_missing_oracle_exercise_fails_open(self):
        corpus = _normalize_for_grounding("FOUND: 42")
        assert _oracle_exercised({"command": "anything", "observed": "FOUND: 42"},
                                 corpus, None)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
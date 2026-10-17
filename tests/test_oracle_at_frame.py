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
    build_oracle_binding_line,
    declares_external_premise,
)


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


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
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
    is_premise_external,
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
            assert is_premise_external(
                {"oracle": "x", "premise_source": src, "first_exercise": "cmd"}
            )

    def test_self_authored_premise_rejected(self):
        # any premise_source outside the external set is self-authored -> False
        assert not is_premise_external(
            {"oracle": "x", "premise_source": "self", "first_exercise": "cmd"}
        )
        assert not is_premise_external(
            {"oracle": "x", "premise_source": "my_own_pipeline", "first_exercise": "cmd"}
        )

    def test_missing_legs_rejected(self):
        # a missing first exercise is not evidence of independence
        assert not is_premise_external(
            {"oracle": "x", "premise_source": "provided", "first_exercise": ""}
        )
        assert not is_premise_external(
            {"oracle": "", "premise_source": "provided", "first_exercise": "cmd"}
        )
        assert not is_premise_external({"premise_source": "provided"})

    def test_non_dict_rejected(self):
        assert not is_premise_external(None)
        assert not is_premise_external("provided")
        assert not is_premise_external([])


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


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])

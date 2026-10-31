# ------------------------------------------------------------ plan_create gate

import tempfile

from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.tools.plan_create import PlanCreateTool
from tests._calibration_fixture import DEFAULT_CALIBRATION
from tests.test_named_standard_requirements import PLAIN_TASK, STANDARD_TASK


def _tool():
    return PlanCreateTool(TaskPlan(tempfile.mkdtemp()), "t-reqs",
                          task_text_fn=lambda: STANDARD_TASK)


def _plan_kwargs():
    return dict(
        title="t", steps=[{"title": "s", "acceptance": ["a"]}],
        oracle={"premise_source": "consumer",
                "oracle": "none: test-only plan",
                "first_exercise": "true"},
    )


def test_plan_create_refuses_missing_requirements_on_standard_task():
    cal = dict(DEFAULT_CALIBRATION)
    cal.pop("requirements_read", None)
    out = _tool().execute(**_plan_kwargs(), calibration=cal)
    assert out.startswith("ERROR:"), out[:200]
    assert "requirements_read" in out
    assert "COPY THIS SHAPE" in out


def test_plan_create_accepts_clause_list():
    cal = dict(DEFAULT_CALIBRATION)
    cal["requirements_read"] = [
        {"clause": "protective base 5' of the site", "source": "protocol p2",
         "covered_by": "added to every segment"},
    ]
    out = _tool().execute(**_plan_kwargs(), calibration=cal)
    assert not out.startswith("ERROR"), out[:200]


def test_plan_create_plain_task_needs_no_list():
    tool = PlanCreateTool(TaskPlan(tempfile.mkdtemp()), "t-reqs-plain",
                          task_text_fn=lambda: PLAIN_TASK)
    cal = dict(DEFAULT_CALIBRATION)
    cal.pop("requirements_read", None)
    out = tool.execute(**_plan_kwargs(), calibration=cal)
    assert not out.startswith("ERROR"), out[:200]


def test_task_text_fn_wired_into_tool():
    """A standard-naming task text must reach the gate through the tool's
    task_text_fn -- the wiring is the mechanism, not just the validator."""
    import inspect
    tool = _tool()
    assert callable(getattr(tool, "_task_text_fn", None)) or \
        callable(getattr(tool, "task_text_fn", None))
    src = inspect.getsource(type(tool))
    assert "task_text_fn" in src


def test_requirements_read_persisted_in_plan_yaml():
    cal = dict(DEFAULT_CALIBRATION)
    cal["requirements_read"] = [
        {"clause": "protective base 5' of the site", "source": "protocol p2",
         "covered_by": "added to every segment"},
    ]
    d = tempfile.mkdtemp()
    tool = PlanCreateTool(TaskPlan(d), "t-reqs-persist",
                          task_text_fn=lambda: STANDARD_TASK)
    out = tool.execute(**_plan_kwargs(), calibration=cal)
    assert not out.startswith("ERROR"), out[:200]
    import glob
    yamls = glob.glob(f"{d}/*.yaml") + glob.glob(f"{d}/**/*.yaml", recursive=True)
    text = "".join(open(p).read() for p in yamls)
    assert "requirements_read" in text, f"not persisted in {yamls}"
    assert "protective base" in text

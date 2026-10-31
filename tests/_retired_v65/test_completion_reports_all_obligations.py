"""The completion boundary must report EVERY unmet obligation at once.

A text-only [TASK_COMPLETE] is ONE decision point guarded by several INDEPENDENT
obligations: the plan's steps must be finished, the acceptance must have been
OBSERVED, and an oracle bound at framing must be both BOUND and EXERCISED. The
old shape returned the first unmet one, so an agent that satisfied it had to
re-claim completion to discover the next — one full round trip per obligation,
the same cascade the registry-level merge removed ACROSS guards.

The falsifier for this file is structural: if check_all_pre were left at the
base-class default (a wrapper around check_pre), the collected count would be
exactly 1 on a context that violates several obligations, and
test_all_obligations_collected_would_fail_under_top1 would go red.
"""
import shutil
import tempfile

from flagscale_agent.react.guard import GuardContext, GuardRegistry
from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.plan import TaskPlan

# --- completion-gate opt-in (GATE_SWITCH_FIXTURE) ---
# The [TASK_COMPLETE] gate defaults to OFF (user instruction 2026-10-22). These
# tests exercise the GATE LOGIC, so they must switch it back on explicitly;
# without this they would report "no block" and look like broken code.
import pytest as _pytest_mod


@_pytest_mod.fixture(autouse=True)
def _enable_completion_gate(monkeypatch):
    from flagscale_agent.react.guard.verification import VerificationGuard
    monkeypatch.setattr(VerificationGuard, "COMPLETION_GATE_ENABLED", True)
    yield
# --- end completion-gate opt-in ---

TASK_TEXT = (
    "Convert the checkpoint to the target format. Acceptance: running "
    "`python convert.py --check` exits 0 and prints the tensor count.\n\n"
    "Proceed."
)


def _plan(statuses, tmpdir):
    plan = TaskPlan(tmpdir)
    plan.create("T", [{"title": f"s{i}", "acceptance": [f"a{i}"]}
                      for i in range(len(statuses))])
    for i, st in enumerate(statuses, start=1):
        if st != "pending":
            plan.update_step(i, st)
    return plan


def _ctx(text, override="", llm=True, tool_name="", args=None):
    return GuardContext(tool_name=tool_name, tool_args=args or {},
                        tool_result=None, assistant_text=text,
                        override_reason=override, llm_responded=llm)


def _guard_with(statuses=("done",), task_text=TASK_TEXT):
    d = tempfile.mkdtemp()
    plan = _plan(statuses, d)
    g = VerificationGuard(plan=plan, task_text_fn=lambda: task_text)
    return d, g


def _reasons(verdicts):
    return [v.reason for v in verdicts]


def test_every_unmet_obligation_in_one_collection():
    """Unfinished step + no acceptance act + no oracle bound: all three, once."""
    d, g = _guard_with(statuses=("pending",))
    try:
        vs = g.check_all_pre(_ctx("[TASK_COMPLETE]"))
        got = _reasons(vs)
        assert "completion_with_unfinished_steps" in got, got
        assert "completion_without_acceptance_observation" in got, got
        assert "completion_without_oracle_bound" in got, got
        assert len(got) == 3, got
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_all_obligations_collected_would_fail_under_top1():
    """FALSIFIER: the base-class shape (check_pre only) loses the rest."""
    d, g = _guard_with(statuses=("pending",))
    try:
        one = g.check_pre(_ctx("[TASK_COMPLETE]"))
        many = g.check_all_pre(_ctx("[TASK_COMPLETE]"))
        assert one is not None and one.reason == "completion_with_unfinished_steps"
        assert len(many) == 3 and len(many) > 1, _reasons(many)
        # The single-verdict route provably cannot carry the other two slugs.
        assert one.reason not in (
            "completion_without_acceptance_observation",
            "completion_without_oracle_bound",
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_bound_but_unexercised_oracle_collected_with_acceptance_gap():
    """A BOUND oracle replaces the 'not bound' demand with 'not exercised'."""
    d, g = _guard_with(statuses=("done",))
    try:
        g._oracle_bound = {"premise_source": "differential",
                           "oracle": "reference impl diff",
                           "first_exercise": "python diff.py"}
        got = _reasons(g.check_all_pre(_ctx("[TASK_COMPLETE]")))
        assert "completion_without_acceptance_observation" in got, got
        assert "completion_without_oracle_exercise" in got, got
        assert "completion_without_oracle_bound" not in got, got
        assert len(got) == 2, got
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_no_obligation_yields_empty_list():
    d, g = _guard_with(statuses=("done",))
    try:
        g._acceptance_observed = True
        g._oracle_exercised_seen = True
        g._oracle_bound = {"premise_source": "provided", "oracle": "golden",
                           "first_exercise": "cat golden"}
        assert g.check_all_pre(_ctx("[TASK_COMPLETE]")) == []
        assert g.check_pre(_ctx("[TASK_COMPLETE]")) is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_override_reason_releases_only_the_steps_obligation():
    """An override quiets the overridable demand, the NON-overridable ones stay."""
    d, g = _guard_with(statuses=("pending",))
    try:
        got = _reasons(g.check_all_pre(
            _ctx("[TASK_COMPLETE]", override="checked: steps were folded in")))
        assert "completion_with_unfinished_steps" not in got, got
        assert "completion_without_acceptance_observation" in got, got
        assert "completion_without_oracle_bound" in got, got
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_non_completion_turns_are_untouched_by_the_collector():
    """Every other route keeps the first-failure shape (single pipeline)."""
    d, g = _guard_with(statuses=("pending",))
    try:
        # No sentinel, no tool -> not the completion arm at all.
        assert g.check_all_pre(_ctx("still working")) == []
        # A stale completion scanned out of history (llm_responded=False) is
        # deliberately not the completion arm either.
        assert g.check_all_pre(_ctx("[TASK_COMPLETE]", llm=False)) == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_registry_merges_the_collected_obligations_into_one_message():
    """End-to-end at the registry level: ONE message names every slug."""
    d, g = _guard_with(statuses=("pending",))
    try:
        reg = GuardRegistry()
        reg.register(g)
        verdict = reg.check_pre(_ctx("[TASK_COMPLETE]"))
        assert verdict is not None, "the registry must block"
        msg = verdict.message
        for slug in ("completion_with_unfinished_steps",
                     "completion_without_oracle_bound",
                     "completion_without_acceptance_observation"):
            assert slug in msg, (slug, msg)
        # The leading verdict is the hardest (non-overridable) one, not merely
        # the first evaluated: an overridable leader would let an unrelated
        # reason release the whole message.
        assert verdict.overridable is False, verdict.reason
    finally:
        shutil.rmtree(d, ignore_errors=True)

"""Plan-close NAMED-ARBITER obligation — the mechanism for the measured loss.

WHY IT EXISTS (measured, run 2026-10-27__03-51-02, trial dna-assembly, reward=0):
The task text said "primer3's oligotm should be considered the ground truth" and
bounded a Tm delta. The agent:
  * calibrated its OWN instrument (two Tm implementations agreed to 0.00 C),
  * at one point measured the arbiter's own reading: 70.07 vs 64.38 -> 5.69,
  * called that a bug in its verification script, "fixed" it by dropping the
    4nt overhang, got 0.61, and closed.
The grader used the arbiter's definition: 5.692 > 5 -> reward 0. It missed by
0.69 C. Nothing in the harness ever required it to point the task's OWN named
decider at the delivered file, so a self-consistent wrong definition shipped.

THE MECHANISM: plan_close (plan_update(action='complete')) refuses unless an
observation exists whose command runs the declared `named_arbiter` against the
declared `deliverable_path`. Deliberately NOT on [TASK_COMPLETE]: a refusal there
was the 2026-10-22 20+ round livelock. Here the refusal has an obvious next act.

WHAT IT CANNOT DO (stated, not papered over): it cannot know the quoted name is
the RIGHT decider. A name copied from the task text is a declaration a reviewer
can audit; the check catches "never looked, or looked and pointed it at a scratch
sample" — the measured loss — not every possible mislabel.
"""
import tempfile

from flagscale_agent.react.guard import GuardContext
from flagscale_agent.react.guard.verification import VerificationGuard
from flagscale_agent.react.plan import TaskPlan
from flagscale_agent.react.oracle import (
    arbiter_observation_matches,
    named_arbiter_obligation,
)

CORPUS = (
    "root@host:/app# oligotm -tp 1 -sc 1 -mv 50 -dv 4 -seq ACGTACGTACGT && cat primers.fasta\n"
    "ACGTACGTACGT 64.38\n"
    ">p1\nACGTACGTACGT\n"
)


def _plan():
    d = tempfile.mkdtemp()
    plan = TaskPlan(d)
    plan.create("T", [{"title": "s1", "acceptance": ["a1"]}])
    return plan


def _guard(plan, task_text='The task says "oligotm is the ground truth".'):
    g = VerificationGuard(plan=plan, task_text_fn=lambda: task_text)
    g._step_done_recheck_reminded = True
    return g


def _ctx(tool_args, corpus=CORPUS):
    return GuardContext(
        tool_name="plan_update",
        tool_args=tool_args,
        tool_result=None,
        assistant_text="",
        override_reason="",
        llm_responded=True,
        messages=[{"role": "tool", "content": corpus}],
    )


ORACLE = {
    "oracle": "primer3 oligotm",
    "premise_source": "provided",
    "first_exercise": "oligotm -seq ACGTACGTACGT",
    "named_arbiter": "oligotm",
    "deliverable_path": "primers.fasta",
}


# ---------------------------------------------------------------- pure helpers

def test_arbiter_none_is_a_declaration_not_an_obligation():
    assert named_arbiter_obligation({"named_arbiter": "none"}) is None
    assert named_arbiter_obligation({"named_arbiter": "  NONE "}) is None


def test_absent_key_obliges_nothing_here_but_is_refused_at_the_gate():
    # The helper returns None (nothing to name); the GATE distinguishes the
    # absent key from the explicit "none" and refuses the absent one.
    assert named_arbiter_obligation({}) is None


def test_malformed_arbiter_is_distinguishable_from_none():
    assert named_arbiter_obligation({"named_arbiter": ""}) == ""
    assert named_arbiter_obligation({"named_arbiter": 17}) == ""


def test_act_must_name_both_arbiter_and_delivered_path():
    ok = {"command": "oligotm -seq $(tail -1 primers.fasta)", "observed": "64.38"}
    assert arbiter_observation_matches(ok, "oligotm", "primers.fasta")
    # right tool, scratch sample -> the measured escape, must NOT match
    scratch = {"command": "oligotm -seq ACGTACGTACGT", "observed": "64.38"}
    assert not arbiter_observation_matches(scratch, "oligotm", "primers.fasta")
    # own script agreeing with itself -> must NOT match
    own = {"command": "python3 my_tm_check.py primers.fasta", "observed": "0.61"}
    assert not arbiter_observation_matches(own, "oligotm", "primers.fasta")


# ------------------------------------------------------------- the gate itself

def test_complete_blocks_when_arbiter_never_declared():
    g = _guard(_plan())
    decl = dict(ORACLE)
    decl.pop("named_arbiter")
    g._oracle_bound = decl
    vs = g.check_all_pre(_ctx({"action": "complete"}))
    assert [v.reason for v in vs] == ["plan_complete_arbiter_undeclared"]


def test_complete_allows_explicit_none():
    """'The task names no deciding tool' is an auditable declaration, not a hole."""
    g = _guard(_plan())
    decl = dict(ORACLE)
    decl["named_arbiter"] = "none"
    g._oracle_bound = decl
    assert g.check_all_pre(_ctx({"action": "complete"})) == []


def test_complete_blocks_when_arbiter_declared_but_never_run():
    g = _guard(_plan())
    g._oracle_bound = dict(ORACLE)
    vs = g.check_all_pre(_ctx({"action": "complete"}))
    assert [v.reason for v in vs] == ["plan_complete_arbiter_not_exercised"]


def test_complete_blocks_when_arbiter_ran_on_a_scratch_sample():
    """The dna-assembly escape in one assertion: the tool ran, but not on the file."""
    g = _guard(_plan())
    g._oracle_bound = dict(ORACLE)
    g._arbiter_act_seen = {"command": "oligotm -seq ACGTACGTACGT",
                           "observed": "64.38"}
    vs = g.check_all_pre(_ctx({"action": "complete"}))
    assert [v.reason for v in vs] == ["plan_complete_arbiter_wrong_target"]


def test_complete_passes_on_the_shipped_bytes():
    g = _guard(_plan())
    g._oracle_bound = dict(ORACLE)
    g._arbiter_act_seen = {
        "command": "oligotm -seq $(grep -v '>' primers.fasta | head -1)",
        "observed": "64.38",
    }
    assert g.check_all_pre(_ctx({"action": "complete"})) == []


def test_no_oracle_declaration_does_not_add_a_round_to_plan_close():
    """The gate must NOT demand an oracle where the run never bound one: that
    would cost a round on every plan close without catching the measured loss."""
    g = _guard(_plan())
    assert g.check_all_pre(_ctx({"action": "complete"})) == []


# ------------------------------------- the livelock this mechanism must not add

def test_act_supplied_on_the_complete_call_itself_is_honoured():
    """The natural shape: agent runs the arbiter, then closes with the reading on
    the same call. If the obligation were judged BEFORE the latch pass, this
    submission could never satisfy it and every retry would block forever."""
    g = _guard(_plan())
    g._oracle_bound = dict(ORACLE)
    args = {"action": "complete",
            "arbiter_act": {"command": "oligotm -seq $(tail -1 primers.fasta)",
                            "observed": "64.38"}}
    assert g.check_all_pre(_ctx(args)) == []


def test_unrecorded_observation_does_not_latch():
    """A command whose `observed` text the harness never recorded is not an act —
    the same non-forgeable corpus rule the other act fields obey."""
    g = _guard(_plan())
    g._oracle_bound = dict(ORACLE)
    args = {"action": "complete",
            "arbiter_act": {"command": "oligotm -seq $(tail -1 primers.fasta)",
                            "observed": "INVENTED-NUMBER-42"}}
    vs = g.check_all_pre(_ctx(args))
    assert [v.reason for v in vs] == ["plan_complete_arbiter_not_exercised"]


def test_act_latched_on_step_done_also_satisfies_plan_close():
    """The step_done call that carries the act may ALSO be blocked on its own
    fixed fields — that must not destroy the latched evidence (the measured
    10-round latch-starvation defect class). The assertion follows the existing
    latch-order contract: the step_done call blocks on formal fields, the latch
    still sets, and the subsequent plan close passes."""
    g = _guard(_plan())
    g._oracle_bound = dict(ORACLE)
    step = {"action": "step_done", "step_id": 1,
            "verification": ["oligotm ran on the delivered file"],
            "arbiter_act": {"command": "oligotm -seq $(tail -1 primers.fasta)",
                            "observed": "64.38"}}
    vs = g.check_all_pre(_ctx(step))
    # step_done still blocks on its own formal fields (convention etc.)...
    assert vs, "step_done should still be judged on its own formal fields"
    # ...but the corpus-verified act LATCHED despite that block...
    assert g._arbiter_act_seen is not None
    # ...and the same submission's act satisfies the plan close afterwards.
    assert g.check_all_pre(_ctx({"action": "complete"})) == []


def test_task_complete_sentinel_does_not_run_the_arbiter_gate():
    """Deliberate placement: the sentinel path keeps its old obligations. Parking
    this one there is the 2026-10-22 livelock (no next act, only an argument)."""
    g = _guard(_plan())
    g._oracle_bound = dict(ORACLE)
    ctx = _ctx({})
    ctx.assistant_text = "[TASK_COMPLETE]"
    reasons = [v.reason for v in g.check_all_pre(ctx)]
    assert "plan_complete_arbiter_not_exercised" not in reasons

"""The two clauses added 2026-11-15 by user ruling must survive verbatim.

Both clauses exist because of a measured behavioural defect, not a stylistic
preference, so each test names the defect it pins:

WHEN A STEP CLOSES (§9 RUNTIME) — the agent kept steps open until the end of the
task, then closed several in one stretch citing the same late output. A step
closed at the end is closed AFTER the mistake it existed to catch, so step_done
stops being a gate and becomes a formality. The clause has to state (a) the
closure rule, (b) the recognisable signature of back-filling, and (c) the
granularity repair — remove any one and the rule is advice rather than a test.

WHEN TO FAN OUT (§8) — the agent defaulted to serial work even when the plan had
independent branches. The clause must state BOTH directions: the three
conditions that justify fanning out (DAG, own output path, cost beats overhead)
AND the conditions that make serial correct. A one-sided clause ("prefer
parallel") would be as wrong as the original silence.
"""
from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC, prompt_len, PROMPT_BUDGET


def _p():
    return SYSTEM_PROMPT_STATIC


# ── clause: WHEN A STEP CLOSES ────────────────────────────────────────────────

def test_step_closes_at_its_own_gate_not_at_the_end():
    p = _p()
    assert "WHEN A STEP CLOSES" in p
    # the operative rule: the closure test is the step's own acceptance
    assert "closes at ITS OWN ACCEPTANCE GATE, not when the whole task is finished" in p
    # the failure being prevented must be named, or the rule reads as style advice
    assert "closed AFTER the mistake it existed to catch" in p


def test_step_closure_gives_a_concrete_run_time_test():
    p = _p()
    # a rule the agent cannot execute on the spot does not constrain behaviour
    assert "CAN YOU NAME THE COMMAND THIS STEP'S ACCEPTANCE REQUIRES" in p
    assert "If yes, close it now" in p


def test_step_closure_names_back_filling_as_the_tell():
    p = _p()
    assert "BACK-FILLING" in p
    # the tell must be observable, not introspective
    assert "several steps closed in one stretch" in p
    # W3 uniqueness is the mechanical symptom of a back-filled batch
    assert "command you already ran in an earlier step" in p


def test_step_closure_prescribes_the_granularity_repair():
    """Detecting back-filling is useless without the repair it implies."""
    p = _p()
    assert "reopen the GRANULARITY" in p
    assert "split too coarse" in p
    assert "A plan whose steps can only be verified at the end is a plan that gives you no gate at all." in p


def test_step_closure_handles_the_far_away_gate_case():
    """A long build/run must not become the excuse for a late closure."""
    p = _p()
    assert "still closes WHEN THE GATE PASSES, while the reading is fresh" in p


# ── clause: WHEN TO FAN OUT ───────────────────────────────────────────────────

def test_fan_out_requires_a_dag_not_just_parallelism_in_words():
    p = _p()
    assert "WHEN TO FAN OUT" in p
    # the user's ruling, verbatim in spirit: the plan must already be a DAG
    assert "it only pays when the plan is already a DAG" in p
    assert "do not CONSUME each other's output" in p


def test_fan_out_states_all_three_conditions():
    p = _p()
    assert "Fan out when ALL THREE hold" in p
    assert "OWN output path" in p
    assert "exceeds the overhead of dispatching it" in p


def test_serial_is_stated_as_correct_for_a_chain():
    """A clause that only says 'fan out' is as wrong as the original silence."""
    p = _p()
    assert "Stay serial when ANY of these holds" in p
    assert "one continuous reasoning chain" in p
    # serial must be framed as the right shape, not as timidity
    assert "Serial is not the timid default" in p


def test_fan_out_keeps_acceptance_with_the_parent():
    p = _p()
    assert "poll_tasks(action='check') runs the acceptance YOURSELF" in p
    assert "never let a worker's self-report stand as evidence" in p


def test_fan_out_justifies_itself_by_total_wall_clock():
    p = _p()
    assert "it is TOTAL WALL-CLOCK" in p
    # and must reject the activity-as-thoroughness reading
    assert 'never "more agents looks thorough"' in p


# ── the budget ratchet ────────────────────────────────────────────────────────

def test_prompt_fits_the_raised_budget():
    n = prompt_len()
    assert n <= PROMPT_BUDGET, f"prompt is {n} chars, budget {PROMPT_BUDGET}"
    # raised to 45000 by user ruling 2026-11-15; never silently lowered
    assert PROMPT_BUDGET >= 45_000, (
        "41000 -> 45000 was a user ruling to seat WHEN A STEP CLOSES and WHEN TO "
        "FAN OUT; lowering it would silently drop one of them"
    )

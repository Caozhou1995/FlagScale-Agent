# Self-correction capability: a non-self-satisfiable pre-submit verification gate

## Failure class
Recurring across TB2.1 runs: the agent's own self-check PASSES, yet the grader
FAILS — because the self-check shares the agent's own (wrong) premise and never
touches the interface the checker actually inspects (documented 4 instances:
dna-assembly clamp direction, install-win socket path, ...).

## The structural gap (why "just write more prose" fails)
The completion boundary already has three arms (verification.py `check_pre`
L1526-1590):
  1. `completion_with_unfinished_steps`
  2. `completion_without_graded_reading`
  3. `completion_without_acceptance_observation`
Arms 2 and 3 exist to force the agent to CONFRONT its grader before declaring
done. But arm 3's release predicate is SELF-SATISFIABLE:

    _acceptance_act_recorded(act, observed):     # verification.py L900-921
        if _is_real_str(act.get("unobservable")):
            return True          # <-- any non-empty string; ZERO corpus check

An agent at the wall writes `acceptance_act={"unobservable": "grader is hidden"}`
and clears the gate without having TRIED to observe anything. The suite even
freezes this loophole: tests/test_verification_guard_completion_closed.py L426
asserts `_acceptance_act_recorded({"unobservable": "hidden corpus"}, None) is True`.
Declaring a dimension invisible is free; only a real attempt distinguishes
"genuinely invisible" from "never looked".

## Mechanism (ONE point)
Make the INCONCLUSIVE (`unobservable`) form cost an ATTEMPT, exactly as the PASS
form costs an OBSERVATION: BOTH forms must carry `command` + `observed`, and the
`observed` text must appear in the session's non-forgeable tool-result corpus
(`_evidence_observed` -> `_observed_corpus`). "Declaration only" -> block;
"declaration + a recorded probe that tried and could not observe" -> INCONCLUSIVE.

    {"command": C, "observed": O}                       -> PASS  (corpus-checked, unchanged)
    {"unobservable": W, "command": C, "observed": O}    -> INCONCLUSIVE (NOW corpus-checked)
    {"unobservable": W}                                 -> BLOCK (was: free pass)

Fail-open preserved: when the corpus source is unavailable (None), the predicate
returns True, so a task whose session log cannot be read is not spuriously
blocked. The `_override_reason` escape remains for a genuinely ungraded task.

## Why this is the self-correction lever, not just a stricter gate
The guard cannot force the agent to LOOP; it can force the agent to LOOK. This
change converts the boundary's cheapest escape from a DECLARATION into an ACTION:
to close, the agent must run a command aimed at its acceptance dimension and paste
its real output. If that probe surfaces the acceptance signal the agent had missed
(the tests, the fixture path, the reference value), the agent now has what it
needs to CORRECT before completion. The failure mode moves from "declared done
without looking" to "looked, then either fixed or recorded an honest gap".

## Falsifiable prediction
The completion boundary will no longer release on a bare `{"unobservable": ...}`.
Mutation probe: revert `_acceptance_act_recorded` to the free-pass form and the
new regression test FAILS; with the change it PASSES.

## Oracle
FlagScale-Agent's own pytest suite (an external, pre-existing consumer of this
guard) + a mutation probe. No task/grader literal named (leak guard
tests/test_no_harness_grader_leak.py stays green).

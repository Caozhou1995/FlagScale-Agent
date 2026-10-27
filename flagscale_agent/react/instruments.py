"""Scientific instruments as FORCED declarations.

WHY THIS MODULE EXISTS (measured, not speculative).

Across two full TB2.1 runs every mechanism that could be *ignored* failed to
change behaviour, and every mechanism that *blocked the next action* changed it.
The negative-doctrine list reached 90% uptake with task-specific content and the
agent still churned to death. So the lever is not more prose: it is a
declaration the agent cannot proceed without, placed at the moment a decision is
made.

The four instruments below are the ones a scientist uses when the grader is
invisible -- they need no access to the answer:

1. CALIBRATION  -- before trusting an instrument, run it on a case whose answer
   you already know. Measured failure this closes: my own probe searched only the
   first record of a multi-record input and reported a wrong-direction value for
   an element whose true value was known. The ONLY way that was caught was
   re-running it against a case with a known answer. An agent that never
   calibrates ships its own bug as evidence -- a self-scored PASS on its own
   wrong definition.

2. DIFFERENTIAL -- compute the same quantity down two paths that do NOT share
   source, and report both numbers. When they agree you have bought confidence
   cheaply. When they disagree the finding is not "my code has a bug", it is
   "this quantity depends on how I define it" -- a fact about the PROBLEM.
   Measured failure this closes: an agent measured a quantity over only the
   region it chose while the named tool's own scope included an extra prefix,
   and scored the result within bounds. It never computed the second number, so
   it never learned the first was *chosen*, not *measured*.

3. PRE-REGISTRATION -- write the expected number BEFORE running, then settle it
   hit/miss. A prediction cannot be re-explained after the fact. Measured on
   this very harness: predicted "~3350 passed", got 3329 -> credible; predicted
   "3 failures", got 1 -> immediately knew the model was wrong. The complement
   also measured: one long-running prediction was never settled at all after the
   run was stopped, which is precisely the drift this prevents.

4. NEGATIVE CONTROL -- run the check where it MUST fail. An instrument that has
   never been observed going red has not been shown able to detect failure.
   This is the last line of defence when there is no known-positive case: if
   calibration could not be done, the instrument must at minimum be proven to
   fire on a bad input, or its PASS is worthless.

SCOPE DISCIPLINE. These apply to B-class decisions -- the ones the outside world
never answers (which segment to measure, which definition to use, which order to
assemble in). They are NOT applied to A-class work (compile errors, library
return values, test red/green) where feedback is free and self-correcting, and
they are NOT applied to C-class assumptions (which get a separate staleness
treatment). Forcing them everywhere would be the "abuse of guards" failure.
"""

from __future__ import annotations

#: The three ways a calibration can come out. `no_known_case` is a legal answer
#: but it CHANGES what else is required (see calibration_needs_negative_control),
#: because an uncalibrated instrument is the one situation where a self-report of
#: PASS carries no information at all.
CALIBRATION_MATCHES = ("yes", "no", "no_known_case")

#: Outcome of a differential comparison. `cannot_compute` is legal and is
#: recorded as an explicit gap rather than silently treated as agreement.
DIFFERENTIAL_OUTCOMES = ("yes", "no", "cannot_compute")

_MISSING_CALIBRATION = (
    "`calibration` is REQUIRED at plan creation and is missing.\n"
    "\n"
    "Before you rely on any instrument to decide whether your output is right, "
    "run it ONCE on a case whose answer you already know, and record what it "
    "said. You authored the instrument, so it shares your blind spot: an "
    "uncalibrated check that reports PASS tells you only that it agrees with "
    "itself.\n"
    "\n"
    "Pass plan_create(calibration={...}) with these keys:\n"
    '  "instrument"  : the exact check/command you will rely on (not a category)\n'
    '  "known_case"  : a case whose answer is ALREADY known and external to the\n'
    "                  instrument -- a value you read from the task's own data,\n"
    "                  a prior observed result, a case the environment supplies\n"
    '  "observed"    : what the instrument ACTUALLY reported on that case\n'
    '  "matches"     : "yes" | "no" | "no_known_case"\n'
    "\n"
    "If no such case exists, say \"no_known_case\" -- that is an honest answer "
    "and it is recorded as a gap, but it obliges you to show a NEGATIVE CONTROL "
    "later (the instrument must be seen to FIRE on a bad input). If your "
    "instrument disagrees with the known case, it is broken: fix it now, because "
    "every PASS it produces afterwards is noise."
)

_CALIBRATION_MALFORMED = (
    "`calibration` is malformed. It must be an object with "
    '"instrument" (str), "known_case" (str), "observed" (str) and "matches" '
    '(one of "yes" | "no" | "no_known_case"). Pass plan_create(calibration={...}).'
)

_CALIBRATION_CONTRADICTED = (
    "`calibration.matches` is \"no\": your instrument disagreed with a case whose "
    "answer you already knew.\n"
    "\n"
    "That is not a step to note and move past -- it means the instrument is "
    "wrong, so every PASS it produces from here is worthless. Fix the instrument "
    "and re-run it on the same known case until it reproduces that answer, then "
    "declare matches=\"yes\" with the new observed value. Do not proceed to use "
    "it on the real problem while it still disagrees with a known answer."
)

_MISSING_DIFFERENTIAL = (
    "This step declares a NAMED convention ({name}) with conflict=\"no\", so its "
    "output rests on a quantity you DEFINED -- and the world will never tell you "
    "if that definition is the one that decides the outcome.\n"
    "\n"
    "Compute the SAME quantity a second time down a path that does not share "
    "source with the first, and report both numbers. Then you learn one of two "
    "things, and both are worth the cost:\n"
    "  * they agree  -> the value does not depend on your definition (cheap "
    "confidence, and you may proceed);\n"
    "  * they differ -> the value is DEFINED BY YOUR CHOICE, which is a fact "
    "about the problem, not a bug in your code. Pick the definition from an "
    "external authority (the task's own wording, the named tool's default, a "
    "supplied reference) and say which.\n"
    "\n"
    "Pass step_done(differential={...}) with:\n"
    '  "quantity"         : what is being measured\n'
    '  "path_a"           : method A and the value it produced\n'
    '  "path_b"           : method B (mechanically different) and its value\n'
    '  "delta"            : the difference between them\n'
    '  "within_tolerance" : "yes" | "no" | "cannot_compute"\n'
    '  "tolerance_basis"  : where the tolerance comes from, or "task states none"\n'
    "\n"
    "If within_tolerance is \"no\" you must either name the external authority "
    "that decides which path is right, or record this as an unverified "
    "assumption -- you may not deliver a number that depends on an arbitrary "
    "choice while calling it verified."
)

_DIFFERENTIAL_MALFORMED = (
    "`differential` is malformed. It must be an object with "
    '"quantity", "path_a", "path_b", "delta", "within_tolerance" '
    '("yes" | "no" | "cannot_compute") and "tolerance_basis".'
)

_DIFFERENTIAL_UNRESOLVED = (
    "`differential.within_tolerance` is \"no\" -- the two paths disagree and the "
    "difference is outside tolerance. That means the quantity is NOT determined "
    "by the problem; it is determined by which definition you picked.\n"
    "\n"
    "Resolve it before this step can close. One of:\n"
    "  (a) name the external authority that decides the definition (the task's "
    "own construction, the named tool's documented default, a supplied "
    "reference) and re-run with that definition;\n"
    "  (b) record it in `assumptions` as an unverified premise carrying the "
    "world in which the other definition is the right one and why that world is "
    "impossible or cheap to be wrong about.\n"
    "\n"
    "Delivering a value that flips depending on an arbitrary choice is exactly "
    "the failure this gate exists for: it looks verified and is not."
)

_MISSING_NEGATIVE_CONTROL = (
    "Your calibration declared matches=\"no_known_case\" -- the instrument has "
    "never been checked against a case with a known answer. An uncalibrated "
    "instrument that reports PASS has told you nothing: it has only ever been "
    "observed agreeing with itself.\n"
    "\n"
    "So show it can go RED, once, now. Run the SAME check on an input that MUST "
    "fail it, and record what it did:\n"
    "  step_done(negative_control={{\n"
    '    "check"           : the check you are relying on,\n'
    '    "must_fail_input" : an input the correct check rejects,\n'
    '    "observed"        : what it actually reported ("FAILED"/"rejected" ...)\n'
    "  }}\n"
    "\n"
    "If the check does NOT fire on the bad input, it cannot detect failure at "
    "all -- it is not evidence, whatever it says about your real output."
)

_NEGATIVE_CONTROL_DID_NOT_FIRE = (
    "`negative_control.observed` does not show the check rejecting the bad input. "
    "An instrument that stays green on an input it should reject cannot "
    "distinguish right from wrong, so its PASS on the real output is not "
    "evidence. Fix the check until the control fires, then report the real run."
)

_MISSING_PREDICTION = (
    "`{action}` requires a `prediction`: write the number/outcome you EXPECT "
    "before you run, so the result cannot be re-explained after the fact. "
    'Pass prediction={{"statement": ..., "expected": ...}}.'
)

UNSETTLED_PREDICTIONS_MSG = (
    "You cannot close the plan with unsettled predictions. Each of these was "
    "recorded BEFORE a run and never settled:\n{items}\n"
    "\n"
    "Settle each with plan_update(action='settle', prediction_id=<id>, "
    'outcome="hit"|"miss"|"not_run"). A prediction left dangling is worse than '
    "no prediction: it looks like foresight while carrying none of its cost. "
    "\"not_run\" is a legal outcome (say so honestly when the run was cut short)."
)


def _is_real_str(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


def check_calibration(cal):
    """Validate a plan-level calibration declaration.

    Returns ``(reason, message)`` when it must be refused, else None.
    The explicit ``no_known_case`` answer is accepted: it is an owned claim, and
    it changes what happens later (negative control becomes mandatory) rather
    than being a silent skip.
    """
    if cal is None:
        return ("calibration_missing", _MISSING_CALIBRATION)
    if not isinstance(cal, dict):
        return ("calibration_not_dict", _CALIBRATION_MALFORMED)
    for f in ("instrument", "matches"):
        if f not in cal:
            return ("calibration_field_missing", _CALIBRATION_MALFORMED)
    matches = str(cal.get("matches") or "").strip().lower()
    if matches not in CALIBRATION_MATCHES:
        return ("calibration_bad_matches", _CALIBRATION_MALFORMED)
    if not _is_real_str(cal.get("instrument")):
        return ("calibration_blank_instrument", _CALIBRATION_MALFORMED)
    # `no_known_case` legitimately has no case or observation to report; every
    # other answer must carry both, or the declaration is a formality.
    if matches != "no_known_case":
        if not (_is_real_str(cal.get("known_case"))
                and _is_real_str(cal.get("observed"))):
            return ("calibration_blank_fields", _CALIBRATION_MALFORMED)
    if matches == "no":
        return ("calibration_contradicted", _CALIBRATION_CONTRADICTED)
    return None


def calibration_needs_negative_control(cal) -> bool:
    """True when the instrument was never checked against a known answer."""
    if not isinstance(cal, dict):
        return False
    return str(cal.get("matches") or "").strip().lower() == "no_known_case"


def check_differential(diff):
    """Validate the alternate-path recomputation for a derived quantity."""
    if diff is None:
        return ("differential_missing", _MISSING_DIFFERENTIAL)
    if not isinstance(diff, dict):
        return ("differential_not_dict", _DIFFERENTIAL_MALFORMED)
    for f in ("quantity", "path_a", "path_b", "delta",
              "within_tolerance", "tolerance_basis"):
        if f not in diff:
            return ("differential_field_missing", _DIFFERENTIAL_MALFORMED)
    outcome = str(diff.get("within_tolerance") or "").strip().lower()
    if outcome not in DIFFERENTIAL_OUTCOMES:
        return ("differential_bad_outcome", _DIFFERENTIAL_MALFORMED)
    if not (_is_real_str(diff.get("quantity")) and _is_real_str(diff.get("path_a"))
            and _is_real_str(diff.get("path_b"))):
        return ("differential_blank_paths", _DIFFERENTIAL_MALFORMED)
    if outcome == "no":
        return ("differential_unresolved", _DIFFERENTIAL_UNRESOLVED)
    return None


def differential_satisfied(diff) -> bool:
    """True when a differential declaration is present and fully resolved."""
    return check_differential(diff) is None


def check_negative_control(nc):
    """Validate that a self-authored check was observed going RED."""
    if nc is None:
        return ("negative_control_missing", _MISSING_NEGATIVE_CONTROL)
    if not isinstance(nc, dict):
        return ("negative_control_not_dict", _MISSING_NEGATIVE_CONTROL)
    if not (_is_real_str(nc.get("check"))
            and _is_real_str(nc.get("must_fail_input"))
            and _is_real_str(nc.get("observed"))):
        return ("negative_control_blank_fields", _MISSING_NEGATIVE_CONTROL)
    if not negative_control_fired(nc):
        return ("negative_control_did_not_fire", _NEGATIVE_CONTROL_DID_NOT_FIRE)
    return None


def negative_control_fired(nc) -> bool:
    """Whether the recorded negative control actually reports a rejection.

    Deliberately generous about WORDING (the agent writes it) but strict about
    there being a real negative signal rather than a bare "ok".
    """
    if not isinstance(nc, dict):
        return False
    obs = str(nc.get("observed") or "").strip().lower()
    if not obs:
        return False
    markers = (
        "fail", "reject", "mismatch", "not match", "does not match",
        "nonzero", "non-zero", "exit 1", "false", "error", "no ", "0 match",
        "invalid", "absent", "missing", "differs", "not equal", "!=",
    )
    return any(m in obs for m in markers)


def check_prediction(pred):
    """Validate a pre-registration made BEFORE an expensive/observational run."""
    if pred is None:
        return "missing"
    if not isinstance(pred, dict):
        return "malformed"
    if not _is_real_str(pred.get("statement")):
        return "malformed"
    return None


__all__ = [
    "CALIBRATION_MATCHES",
    "DIFFERENTIAL_OUTCOMES",
    "check_calibration",
    "calibration_needs_negative_control",
    "check_differential",
    "differential_satisfied",
    "check_negative_control",
    "negative_control_fired",
    "check_prediction",
    "UNSETTLED_PREDICTIONS_MSG",
]

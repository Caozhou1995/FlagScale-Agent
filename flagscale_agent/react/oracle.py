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

"""Independent-oracle discipline, bound at plan-framing time.

The loss signature this targets: an agent's SELF-authored oracle shares the
agent's own wrong premise, so it green-washes the error instead of catching it
(recurring convention/form failures -- a base address, an area-vs-height unit, a
serialization form, a clamp direction, a call signature, a container type).

HONEST SCOPE. This module checks the DECLARED SHAPE of an oracle: that the agent
named one of the external-premise SOURCES and supplied both an oracle and its
first runnable exercise. A string label is not evidence -- an agent can mislabel
its own pipeline as 'provided'. No predicate over a declaration can verify
externalness; only a human/reviewer reading the named artifact, or a runnable
differential the agent actually executes, can. So this is a DECLARATION gate,
not an enforcement of independence, and it is named and documented as such. Its
value is at framing time: it forces the agent to name a specific external
artifact up front, which is auditable, instead of silently trusting its own
pipeline.

Generic by construction: no parser, corpus, library, or task is named -- naming a
concrete oracle would hand the agent the checker (see
``tests/test_no_harness_grader_leak.py``). Only the three premise sources that are
EXTERNAL to the agent's own pipeline are named.
"""

# Premise sources external to the agent's own implementation.
#   differential -- two independent implementations; diff the two.
#   provided     -- a reference/golden artifact the task or environment supplies.
#   consumer     -- the exact parse/serialize path the CONSUMER uses, replayed.
PREMISE_SOURCES = ("differential", "provided", "consumer")

# Fields a declaration may carry; any other key is rejected (F6): an incriminating
# annex ("note: my own script") must not ride along a passing declaration.
_DECLARATION_KEYS = frozenset(("oracle", "premise_source", "first_exercise"))

# The FRAME acceptance item the harness injects into every plan. It carries the
# framing-time discipline as ORDINARY acceptance text (worked by the agent like
# any other criterion); it is NOT a separate guard and makes no claim of
# mechanical enforcement -- see the module docstring for why a declaration cannot
# be enforced as proof of independence.
ORACLE_ACCEPTANCE_ITEM = (
    "Independent-oracle binding: name the oracle that will decide this task, and "
    "classify its PREMISE -- it must be DECLARED as coming from a source YOU did "
    "not author, one of: 'differential' (two independent implementations; diff "
    "the two), 'provided' (a reference/golden artifact the task or environment "
    "supplies), or 'consumer' (the exact parse/serialize path the CONSUMER uses, "
    "replayed). A self-authored oracle shares your premise and only confirms it -- "
    "reject it. Naming the source is a DECLARATION, not proof: point to the "
    "specific external artifact so a reader can check it, rather than relabeling "
    "your own pipeline. Then write the oracle's FIRST runnable exercise (the exact "
    "command/check) into the acceptance of the FIRST load-bearing step (list that "
    "step first, or pass its index), so the acceptance->verification channel "
    "forces its evidence to exist early, while the premise is still cheap to "
    "revise. If no external-premise oracle exists for this task, say so "
    "explicitly -- that is a load-bearing gap, not a pass."
)


# The FRAME acceptance item that targets the COMPLEMENT of the oracle gate. The
# oracle gate forces ONE axis to be decided by an external premise -- but an
# agent that guards only that one axis can still ship with every OTHER stated
# constraint unverified (its own report literally says so). Observed loss:
# a fit task where the agent verified one peak's position and shipped with the
# other three parameters "NOT VERIFIED", failing a tolerance the task stated.
# So this item demands that EVERY numeric/form constraint the task TEXT states be
# COMPILED into a runnable assertion executed against the artifact before
# delivery -- not declared in prose. It is generic: it names no task, unit, or
# threshold (see tests/test_no_harness_grader_leak.py).
STATED_CONSTRAINT_ITEM = (
    "Stated-constraint coverage: the independent-oracle item above fixes ONE "
    "axis; it does NOT discharge the others. Re-read the task text and ENUMERATE "
    "every numeric or form constraint it states (a tolerance, a range, an upper "
    "or lower bound, a count, a field name, a type, an order, a required file). "
    "For EACH, COMPILE a runnable assertion against the delivered artifact and "
    "actually RUN it before you complete -- a prose 'NOT VERIFIED' note is not "
    "coverage. The trap this closes: you verify the one quantity your oracle "
    "happens to cover, leave the rest unchecked, and a stated bound you never "
    "tested fails at the grader. When a stated constraint has no expected VALUE, "
    "the task still states its FORM (bound/direction/type/order) -- assert THAT. "
    "List the enumerated constraints and each assertion's observed result in the "
    "plan step; a constraint you cannot compile is an explicit, named gap, not a "
    "silent omission. "
    "For each such assertion state WHERE the measured quantity's DEFINITION comes "
    "from: the task's own construction (the mechanical order it states, or the "
    "tool it NAMES and that tool's own default path), NOT a definition you chose "
    "because it was easy to measure on the objects you happen to hold (anchoring "
    "on your template, counting from your boundary, skipping the prompt the named "
    "library would add). When two defensible definitions exist, compute BOTH and "
    "diff the numbers before delivering; if they disagree, the grader-side "
    "definition wins, and a passing self-check built on the other one is not "
    "evidence. Observed loss: an agent's self-check reported PASS on a quantity "
    "it defined by template anchoring (15-17 nt) while the grader's construction "
    "gave 56 nt and the stated bound 15-45 failed; another delivered the "
    "no-instruction variant of a value when the task NAMED the library whose "
    "default applies the instruction."
)


# The FRAME acceptance item that targets the AMBIGUITY the oracle gate assumes
# away: the oracle gate demands an external premise when one EXISTS, but a spec
# that leaves a choice OPEN (a convention, a unit, a layout, a direction, a
# default, an edge case, an interface signature) is exactly where the agent's
# self-authored oracle shares the wrong premise and green-washes it. Observed
# loss: an ambiguous interface whose parameter semantics the text never pinned --
# the agent picked a reading by internal preference, self-verified, and failed
# the grader's expected convention. So this item forces the OPEN axis to be
# ANCHORED on the strongest EXTERNAL authority the task/environment makes
# available and PROBED by running against it -- never settled by the agent's own
# reasoning. Generic: it names no task, unit, library, or threshold (see
# tests/test_no_harness_grader_leak.py).
AMBIGUITY_ANCHOR_ITEM = (
    "Ambiguity anchoring: when the task's own text leaves a choice OPEN -- a "
    "convention, a unit, a layout, a direction, a default, a naming, an edge "
    "case, or an interface signature the text does not pin -- do NOT resolve it "
    "by your own preference or an internal consistency argument. NAME the "
    "ambiguity axis in one line, then ANCHOR the choice on the STRONGEST EXTERNAL "
    "authority the task or environment makes available and RUN a probe against "
    "it: the real library/API the task names (call it and read what it returns), "
    "the framework or tool the task is built on (exercise its actual behavior), "
    "or a reference/golden artifact the environment supplies. A choice confirmed "
    "only by your own reasoning, or by a script you wrote to agree with itself, "
    "is the same wrong premise restated -- it cannot catch an error you already "
    "hold. Record the probe and its real observed output, not just the "
    "conclusion. If no external authority can decide the axis, say so explicitly "
    "and name the residual risk -- an honest gap, not a silent guess."
)


def declares_external_premise(declaration):
    """Return True iff ``declaration`` has the SHAPE of an external-premise oracle.

    This validates the DECLARED shape only: ``premise_source`` is one of
    ``PREMISE_SOURCES``, both ``oracle`` and ``first_exercise`` are non-empty
    strings, and no unknown keys are present. It does NOT verify the premise is
    actually external -- a label is not evidence (see the module docstring).

    ``declaration`` is a mapping with keys:
      ``oracle``        -- the oracle description (str)
      ``premise_source``-- one of ``PREMISE_SOURCES`` (declared external)
      ``first_exercise``-- exact command/check that runs it (str)
    """
    if not isinstance(declaration, dict):
        return False
    if set(declaration) - _DECLARATION_KEYS:
        # unknown key present -- reject the annex
        return False
    source = declaration.get("premise_source")
    if source not in PREMISE_SOURCES:
        return False
    for key in ("oracle", "first_exercise"):
        val = declaration.get(key)
        if not isinstance(val, str) or not val.strip():
            return False
    return True


def build_oracle_binding_line(oracle, first_exercise):
    """Render the acceptance line to place on the FIRST load-bearing step."""
    return (
        f"Oracle: {oracle.strip()} -- first exercise (run and read): "
        f"{first_exercise.strip()}"
    )

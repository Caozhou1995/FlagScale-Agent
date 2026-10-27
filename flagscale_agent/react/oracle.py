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
serialization form, a boundary convention, a call signature, a container type).

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
#
# `named_arbiter` / `deliverable_path` extend the declaration with the ONE thing a
# self-authored oracle structurally cannot supply: WHICH external tool the task
# text itself names as the decider, and which delivered bytes that tool must run
# on. The loss signature this targets: an agent calibrates its own instrument
# (its two implementations of the quantity agree to 0.00) but never points the
# task's named arbiter at the delivered artifact, so the definition it chose --
# not the arbiter's -- decides. Observed: the run recorded the arbiter's own
# reading (a quantity the task bounded) mid-run, then explained it away as its
# own script bug and shipped one unit of that delta over the bound.
_DECLARATION_KEYS = frozenset(
    ("oracle", "premise_source", "first_exercise", "named_arbiter", "deliverable_path")
)

# The value an agent writes when the task text names NO deciding tool at all.
# Explicitly recorded as a declaration (not an omission) so "the task named
# nothing" is auditable instead of silently equal to "I did not look".
ARBITER_NONE = "none"

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
    "explicitly -- that is a load-bearing gap, not a pass. "
    "NAMED ARBITER (declare it on the oracle): when the task text itself names "
    "who decides a disputed quantity -- a sentence like 'consider X's output the "
    "ground truth', 'must match the reference implementation', or 'verify with "
    "tool Y' -- record that tool as the oracle's named_arbiter (quote it from the "
    "task text; it is in the text you can read, it is NOT the hidden checker), "
    "and the file you will deliver as deliverable_path. Before closing the plan, "
    "run THAT tool against YOUR delivered file and record the reading -- an "
    "instrument that only your own code calibrates agrees with your own "
    "definition of the quantity by construction, and the arbiter's definition is "
    "what decides. If the task text names no deciding tool, declare "
    "named_arbiter: 'none' -- the point is that you LOOKED; an omitted key reads "
    "as never having looked."
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
    "it had defined by anchoring on its own template, while the grader's "
    "construction of that same quantity produced a value outside the stated "
    "bound; another delivered the plain variant of a value when the task NAMED "
    "the library whose default path applies the additional handling."
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


def named_arbiter_obligation(declaration):
    """Return the arbiter this declaration OBLIGES, or None when it obliges none.

    Three outcomes, deliberately distinguishable:

      * ``None``  -- no obligation. Either no declaration exists, or the agent
        explicitly recorded ``named_arbiter: "none"`` (the task text names no
        deciding tool). The second case is a DECLARATION, not an omission: the
        agent had to look at the task text and say so, which is auditable.
      * a non-empty str -- the task text NAMES this tool as the decider, so the
        completion boundary must find an observation that ran THIS tool on the
        DELIVERED artifact. Nothing here checks that the name is the right one;
        the arbiter name is quoted from the task text and a reviewer can read it.
      * ``""`` (empty/malformed) -- the agent wrote the key but left it blank or
        non-string. Returned as ``""`` so a caller can refuse it as malformed
        rather than silently treating it as a declaration of "none".
    """
    if not isinstance(declaration, dict):
        return None
    if "named_arbiter" not in declaration:
        return None
    raw = declaration.get("named_arbiter")
    if not isinstance(raw, str):
        return ""
    name = raw.strip()
    if not name:
        return ""
    if name.lower() == ARBITER_NONE:
        return None
    return name


def arbiter_observation_matches(act, arbiter, deliverable_path=""):
    """Does this recorded act run ``arbiter`` against ``deliverable_path``?

    Both halves are required, and each closes a distinct escape:

      * the ARBITER NAME must appear in the command. An act that runs the agent's
        own verifier script never satisfies the obligation -- that is the whole
        point (its script agrees with its own definition by construction).
      * the DELIVERABLE PATH must appear in the command when the declaration
        named one. Otherwise the arbiter runs on a scratch sample the agent
        constructed (a hand-typed sequence, a re-derived substring), which
        measures the arbiter's arithmetic and not the shipped bytes.

    Matching is plain case-insensitive substring on the command text. That is
    deliberately weak and honest: it catches "never ran the arbiter", not a
    determined disguise.
    """
    if not isinstance(act, dict):
        return False
    cmd = act.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return False
    low = cmd.lower()
    if arbiter.lower() not in low:
        return False
    path = (deliverable_path or "").strip()
    if path and path.lower() not in low:
        return False
    return True


# Sentinel an agent writes when it can defend that the quantity has exactly ONE
# defensible definition (so a second reading would be a copy, not a rival).
RIVAL_NONE = "none"


def rival_reading_obligation(act):
    """Does this recorded act carry the SECOND reading of the same quantity?

    Three outcomes, mirroring ``named_arbiter_obligation``'s shape so the two
    obligations are auditable the same way:

      * ``""``  -- the key is absent or blank: the agent ran ONE reading. That is
        the measured loss this exists for -- a command that names the delivered
        file (so the weaker path check passes) while the rule it uses to CUT the
        measured substring out of those bytes is the agent's own choice, and the
        consumer's construction of the same quantity differs by exactly that cut.
        The obligation is unmet, not malformed: the message teaches the shape.
      * ``None``-- the act explicitly records ``rival_command: "none"``, i.e. the
        agent looked for a second defensible definition and can defend there is
        none. A DECLARATION, auditable, not a silent omission.
      * a dict  -- ``{command, observed}``: a genuinely different extraction rule
        run over the SAME delivered bytes.

    Nothing here judges which reading is right. It forces both to exist, so the
    disagreement becomes an observation the agent must look at instead of a
    definition it never had to name.
    """
    if not isinstance(act, dict):
        return ""
    raw = act.get("rival_command")
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return ""
        if txt.lower() == RIVAL_NONE:
            return None
        return {"command": txt, "observed": act.get("rival_observed")}
    if isinstance(raw, dict):
        return raw
    return ""


def rival_reading_distinct(act, rival):
    """Is ``rival`` a DIFFERENT extraction from the primary command?

    A copy is not a rival: two commands that cut the measured substring the same
    way agree by construction and confirm nothing. The comparison is textual over
    the command bodies with whitespace collapsed -- deliberately weak and honest
    (it catches "you pasted the same command twice", not a determined disguise),
    matching ``arbiter_observation_matches``' stated limit.
    """
    if not isinstance(act, dict) or not isinstance(rival, dict):
        return False
    a = " ".join(str(act.get("command") or "").split())
    b = " ".join(str(rival.get("command") or "").split())
    if not b:
        return False
    return a != b


def build_oracle_binding_line(oracle, first_exercise):
    """Render the acceptance line to place on the FIRST load-bearing step."""
    return (
        f"Oracle: {oracle.strip()} -- first exercise (run and read): "
        f"{first_exercise.strip()}"
    )

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

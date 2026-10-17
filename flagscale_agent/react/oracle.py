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

One job: force the oracle's PREMISE to come from somewhere the agent did not
author, and bind its first runnable exercise into the acceptance of the FIRST
load-bearing step -- so the acceptance -> verification channel makes the oracle's
evidence exist early, while the premise is still cheap to revise.

Generic by construction: no parser, corpus, library, or task is named -- naming a
concrete oracle would hand the agent the checker (see
``tests/test_no_harness_grader_leak.py``). Only the three premise sources that are
EXTERNAL to the agent's own pipeline are named.
"""

# Premise sources external to the agent's own implementation.
#   differential -- a second, independent implementation; diff the two.
#   provided     -- a reference/golden artifact the task or environment supplies.
#   consumer     -- the exact parse/serialize path the CONSUMER uses, replayed.
PREMISE_SOURCES = ("differential", "provided", "consumer")

# The FRAME acceptance item the harness injects into every plan. It is enforced
# by the ordinary step_done evidence gate (same channel as the other frame
# items), so it needs no separate guard.
ORACLE_ACCEPTANCE_ITEM = (
    "Independent-oracle binding: name the oracle that will decide this task, and "
    "classify its PREMISE -- it must come from a source YOU did not author, one "
    "of: 'differential' (two independent implementations; diff the two), "
    "'provided' (a reference/golden artifact the task or environment supplies), "
    "or 'consumer' (the exact parse/serialize path the CONSUMER uses, replayed). "
    "A self-authored oracle shares your premise and only confirms it -- reject it. "
    "Then write the oracle's FIRST runnable exercise (the exact command/check) "
    "into the acceptance of the FIRST load-bearing step, so the "
    "acceptance->verification channel forces its evidence to exist early, while "
    "the premise is still cheap to revise. If no external-premise oracle exists "
    "for this task, say so explicitly -- that is a load-bearing gap, not a pass."
)


def is_premise_external(declaration):
    """Return True iff ``declaration`` names an EXTERNAL-premise oracle bound to a
    first runnable exercise.

    ``declaration`` is a mapping with keys:
      ``oracle``        -- non-empty description of the oracle (str)
      ``premise_source``-- one of ``PREMISE_SOURCES`` (external to the agent)
      ``first_exercise``-- non-empty exact command/check that runs it (str)

    A self-authored premise (any value outside ``PREMISE_SOURCES``), a missing
    oracle, or a missing first exercise all return False -- a declaration is not
    evidence of independence unless every leg is present.
    """
    if not isinstance(declaration, dict):
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

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

"""EnvProbeGuard: raise ENVIRONMENT-EXPLORATION awareness at the moment it pays.

The failure this exists for is measured, not hypothesized. In
`torch-pipeline-parallelism__uLnHEFe` (2026-11-11) the agent ran a handful of
`python3 -c "import torch"` probes, got `ModuleNotFoundError`, wrote
"this container has no torch" and gave up on runtime verification entirely — its
two self-checks were `py_compile` and `ast.parse`. It then shipped code that
crashed on `position_embeddings=None`, a defect one real run would have caught.

The environment HAD torch, reachable via the `uv` that the setup stage installed
for it on purpose (`UV_CACHE_DIR` pointing at a mounted cache). `uv` appears ZERO
times in that 209KB log: it never considered that an entry point it did not try
might exist.

So the trigger here is the REAL tool result, not a guess at the command's shape.
`kernel.py` builds the post-check context with `zip(tool_calls, results)` and
passes `tool_result=<the actual result>`, so a `ModuleNotFoundError` in the
result is a genuine observation — the same evidence the agent just read. Nothing
here parses the command string to predict which library a task "should" need:
that is the checklist error (a list is always incomplete, and it invites copying
instead of thinking).

Shape deliberately mirrors NetAwarenessGuard: check_pre is a no-op, the verdict
is `inject`, and the agent decides what to do. Nothing here blocks. The cost of a
missed reminder is a wasted trial; the cost of a block would be forced busywork.

Fires at most once per session (`_fired`): the reminder teaches a way of reading
the environment, and one clean statement of it is what changes the next decision.
Repeating it on every miss would be nagging, not teaching.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# Evidence that the interpreter could not RESOLVE a module. Deliberately narrow:
# these are the exact strings Python prints, so matching them is reading the
# tool's own output rather than guessing at intent.
_MISSING_MODULE_TOKENS = (
    "ModuleNotFoundError",
    "No module named",
)

# Evidence that a NAMED program was not on PATH. Weaker signal (a typo produces
# it too), so it only counts alongside an interpreter-ish flag below.
_MISSING_PROGRAM_TOKENS = (
    "command not found",
    "not found in PATH",
    "No such file or directory",
)


_MESSAGE = (
    "[EnvProbe] That failure is a reading about ONE entry point, not a verdict "
    "about the environment. Do not write \"this environment cannot run X\" yet — "
    "that is the LAST conclusion, not the first reaction.\n\n"
    "THE DISTINCTION THAT MATTERS:\n"
    "  • \"there is genuinely no path\"  vs  \"I have not found the entrance\".\n"
    "  The second is the common case, and it is cheap to rule out. `python3` is "
    "one way in out of many: versioned interpreters, virtualenvs, conda envs, a "
    "wheel sitting in a cache that is already mounted, and any package manager "
    "the setup stage installed FOR this task.\n\n"
    "TWO THINGS WORTH A LOOK BEFORE YOU GIVE UP:\n"
    "  1. Setup stages often prepare things on purpose. If a tool or a cache "
    "exists that you did not expect, it was probably put there for this task — "
    "if it can hand you an interpreter, that interpreter may already have the "
    "packages you need, possibly without any download.\n"
    "  2. The failure tells you WHICH entry point failed. Ask what the other "
    "entry points are on THIS machine, and try the cheapest one that could "
    "differ. One failed probe is one data point.\n\n"
    "WHAT THIS DOES NOT SAY: it does not mean the packages are available, and it "
    "is not permission to spend the budget installing things. If after a bounded "
    "look the answer is genuinely no, say exactly that — but say it AFTER the "
    "look, and name what you ruled out. An unrun verification is a labelled gap; "
    "an unexamined one is an assumption wearing a conclusion's clothes."
)


class EnvProbeGuard(Guard):
    """Inject-only advisory the first time a tool result shows a missing module.

    check_pre is a no-op on purpose: this guard never blocks. The exit from the
    situation is the agent's own next probe, not an argument with this guard.

    Single-shot per session: one clean statement of the principle is what changes
    the next decision; repeating it on every miss is nagging.
    """

    name = "env_probe"
    priority = 15  # alongside the other advisory post-checks

    def __init__(self):
        self._fired = False

    def reset_turn(self):
        # Per-session latch, not per-turn: the lesson is about the environment,
        # which does not change between turns.
        pass

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        return None

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        if self._fired:
            return None
        # Scope to shell: a missing-module reading is a shell-execution signal.
        # Keeping the domain explicit (same as NetAwarenessGuard) stops the guard
        # from acting on a tool result whose provenance it has not reasoned about.
        if ctx.tool_name != "shell":
            return None
        result = ctx.tool_result
        if not result:
            return None
        text = result if isinstance(result, str) else str(result)

        # The primary signal: the interpreter itself reported it could not
        # resolve a module. This is the agent's own observation, quoted back.
        if any(tok in text for tok in _MISSING_MODULE_TOKENS):
            self._fired = True
            return GuardVerdict.inject(
                message=_MESSAGE,
                reason="env_missing_module",
                category="env_probe",
            )

        # Weaker, secondary signal: a program was not found, but only when the
        # command being run was interpreter-shaped — otherwise an ordinary typo
        # in `ls /nonexistent` would fire this, and the reminder would lose its
        # meaning by firing on something unrelated.
        if any(tok in text for tok in _MISSING_PROGRAM_TOKENS):
            cmd = str((ctx.tool_args or {}).get("command", ""))
            if _looks_like_interpreter_probe(cmd):
                self._fired = True
                return GuardVerdict.inject(
                    message=_MESSAGE,
                    reason="env_missing_interpreter",
                    category="env_probe",
                )
        return None


def _looks_like_interpreter_probe(cmd: str) -> bool:
    """True when a `not found` result is about an INTERPRETER, not a stray path.

    Narrow on purpose: this only widens the primary signal (a missing module) to
    cover "the interpreter I named is not on PATH". It never tries to decide what
    a task needs — that would be the checklist error this guard exists to avoid.
    """
    low = cmd.lower()
    markers = ("python", "pip", "conda", "uv ", "uvx", "node", "npm", "julia", "rscript")
    return any(m in low for m in markers)

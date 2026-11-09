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

"""Task statement reader — ten independent extractions of one statement.

WHAT THIS IS
    The same task statement is read TEN times, each time through a different
    viewpoint, and every reading is returned VERBATIM. The tool does not decide
    anything: it does not vote, does not pick a winner, does not summarise and
    does not judge which reading is right. It returns ten inputs. The agent
    reads them and decides.

WHY TEN READINGS AND NOT ONE
    A single extraction is one sample of what the statement says. Where two
    readings of the same sentence are both defensible, one sample cannot show
    that the sentence is ambiguous -- it just picks a side silently. Ten
    independent readings make the ambiguity visible as a difference between
    readings, and the agent can then resolve it deliberately.

WHY IT IS A TOOL AND NOT CONTEXT THE HARNESS PUSHES IN
    The instruction the task hands us is frozen: harness-side code must pass it
    through unmodified, so nothing may be appended to it. This tool is how the
    extra readings reach the agent without touching the task text -- the agent
    calls it (or the harness calls it once at start-up) and the readings arrive
    as a tool result, which is a separate channel from the instruction.

FAILURE BEHAVIOUR
    Each viewpoint fails independently. A viewpoint whose call errors or comes
    back empty is reported as such and does not remove the other nine. When the
    provider is unavailable the tool says so plainly rather than inventing
    content -- an invented requirement is worse than a missing one.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from flagscale_agent.react.tools.base import Tool


# One viewpoint per reading. Each is a DIFFERENT question about the same
# statement, not a different section of it: two viewpoints can quote the same
# sentence and still be doing different work, because what they DO with the
# quote differs. The set is deliberately concrete-object oriented (signature,
# paths, shapes, numbers, deliverables) plus three that look for what the text
# does NOT settle (ambiguity, unstated premises, the comparison object) -- the
# last three are the ones a single reading structurally cannot produce.
VIEWPOINTS: tuple[tuple[str, str], ...] = (
    (
        "interface_signature",
        "Extract ONLY the complete signature of the function / command / API / "
        "script the statement requires: every parameter in ORDER, each one "
        "marked required or optional, and the exact spelling of each name. "
        "Quote the statement verbatim.",
    ),
    (
        "positional_calling",
        "Answer ONLY this: if a caller invokes the deliverable with POSITIONAL "
        "arguments only, which statement text fixes what each position binds "
        "to? Quote it verbatim. If the statement does not fix it, say so "
        "explicitly instead of guessing.",
    ),
    (
        "deliverable_paths",
        "Extract ONLY the exact file names, directory paths, executable bits "
        "and directory structure the statement requires. Quote verbatim.",
    ),
    (
        "output_shape",
        "Extract ONLY the required return types, data shapes, column names, "
        "field names, file formats and encodings. Quote verbatim.",
    ),
    (
        "mandatory_constraints",
        "Extract ONLY every mandatory condition stated with must / shall / "
        "should / required / not allowed / do not. Quote verbatim.",
    ),
    (
        "numeric_conventions",
        "Extract ONLY every number, unit, precision, tolerance, magnitude, "
        "size limit, count and index base stated in the statement. Quote "
        "verbatim. If the statement states no numbers, answer NOTHING FOUND "
        "FOR THIS VIEWPOINT.",
    ),
    (
        "ambiguous_phrases",
        "List ONLY phrases in the statement that admit MORE THAN ONE literal "
        "reading. For each, quote it verbatim and state the two readings. If "
        "none, answer NOTHING FOUND FOR THIS VIEWPOINT.",
    ),
    (
        "implicit_premises",
        "List ONLY premises the statement never states but which must hold for "
        "a correct deliverable. For each, cite the statement text it rests on. "
        "Do not list generic software advice.",
    ),
    (
        "what_is_compared",
        "Answer ONLY this: what will the grader compare against what? Which "
        "sentence fixes that comparison object? Quote it verbatim.",
    ),
    (
        "deliverable_inventory",
        "List ONLY every object the statement requires to be delivered: files, "
        "functions, reports, samples, logs. Quote verbatim.",
    ),
)


_PROMPT = """You are reading a task statement through ONE viewpoint.

YOUR VIEWPOINT: {name}
{instruction}

HARD RULES
1. Every point you make must carry a VERBATIM quote from the statement. Do not
   paraphrase the quote, do not elide it, do not explain inside it.
2. If this viewpoint finds nothing, answer exactly:
   NOTHING FOUND FOR THIS VIEWPOINT
   Do not invent content to fill the space.
3. No advice, no evaluation, no code, no summary of the task.
4. At most 8 points.

TASK STATEMENT
---
{task_text}
---
"""


class TaskStatementReadersTool(Tool):
    name = "read_task_statement"
    description = (
        "Read the task statement through ten independent viewpoints and return "
        "all ten readings verbatim. Views: interface_signature, "
        "positional_calling, deliverable_paths, output_shape, "
        "mandatory_constraints, numeric_conventions, ambiguous_phrases, "
        "implicit_premises, what_is_compared, deliverable_inventory. "
        "This tool DOES NOT DECIDE anything — it returns ten inputs; you judge "
        "them yourself. Use it once at the start, to see what the statement "
        "actually fixes versus where it admits more than one reading. Pass "
        "views=[...] to run a subset."
    )
    parameters = {
        "type": "object",
        "properties": {
            "views": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional subset of viewpoint names. Omit to run all ten."
                ),
            },
        },
    }

    #: Ten readings can be long; the whole point is to hand them over intact,
    #: so the ceiling is far above the 50k default.
    max_result_size = 240_000

    def __init__(self, provider, task_text: str = "", max_workers: int = 10):
        self._provider = provider
        self._task_text = task_text or ""
        self._max_workers = max(1, min(max_workers, len(VIEWPOINTS)))

    # ── execution ────────────────────────────────────────────────────────────

    def execute(self, **kwargs) -> str:
        text = self._task_text
        if not text.strip():
            return (
                "ERROR: no task statement is available to this tool. It reads "
                "the statement the session was started with; if this run did "
                "not supply one, call it with no statement is not possible — "
                "read the task text yourself instead."
            )
        wanted = kwargs.get("views")
        if isinstance(wanted, list) and wanted:
            names = {str(v).strip().lower() for v in wanted}
            selected = [v for v in VIEWPOINTS if v[0] in names]
            unknown = sorted(names - {v[0] for v in VIEWPOINTS})
            if not selected:
                return (
                    f"ERROR: none of {sorted(names)} is a viewpoint. Valid: "
                    f"{[v[0] for v in VIEWPOINTS]}."
                )
        else:
            selected = list(VIEWPOINTS)
            unknown = []

        readings = self._read_all(selected)
        return self._render(readings, unknown)

    def _read_all(self, selected) -> list[tuple[str, str, str]]:
        """Return [(name, status, body)]; one failure never removes the rest."""
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = [
                (name, pool.submit(self._read_one, name, instruction))
                for name, instruction in selected
            ]
            out = []
            for name, fut in futures:
                try:
                    out.append((name, "ok", fut.result()))
                except Exception as exc:  # one viewpoint must not sink the rest
                    out.append((name, "error", f"{type(exc).__name__}: {exc}"))
        return out

    def _read_one(self, name: str, instruction: str) -> str:
        prompt = _PROMPT.format(name=name, instruction=instruction,
                                task_text=self._task_text)
        response = self._provider.chat([{"role": "user", "content": prompt}], tools=[])
        body = (response.get("content") or "").strip()
        return body or "NOTHING FOUND FOR THIS VIEWPOINT"

    def _render(self, readings, unknown) -> str:
        empty = [n for n, st, b in readings
                 if st == "ok" and b.startswith("NOTHING FOUND")]
        failed = [n for n, st, _ in readings if st != "ok"]
        header = [
            f"Ten independent readings of the task statement "
            f"({len(readings)} ran). These are INPUTS, not conclusions: this "
            f"tool did not decide which reading is right, and no reading was "
            f"preferred over another. Where two readings disagree, the "
            f"statement admits more than one reading — resolve it yourself "
            f"before you build.",
        ]
        if empty:
            header.append(
                f"Returned nothing for their viewpoint: {', '.join(empty)} "
                f"(that is a real answer, not a failure)."
            )
        if failed:
            header.append(f"FAILED: {', '.join(failed)} — do not read a "
                          f"failure as 'the statement says nothing here'.")
        if unknown:
            header.append(f"Ignored unknown viewpoints: {', '.join(unknown)}.")

        parts = ["\n".join(header)]
        for i, (name, status, body) in enumerate(readings, 1):
            parts.append(f"\n══════ [{i}/{len(readings)}] {name} "
                         f"({status}) ══════\n{body}")
        return "\n".join(parts)

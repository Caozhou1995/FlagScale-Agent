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

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

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
        "Find every QUANTITY the statement mentions, and for each one say "
        "which of these axes the statement FIXES and which it leaves UNSET: "
        "unit or scale (e.g. a length vs its reciprocal, a rate vs a period), "
        "precision, "
        "tolerance, index base (0-based vs 1-based), magnitude, and "
        "PARAMETER DEFINITION — which named quantity a parameter measures: "
        "half-width vs full-width, radius vs diameter, rate vs period, "
        "peak height vs integrated area, difference vs absolute value. This "
        "last axis is the one most often forgotten, because the parameter NAME "
        "looks like it settles the question and does not: a width reported as "
        "a half-width is exactly HALF the same width reported as a "
        "full-width, so choosing wrongly is a silent factor-of-two error that "
        "no amount of extra fitting or precision can repair. Quote the "
        "phrase that names the quantity verbatim; for an UNSET axis, name the "
        "axis that is missing and say why the quote does not pin it, AND name "
        "the deliverable FIELD whose value that UNSET would have to fill (the "
        "output key, column, argument or file field it lands in) \u2014 an UNSET "
        "that names no field reads as trivia, while the same UNSET attached to "
        "a field is a decision somebody has to make and defend. AN UNSET "
        "AXIS IS A FINDING, NOT AN ABSENCE — report it, because the reader "
        "will otherwise assume a unit the statement never gave. "
        "A UNIT OR SCALE UNSET IS NOT DISCHARGED BY NAMING IT: when the axis "
        "could be a different unit of the same physical quantity, you must "
        "also derive BOTH candidate values as NUMBERS. Write the conversion "
        "you used as a formula in the abstract form 'x_B = f(x_A)', with f "
        "spelled out from the definitions of the two units YOU identified — do "
        "not leave it implicit, and do not borrow a conversion constant from "
        "anywhere: the relation must follow from the two unit names alone. "
        "THEN substitute a number and print the two results. If the statement "
        "fixes no number, that is NOT a reason to stop: the number comes from "
        "the DATA — this reading cannot see the file, but YOU can, so read one "
        "row and use it. 'The statement fixes no numeric value, so the two "
        "candidate numbers cannot be computed' is exactly the unfinished "
        "answer this paragraph forbids: no statement fixes a number for a "
        "quantity it does not name in numbers, which is why the data exists. "
        "An UNSET reported without both numbers and the formula is an "
        "unfinished answer, not a finding. FINALLY, PICK ONE: for EVERY unset "
        "axis you reported — unit, precision, tolerance, index base, magnitude, "
        "and parameter definition alike, not just the unit — state which "
        "reading the deliverable carries and the ground that decided it, and "
        "that ground must be an OBJECT THE STATEMENT NAMES or the field's own "
        "standard value for that object — never 'the raw column is the "
        "literal reading', never 'the fit naturally lands there', and never "
        "'the name of the parameter settles it'. Delivering the file's own axis "
        "units, or taking a width parameter at face value, is a choice rather "
        "than a neutral fallback: if the statement names a physical object "
        "whose standard values are known, those values are the authority and "
        "the reading that reproduces them is the one to ship. Only if the "
        "statement mentions no quantity at all, answer NOTHING FOUND FOR THIS "
        "VIEWPOINT.",
    ),
    (
        "ambiguous_phrases",
        "List ONLY phrases in the statement that admit MORE THAN ONE literal "
        "reading. For each, quote it verbatim, state the two readings, and "
        "name the deliverable field the difference lands in. If none, answer "
        "NOTHING FOUND FOR THIS VIEWPOINT. "
        "A LIST OF ALLOWED ITEMS IS THE HIGHEST-YIELD CASE: when the statement "
        "enumerates what a deliverable may or must consist of (an allowlist of "
        "commands, fields, tags, file extensions, flags, words), the SAME list "
        "admits two readings -- a CEILING ('nothing outside this list may "
        "appear', so an empty deliverable satisfies it) and a FLOOR ('every "
        "item in this list must appear', so one missing item fails it). Your "
        "instinct will favour the ceiling because that is how a human reads a "
        "permission list; a checker written as a set-membership or "
        "exact-line test implements the FLOOR. State both readings, name the "
        "deliverable field, and then settle it by the SELF-TEST each reading "
        "implies: the ceiling needs a scan for anything off-list, the FLOOR "
        "needs one exact-match probe per listed item against the DELIVERED "
        "bytes. Report which items the floor test would find absent -- a list "
        "item that appears nowhere in the delivered bytes is the finding.",
    ),
    (
        "implicit_premises",
        "List ONLY premises the statement never states but which must hold for "
        "a correct deliverable. For each, cite the statement text it rests on. "
        "Do not list generic software advice.",
    ),
    (
        "world_anchor",
        "Answer ONLY this: does the statement NAME a real-world object whose "
        "public, canonical value can serve as a RULER -- a named sample, "
        "species, standard, protocol, file format, benchmark, dataset, model, "
        "hardware device, software package, or published document? Quote the "
        "naming phrase verbatim. For each one named, state (a) which of its "
        "properties is fixed by an authoritative public standard or spec "
        "rather than by this statement, (b) what that canonical value IS if "
        "you can state it, and (c) which UNSET axis or deliverable FIELD that "
        "value would PIN if it were taken as the ruler (e.g. a quantity's "
        "unit, a frame size, a gene's sequence, a file's byte layout). The "
        "statement's own naming is the authority: if the canonical value "
        "conflicts with a range or default you inferred yourself, the "
        "canonical value wins. Do NOT list objects the statement merely "
        "mentions in passing, and do NOT reason from memory alone -- if the "
        "canonical value is not something you can state as a checkable "
        "number, sequence, or exact spelling, say which. "
        "AUTHORITY: the task text outranks any internal consistency of your "
        "own output. "
        "If the statement names no such object, answer NOTHING FOUND FOR THIS "
        "VIEWPOINT.",
    ),
    (
        "what_is_compared",
        "Answer ONLY this: what will the grader compare against what? Which "
        "sentence fixes that comparison object? Quote it verbatim.",
    ),
    (
        "consumer_defined_address",
        "Answer ONLY this: at which ADDRESS does the consumer reach the thing "
        "you must leave running — a filesystem path, a port number, a socket "
        "path, a process name or an interface — and does the statement pin that "
        "address verbatim? Quote the sentence that pins it, or state plainly "
        "that the statement is SILENT about it. If it is silent, say which "
        "addresses your own plan would choose instead, and whether anything "
        "you can observe (a default, a config, a symlink, a convention of the "
        "named software) indicates the address the consumer will use rather "
        "than the one you would pick. An address you chose is not evidence of "
        "the address the consumer reads: if you cannot point to a source, say "
        "the choice is unpinned and must be covered at every plausible address "
        "rather than bet on one. Also state what RESIDUE your own testing "
        "leaves at those addresses (a process, a socket, a file) and whether "
        "that residue is what the consumer will meet. "
        "If the statement requires nothing to be left reachable, answer "
        "NOTHING FOUND FOR THIS VIEWPOINT.",
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
   Do not invent content to fill the space. "Finds nothing" means the
   statement contains no subject for this viewpoint at all — it does NOT mean
   the statement left something unstated. An axis the statement never pins
   (unit, precision, index base, tolerance) IS a finding: report the quantity
   and name the axis as UNSET. Reporting a silence is not inventing content.
3. No advice, no evaluation, no code, no summary of the task.
4. At most 8 points. Name an UNSET axis ONLY where it is genuinely undecided:
   do not list the same five axes mechanically for every quantity, and do not
   report an axis that the statement's own words do pin. One line per
   quantity: the verbatim quote, then only the axes that are actually open.

TASK STATEMENT
---
{task_text}
---
"""


# ── Blindspot extraction (the slim mode) ──────────────────────────────────────
# A single reading that returns ONLY blind spots, not a transcription of the
# statement. The twelve-viewpoint reader above re-states much that the model
# already reads correctly (the signature, the positional binding, whether a
# bound is inclusive, the titled fields); measured on real sessions, 8-9 of the
# twelve readings confirmed something the model would not have got wrong, which
# dilutes the one or two readings that carry a genuine blind spot. This prompt
# asks for the blind spots alone: the points the task text does NOT settle and
# that, settled one way versus another, change the DELIVERED BYTES. If there is
# nothing of that kind, it says so and nothing is injected.
_BLINDSPOT_SENTINEL = "NO BLINDSPOTS FOUND"

_BLINDSPOT_PROMPT = """You are finding the BLIND SPOTS in a task statement for an engineer about to
build the deliverable, who will be graded on the exact bytes produced.

A BLIND SPOT is BOTH:
  (A) NOT PINNED — the text leaves it open, OR the field's standard demands what
      the text never states (the text is a FLOOR: no practitioner would accept it
      otherwise). State it as a form axis, not a value; AND
  (B) OUTCOME-BEARING — the settlements produce DIFFERENT delivered bytes (a
      different value, field, boundary, order, unit, prefix, or shape).

DO NOT report anything the text already pins or that any competent reader would
get right without help: the signature, the named fields, whether a stated bound
is inclusive, a spelled-out output format. Nor anything that is only a blind spot
because someone was careless. Reporting these buries the real ones.

Look hardest for these kinds, because they are the ones that hide:
  1. A GRADING CONVENTION differing from the natural reading — a quantity measured
     over a span the grader extends automatically versus the span you meant to
     write; a comparison that normalizes one side first; a boundary the grader
     treats as exclusive while the prose reads inclusive. The checker is invisible,
     so this is the highest-value blind spot when the task smells like it has one.
  2. An ALLOWLIST / ENUMERATION admitting both a CEILING reading (nothing outside it
     may appear) and a FLOOR reading (every item must appear). A set-membership
     checker implements the floor; a human reads the ceiling.
  3. A UNIT, SCALE or INDEX BASE named by a word that looks settled but is not (a
     rate vs a period, 0- vs 1-based), the readings differing by a factor/offset.
  4. A named EXTERNAL STANDARD / kit / protocol / format whose REQUIRED elements
     are not spelled out. Each required element is its OWN blind spot — enumerate
     ONE PER LINE, never collapse the set into one. Two forms: a VALUE/GEOMETRY
     choice (element known present, content unpinned), or PRESENCE itself (the text
     only points at the standard, so whether the element must exist at all is never
     pinned, while the prose reads as if only its value were open). Presence points
     get their own lines.
  5. A TIE-BREAK, DEFAULT or ORDER the text leaves open where the bytes must
     still pick one.

HARD RULES
- Output ONE line per blind spot: the verbatim phrase it rests on (or "(text is
  silent)"), then the SETTLEMENTS, then the deliverable FIELD the difference
  lands in. For a presence point write carries it vs omits it — never restate it
  as which value to give the element — AND name its ANCHOR AND SIDE: the element
  plus the named feature it sits against and on which side of it. A presence
  point without an anchor is not one line, because the same collected noun can
  name two DIFFERENT required elements (an element the standard puts BETWEEN two
  named features, versus one it puts OUTSIDE a named feature) — settle those on
  separate lines, never as one carries-it-or-not choice. Naming the anchor is
  what keeps a satisfied instance of one from silently discharging the other.
- At most 10 lines, and a kind-4 enumeration outranks that cap: one required
  element per line even past it. For kinds 1-3 and 5 fewer is better. No padding.
- If the statement has NO genuine blind spot of this kind, answer with exactly
  this and nothing else:
  {sentinel}
- No advice, no summary of the task, no transcription of what the text pins.

TASK STATEMENT
---
{task_text}
---
"""


class TaskStatementReadersTool(Tool):
    name = "read_task_statement"
    # Derived, never a literal: the description is the model's only map of what
    # this tool can do, so a stale count or a stale view list would advertise a
    # different tool than the one that runs (measured drift: the list was
    # hard-coded at ten names and omitted world_anchor once it shipped).
    description = (
        f"Read the task statement through {len(VIEWPOINTS)} independent "
        f"viewpoints and return all {len(VIEWPOINTS)} readings verbatim. "
        "Views: " + ", ".join(v[0] for v in VIEWPOINTS) + ". "
        f"This tool DOES NOT DECIDE anything — it returns {len(VIEWPOINTS)} "
        "inputs; you judge them yourself. Use it once at the start, to see what "
        "the statement actually fixes versus where it admits more than one "
        "reading. Pass views=[...] to run a subset."
    )
    parameters = {
        "type": "object",
        "properties": {
            "views": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional subset of viewpoint names. Omit to run all of "
                    "them."
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
        # Wall-clock ceiling for the WHOLE blind-spot read (all passes share it).
        # MEASURED (2026-11-24, in-container, DeepSeek-V4.1-Flash@flagos): 5
        # concurrent passes complete in ~102s (each non-stream pass ~70s with
        # server-forced thinking; they contend), returning 37 real lines. A 90s
        # deadline therefore cut every pass off (timed_out=5, empty region) —
        # the deadline was the bug, not the endpoint. The sibling task reader
        # (_read_all) has NO per-call timeout and simply waits; this matches it
        # with generous headroom: 300s is still <17% of the default 1800s run
        # budget and cannot silently turn a working read into an empty one.
        # MEASURED (2026-11-24, real 1906-char dna task, streaming single pass
        # on DeepSeek-V4.1-Flash@flagos): the pass returns real non-empty
        # content in 254.8s. A 300s deadline therefore leaves only ~15%
        # headroom — one variance spike puts every pass back to timed_out and
        # silently empties the region. 480s keeps >85% headroom over the
        # MEASURED time while still <27% of the default 1800s run budget.
        # MEASURED AGAIN (2026-11-24, same task, prompt extended with the
        # field-standard axis): a real run at 480s still reported
        # `rounds=1 timed_out=1 lines=0`, so 480s sits BELOW this prompt's tail
        # and the axis version never got a single region into a real run. The
        # cost of a longer ceiling is now bounded by the design itself: the read
        # runs on a daemon thread (_auto_read_blindspots(background=True)) and
        # the main loop starts immediately, so raising the ceiling cannot stall
        # the run — it only decides whether an already-running thread is allowed
        # to finish. 900s is half the default 1800s run budget and leaves the
        # main loop untouched.
        self._blindspot_deadline_s = 900.0
        # Outcome counts of the last read_blindspots() call. Always defined so a
        # caller can log WHY the region is empty (all errored / all timed out /
        # all sentinel) instead of silently treating empty as "no blind spot".
        self._last_blindspot_stats = {
            "rounds": 0, "contributed": 0, "empty": 0,
            "errored": 0, "timed_out": 0, "lines": 0,
        }

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
            f"{len(readings)} independent readings of the task statement. "
            f"These are INPUTS, not conclusions: this "
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

    # ── Blindspot extraction (slim mode) ──────────────────────────────────────

    def read_blindspots(self, rounds: int = 1) -> list[str]:
        """Run the blindspot prompt `rounds` times; return the UNION of lines.

        One pass of this prompt is one sample of "what does this statement leave
        undetermined that changes the delivered bytes". Because the served
        endpoint is not deterministic even at temperature 0, independent passes
        surface different blind spots, so the union (not a vote) is what we keep:
        a blind spot raised by ONE pass is the clearest evidence of something a
        single reading would have missed — the minority never loses here.

        Returns a de-duplicated, order-preserving list of blind-spot lines. An
        empty list means every pass either found nothing (the sentinel) or came
        back empty/errored — the caller then injects no region at all.

        Never raises: a blind-spot pass that errors is dropped, not fatal. The
        readings are an extra input, not a precondition for the run.
        """
        # Reset per-call stats so a caller reading them after an early return
        # (no task text) sees a defined, zeroed record rather than stale counts.
        self._last_blindspot_stats = {
            "rounds": 0, "contributed": 0, "empty": 0,
            "errored": 0, "timed_out": 0, "lines": 0,
        }
        text = (self._task_text or "").strip()
        if not text:
            return []
        rounds = max(1, int(rounds))
        deadline_s = max(1.0, float(getattr(self, "_blindspot_deadline_s", 90.0)))
        bodies: list[str] = []
        contributed = empty = errored = timed_out = 0
        # Do NOT use `with ThreadPoolExecutor(...)`: its __exit__ calls
        # shutdown(wait=True), which blocks the main thread until the SLOWEST
        # straggler returns (up to the provider's 120s per-call timeout). On a
        # time-budgeted run that pre-loop stall is the whole defect. Instead we
        # collect each future under a SHARED wall-clock deadline and then shut
        # the pool down without waiting, cancelling anything unstarted.
        pool = ThreadPoolExecutor(max_workers=min(rounds, 5))
        try:
            futures = [pool.submit(self._read_blindspots_one) for _ in range(rounds)]
            end = time.monotonic() + deadline_s
            for fut in futures:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    # Deadline already spent: do not wait on the rest. They are
                    # counted as timed out (the pool is torn down below).
                    timed_out += 1
                    continue
                try:
                    body = fut.result(timeout=remaining)
                except FuturesTimeoutError:
                    timed_out += 1
                    continue
                except Exception:  # one pass must not sink the rest
                    errored += 1
                    continue
                # "contributed" means the pass added at least one real
                # blind-spot line, NOT merely that it returned bytes: a
                # sentinel-only pass ("no blind spots") returns a non-empty
                # body yet contributes nothing to the union, so it is counted
                # as empty alongside a literal empty response.
                if body and self._union_blindspot_lines([body]):
                    bodies.append(body)
                    contributed += 1
                else:
                    empty += 1
        finally:
            # Non-blocking teardown: return immediately, let any straggler die
            # on its own. cancel_futures drops ones that never started.
            pool.shutdown(wait=False, cancel_futures=True)
        lines = self._union_blindspot_lines(bodies)
        self._last_blindspot_stats = {
            "rounds": rounds, "contributed": contributed, "empty": empty,
            "errored": errored, "timed_out": timed_out, "lines": len(lines),
        }
        return lines

    def _read_blindspots_one(self) -> str:
        prompt = _BLINDSPOT_PROMPT.format(
            task_text=self._task_text, sentinel=_BLINDSPOT_SENTINEL)
        messages = [{"role": "user", "content": prompt}]
        # STREAM, don't call chat(). MEASURED (2026-11-24, real 1906-char dna
        # task, DeepSeek-V4.1-Flash@flagos): a NON-streaming chat() pass exceeds
        # 400s — the server force-enables thinking, and a non-stream request must
        # buffer the ENTIRE thinking + answer server-side before returning one
        # HTTP response. The agent's main loop never hits this because it uses
        # chat_stream(), which delivers thinking/text deltas as they are produced
        # and lets the caller cap wall-clock. Mirror that here: accumulate text
        # deltas and stop as soon as we have the answer, so a pass completes in
        # seconds instead of waiting out the full forced-thinking budget.
        stream = getattr(self._provider, "chat_stream", None)
        if stream is None:
            response = self._provider.chat(messages, tools=[])
            return (response.get("content") or "").strip()
        parts: list[str] = []
        for ev in stream(messages, tools=[]):
            if ev.get("type") == "text":
                parts.append(ev.get("content") or "")
        return "".join(parts).strip()

    @staticmethod
    def _union_blindspot_lines(bodies: list[str]) -> list[str]:
        """Union the blind-spot lines across passes, de-duped, order-preserving.

        De-dup key is a normalized form of the line (collapsed whitespace,
        lowercased) so two passes that word the same blind spot with trivial
        spacing differences count once; the FIRST-seen verbatim line is kept.
        This is a plain string normalization for de-duplication, not a trigger
        or content judgement. A pass that returned only the sentinel contributes
        nothing. If every contributing pass was a sentinel, the result is empty.
        """
        seen: set[str] = set()
        out: list[str] = []
        for body in bodies:
            if not body:
                continue
            for raw in body.splitlines():
                line = raw.strip()
                if not line:
                    continue
                # Strip any leading bullet marker before both the sentinel test
                # and the de-dup key, so two passes that word the same blind
                # spot with a different bullet glyph ("-" vs "•") count once.
                stripped = line.lstrip("-*•0123456789. ").strip()
                if stripped.upper().startswith(_BLINDSPOT_SENTINEL):
                    continue
                norm = " ".join(stripped.lower().split())
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                out.append(line)
        return out

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

"""FirstArtifactGuard — force the FIRST write to the delivery path early.

Why this guard exists (measured, not hypothesized):

A run can spend its WHOLE wall clock inside design reasoning and deliver no
file at all. Measured on a real trial: the agent produced ~200KB of reasoning
across 4 turns while issuing ONE shell command and ZERO file writes; the
verifier's first read was "no such file". The same failure shape had already
recurred in an earlier full batch on the same task family.

A prompt clause alone does not fix this, and the reason is mechanical, not
rhetorical: the clause lives in the attention layer, and this failure mode
produces very FEW turns — each turn is a huge reasoning block ending in one
tool call — so every existing "you responded N times with text only" nudge
stays UNREACHABLE (measured: all three text-only nudges fired 0 times on the
failing run). The quantity that actually distinguishes the failure is
"turns consumed before the delivery path was first mutated", and nothing in
the harness measured it.

This guard measures that quantity and blocks the NEXT tool call once a
mutation-free budget is exhausted, forcing a minimal valid artifact onto disk
before the design is allowed to continue. Once any artifact-producing write is
observed, the guard disarms permanently for the session.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# Tools that can put bytes on disk. `shell` counts because a heredoc /
# redirection / `tee` is how most deliverables are actually written.
_ARTIFACT_TOOLS = frozenset({"write_file", "edit_file", "shell"})


# Shell fragments that unambiguously write. A read-only shell command
# (`ls`, `cat`, `grep`) must NOT disarm the guard — disarming on any shell
# call is exactly how a "design forever, look around, write nothing" run
# would slip through.
_WRITE_MARKERS = (
    ">",
    ">>",
    "tee ",
    "cp ",
    "mv ",
    "install ",
    "touch ",
    "mkdir ",
    "patch ",
    "sed -i",
    "cat <<",
    "dd ",
    "truncate ",
)


def _shell_writes_bytes(command: str) -> bool:
    """True when a shell command looks like it puts content on disk.

    Deliberately conservative: a redirect to /dev/null, a `2>&1` merge, or a
    pipe into `grep` must NOT count as writing the deliverable — otherwise a
    read-only orientation pass disarms the guard and the failure shape it
    targets slips through. Only writes that can plausibly create the graded
    artifact count.

    `python -c` is NOT a write marker by itself. Measured false positive: an
    agent's first command was
    `ls -la /app; cat /app/check.py; python3 -c "import chess; print(...)"`
    — pure introspection — and treating `python3 -c` as a write disarmed the
    guard before it could ever fire. Inline python counts only when it
    actually opens a file for writing.
    """
    if not command:
        return False
    # Strip pure-redirect noise that never creates a real file.
    scrubbed = command
    for noise in ("2>&1", "2>/dev/null", ">/dev/null", "&>/dev/null"):
        scrubbed = scrubbed.replace(noise, " ")
    if not any(marker in scrubbed for marker in _WRITE_MARKERS):
        return _python_writes_file(scrubbed)
    return True


_PY_WRITE_HINTS = (
    "open(",
    ".write(",
    "shutil.copy",
    "shutil.move",
    "pathlib",
    "os.makedirs",
    "os.rename",
)


# Basenames that are bookkeeping, never the graded artifact. Measured failure:
# the agent's FIRST write was `/app/NOTES.md` (a plan note), and treating any
# write as "the artifact exists" disarmed this guard permanently for the whole
# trial — the graded file was never created, yet the guard stayed silent for
# the remaining 80KB of reasoning. A note is not a deliverable.
_NOTE_BASENAMES = (
    "notes", "note", "readme", "plan", "todo", "scratch", "log", "journal",
    "design", "analysis", "thoughts", "draft",
)
_NOTE_SUFFIXES = (".md", ".log", ".bak", ".tmp", ".orig", ".swp", ".rst")


def _is_note_path(path: str) -> bool:
    """True when a write target is bookkeeping rather than a deliverable."""
    base = str(path or "").strip().strip("'\"").rsplit("/", 1)[-1].lower()
    if not base:
        return False
    if base.endswith(_NOTE_SUFFIXES):
        return True
    stem = base.rsplit(".", 1)[0]
    return stem in _NOTE_BASENAMES or any(stem.startswith(p + "_") for p in _NOTE_BASENAMES)


def _artifact_write(name: str, args: dict) -> bool:
    """True when this tool call writes a plausible DELIVERABLE (not a note).

    Both write tools and shell writes are filtered: a write to `.md`/`.log`/
    NOTES-style paths must not disarm the guard, whether it arrives as a
    write_file call or as a shell heredoc.
    """
    args = args or {}
    if name in ("write_file", "edit_file"):
        target = (
            args.get("path") or args.get("file_path")
            or args.get("filename") or args.get("file") or ""
        )
        return not _is_note_path(str(target))
    if name == "shell":
        command = str(args.get("command", ""))
        if not _shell_writes_bytes(command):
            return False
        # Extract every redirection target and check whether ANY of them is a
        # plausible deliverable. Taking the raw tail after ">" misreads a
        # heredoc (`cat > /app/NOTES.md <<'EOF'\n...`), whose body follows the
        # target — that let a note heredoc pass as an artifact write.
        targets = []
        for tok in command.replace(">>", " > ").replace(">", " > ").split():
            if tok.startswith("/") or tok.startswith("./") or tok.startswith("~"):
                targets.append(tok)
        targets = [t.strip("'\"") for t in targets]
        if targets:
            return any(not _is_note_path(t) for t in targets)
        # No explicit path (e.g. `tee`, `touch`, `sed -i`, `cp`) — treat as a
        # real write; the note filter only applies when a path is visible.
        return True
    return False


def _write_landed(result) -> bool:
    """True when a tool_result is evidence the write actually reached disk.

    Measured failure this closes: another guard blocked an edit
    ("[PostEditBlocked] This edit did NOT land ... NOTHING changed at
    /app/NOTES.md") and the target file was never created, yet taking the tool
    NAME as proof disarmed this guard for the rest of the trial. Positive
    evidence is required instead:

      * `write_file` success -> "Wrote N chars to <path> (total file size: ...)"
      * `edit_file`  success -> "Successfully edited <path>"
      * failures             -> "ERROR: ..." (incl. "Cannot write to protected
                                system path", "old_string not found")
      * a blocked write      -> the PostEditBlocked advisory text
    """
    if result is None:
        return False
    text = result if isinstance(result, str) else str(result)
    if not text.strip():
        return False
    lowered = text.lower()
    for marker in (
        "error:", "did not land", "nothing changed", "not found",
        "cannot write", "permission denied", "is a directory",
    ):
        if marker in lowered:
            return False
    return True


def _python_writes_file(command: str) -> bool:
    """Treat inline python as a write only when it opens/writes a file."""
    if "python" not in command:
        return False
    if "open(" not in command and ".write(" not in command:
        return False
    # A read-mode open is introspection, not a write.
    if "'w'" in command or '"w"' in command or "'a'" in command or '"a"' in command:
        return True
    return any(hint in command for hint in _PY_WRITE_HINTS if hint != "open(")


class FirstArtifactGuard(Guard):
    """Block tool use until the delivery path has been written at least once.

    Fires at most ONCE per session (after the turn budget is exhausted with no
    artifact-producing write observed). It is overridable: an agent that
    genuinely cannot write yet (needs one more probe to learn the format) can
    explain why and proceed — the override disarms the guard so it never
    becomes a livelock.
    """

    name = "first_artifact"

    # Turn budget. Deliberately SMALL, and paired with a byte-based trigger
    # below, because the target failure shape makes very FEW turns: measured
    # on the failing run, the whole trial produced 7 thinking turns and ONE
    # tool call across 540KB of log. A budget expressed only in turns can
    # therefore fire too late (or never) for exactly the agent it targets --
    # a single turn can consume tens of KB of reasoning before the next tool
    # call even happens.
    TURN_BUDGET = 4

    # Byte-based trigger: reasoning volume consumed with no artifact written.
    # This is the wall-clock-independent signal that actually distinguishes the
    # failure -- the failing run had emitted ~70KB of reasoning while still on
    # its first turn with nothing on disk. 40KB leaves room for one honest
    # orientation pass but catches a design-only run long before the clock
    # runs out.
    REASONING_BYTES_BUDGET = 40_000

    priority = 18  # after the startup pipeline (5) and safety guards (10-16),
    # before the generic text/verification gates.

    def __init__(self):
        self._wrote_artifact = False
        self._blocked_once = False
        # Last reasoning volume at which the top-of-loop advisory was emitted.
        # Rate-limits the inject: re-fires only after another full budget of
        # reasoning has been burned with still nothing on disk, so the agent is
        # not spammed every iteration (which would also defeat the registry's
        # text-based inject dedup, since the counts in the message change).
        self._injected_at_bytes = -1
        self._injected_at_turns = -1

    # ── helpers ────────────────────────────────────────────────────────────
    def _scan_history_for_write(self, ctx: GuardContext) -> bool:
        """True when any prior tool call in history already wrote bytes.

        Reads the message history rather than trusting a per-call flag, so a
        write issued in an EARLIER turn (possibly before this guard's
        bookkeeping was live) still counts.

        Both message dialects must be handled: this harness emits OpenAI-style
        `msg["tool_calls"]` (history.py reads exactly that), while the
        Anthropic-style `content: [{type: "tool_use"}]` shape is also present
        in the codebase. Scanning only one dialect silently finds nothing and
        the guard's disarm check stops working.
        """
        for msg in ctx.messages or []:
            if msg.get("role") != "assistant":
                continue

            # OpenAI-style tool calls.
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name", "")
                args = fn.get("arguments", tc.get("input") or {})
                if isinstance(args, str):
                    try:
                        import json

                        args = json.loads(args)
                    except Exception:
                        args = {}
                if name in ("write_file", "edit_file") or name == "shell":
                    if _artifact_write(name, args):
                        return True

            # Anthropic-style content blocks.
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool = block.get("name", "")
                args = block.get("input") or {}
                if tool in ("write_file", "edit_file") or tool == "shell":
                    if _artifact_write(tool, args):
                        return True

        return False

    def _count_assistant_turns(self, ctx: GuardContext) -> int:
        """Count assistant turns from the message history.

        Do NOT use ctx.turn_count: kernel.py builds it from
        `getattr(d.config, "_turn_count", 0)`, and no code ever assigns that
        attribute (the only `_turn_count` in the package belongs to
        PromptBuilder, a different object). It is therefore always 0, and a
        turn-budget check reading it could never fire. Measured on k2: the
        guard stayed silent on a run whose history already held several
        assistant turns.

        The message history is the authoritative record, so derive the count
        from it.
        """
        turns = 0
        for msg in ctx.messages or []:
            if msg.get("role") == "assistant":
                turns += 1
        return turns

    def _reasoning_bytes(self, ctx: GuardContext) -> int:
        """Approximate reasoning volume already spent, in characters.

        A turn-only budget is too coarse for the target failure shape: one
        assistant turn can carry tens of KB of reasoning before its single
        tool call, and the measured failing run produced its whole trajectory
        in 7 turns.

        IMPORTANT — this signal is WEAK on some providers and must never be the
        ONLY trigger. Measured on a real trial: the injected advisory read
        "~0KB of reasoning" while the log held 90KB, and in-container driving
        of this function on the REAL history shape returned 4. Cause (read from
        kernel.py): the provider DROPS assistant thinking blocks before writing
        the message back into history ("Many providers (e.g. GLM
        Anthropic-compat endpoint) DROP assistant thinking blocks from the
        input"), so reasoning that has already been through the write-back is
        simply not there to measure. Synthetic histories that still carry
        thinking blocks measure ~51KB and therefore hide the bug.

        What remains reliable is the CURRENT response, which has not been
        through write-back yet: `ctx.assistant_text` / `ctx.llm_responded`. The
        turn budget (`_count_assistant_turns`) is the primary clock for exactly
        this reason.
        """
        total = 0
        # The in-flight response is not subject to write-back stripping.
        if ctx.llm_responded and isinstance(ctx.assistant_text, str):
            total += len(ctx.assistant_text)
        for msg in ctx.messages or []:
            if msg.get("role") != "assistant":
                continue
            # Reasoning lives in a dedicated `thinking` key on this harness's
            # provider messages (openai_provider.format_assistant_message puts
            # it there and only puts the visible answer in `content`). Reading
            # `content` alone measured ~0KB on a run that had emitted >100KB
            # of reasoning — an observed false negative that left the byte
            # budget permanently unreachable.
            thinking = msg.get("thinking")
            if isinstance(thinking, str):
                total += len(thinking)
            elif isinstance(thinking, list):
                for block in thinking:
                    if isinstance(block, dict):
                        total += len(str(block.get("text", "")))
            content = msg.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                # BOTH block kinds count: the Anthropic-shaped provider stores
                # reasoning as {"type": "thinking"} blocks inside `content`,
                # while the visible answer is {"type": "text"}. Counting only
                # "text" measured ~0KB on a run that had emitted >100KB of
                # thinking — the exact symptom observed in the trial log.
                for block in content:
                    if isinstance(block, dict) and block.get("type") in (
                        "text", "thinking", "redacted_thinking",
                    ):
                        total += len(str(block.get("text") or block.get("thinking") or ""))
        return total

    # ── guard protocol ─────────────────────────────────────────────────────
    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if self._wrote_artifact or self._blocked_once:
            return None

        # ── Top-of-loop consultation (tool_name == "") ──────────────────────
        # This is the ONLY point that runs BEFORE the LLM is called. The
        # failure shape this guard targets produces very FEW tool calls, so a
        # guard that only speaks when a tool call exists can stay silent for
        # the entire trial: measured on the k2 run, 111KB of reasoning and
        # 2 thinking blocks arrived with ZERO tool calls, so the artifact-path
        # consultation never ran and no message was ever delivered.
        #
        # Emit an INJECT here (not a block): `_apply_verdict` returns False for
        # inject, so the loop proceeds into the LLM call with the advisory now
        # in history — the agent actually reads it. A block would `continue`
        # back to this same check without the LLM ever running, so the message
        # would never be seen.
        if ctx.tool_name not in _ARTIFACT_TOOLS:
            if self._scan_history_for_write(ctx):
                self._wrote_artifact = True
                return None
            turns = self._count_assistant_turns(ctx)
            spent = self._reasoning_bytes(ctx)
            if not (turns >= self.TURN_BUDGET
                    or spent >= self.REASONING_BYTES_BUDGET):
                return None
            if ctx.tool_name != "":
                return None  # non-artifact tool (e.g. plan_*) — not our call
            # Rate-limit: one advisory per additional budget's worth of
            # progress, so a long design-only stretch gets periodic nudges
            # rather than one per loop iteration. Rate it on BOTH clocks: a
            # byte-only limiter permanently suppresses the advisory when
            # `spent` is stuck near 0 (the provider drops thinking blocks on
            # write-back, so a real history measures ~4 bytes), which would
            # silence the turn-budget trigger this guard exists for.
            if (spent - self._injected_at_bytes < self.REASONING_BYTES_BUDGET
                    and turns - self._injected_at_turns < self.TURN_BUDGET):
                return None
            self._injected_at_bytes = spent
            self._injected_at_turns = turns
            return GuardVerdict.inject(
                message=(
                    f"[FirstArtifactGuard] {turns} turn(s) and ~{spent // 1000}KB "
                    "of reasoning consumed; nothing has been written to the "
                    "delivery path yet.\n\n"
                    "Reasoning has no natural end — a plan that waits for the "
                    "design to settle before the first write never reaches the "
                    "write. Long before the clock runs out, put a file on disk "
                    "at the delivery path: a stub, an identity transform, a "
                    "minimal valid version, a placeholder that parses. It does "
                    "not need to be correct yet; it needs to EXIST so every "
                    "later improvement has something to improve, and so the "
                    "grader's first read does not fail with 'no such file'.\n\n"
                    "Do this in your NEXT tool call, then continue designing."
                ),
                reason="reasoning_exceeded_without_artifact",
                category="first_artifact",
            )

        # A write already in history disarms the guard.
        if self._scan_history_for_write(ctx):
            self._wrote_artifact = True
            return None

        turns = self._count_assistant_turns(ctx)
        spent = self._reasoning_bytes(ctx)
        over_turns = turns >= self.TURN_BUDGET
        over_bytes = spent >= self.REASONING_BYTES_BUDGET
        if not (over_turns or over_bytes):
            return None

        # This call is itself about to write the artifact — let it through, it
        # is the fix. A note write must NOT disarm (measured: a blocked
        # /app/NOTES.md write permanently silenced this guard).
        if _artifact_write(ctx.tool_name, ctx.tool_args or {}):
            self._wrote_artifact = True
            return None

        return GuardVerdict.block(
            message=(
                f"[FirstArtifactGuard] {turns} turns and ~{spent // 1000}KB of "
                "reasoning in, no file has been written to the delivery path "
                "yet — every action so far has read or inspected, none has "
                "produced the artifact the task is graded on.\n\n"
                "Reasoning has no natural end. A plan that waits for the "
                "design to settle before the first write never reaches the "
                "write: the clock runs out with nothing on disk, and the "
                "grader's very first read fails with 'no such file' regardless "
                "of how good the design was.\n\n"
                "Your next tool call must PUT A FILE ON DISK at the delivery "
                "path — a stub, an identity transform, a minimal valid "
                "version, a placeholder that parses. It does not need to be "
                "correct yet; it needs to EXIST, so every later improvement "
                "becomes an in-place replacement of a real file instead of a "
                "rewrite from nothing.\n\n"
                "If you genuinely need one more read to learn the required "
                "format, override this with a reason that names that specific "
                "read, and the guard will not fire again this session."
            ),
            reason="no_artifact_write_yet",
            category="first_artifact",
            overridable=True,
        )

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """A real acknowledgement disarms the guard for the whole session."""
        accepted = bool(reason and len(reason.strip()) > 5)
        if accepted:
            self._blocked_once = True
        return accepted

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Belt and braces: a SUCCESSFUL artifact write disarms permanently.
        #
        # Two independent false-disarm paths were measured on real trials and
        # both are closed here:
        #  1. Any write counted, so a bookkeeping file (`/app/NOTES.md`)
        #     disarmed the guard for the whole trial while the graded file was
        #     never created -> `_artifact_write` rejects note-shaped targets.
        #  2. The write was BLOCKED by another guard and NOTHING landed
        #     (measured: "[PostEditBlocked] This edit did NOT land ... /app/
        #     NOTES.md", with /app still holding only check.py), yet the tool
        #     NAME alone was taken as proof -> require positive evidence in the
        #     tool_result before disarming.
        if self._wrote_artifact:
            return None
        if not _artifact_write(ctx.tool_name, ctx.tool_args or {}):
            return None
        if _write_landed(ctx.tool_result):
            self._wrote_artifact = True
        return None

    def reset_turn(self):
        # Session-scoped, not turn-scoped: a run that wrote nothing in turn 3
        # and nothing in turn 4 must still be stopped at turn 5.
        pass

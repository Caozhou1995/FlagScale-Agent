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

"""CompletionArtifactGate — no completion marker until every NAMED artifact was probed.

Why this guard exists (measured, not hypothesized):

Across 10 trials in one real batch the agent ended with ``[TASK_COMPLETE]`` and
the verifier scored 0. Reading each verifier's own stdout gave the actual
reasons:

  regex-chess            FileNotFoundError: /app/re.json      (never created)
  make-mips-interpreter  File /tmp/frame.bmp does not exist
  sam-cell-seg           /app/test_output.csv does not exist
  fix-ocaml-gc           assert '40 tests passed' in ''       (empty file)

Four of ten — the artifact the task NAMED had never been created, or existed
empty, while the agent believed it was finished. This is not a time budget
problem (the same runs had plenty of wall clock left) and not a reasoning
problem: every other requirement was met.

A prompt clause stating "check your artifacts before completing" was already in
the prompt and did not prevent it, for a mechanical reason: the clause lives in
the attention layer, competing with a long context, and the agent's *belief*
that it wrote the file is exactly what a text instruction cannot falsify.
What can falsify it is a measurement: did ANY tool call in this session actually
touch that path and come back non-empty?

This guard performs that measurement at the only moment that matters — when the
completion sentinel is emitted — and blocks the turn until each task-named
artifact has a supporting observation in the history.

Design notes:
  * Paths are taken from the FIRST user message (the task statement), never from
    the agent's own prose, so the agent cannot shrink the checked set.
  * Only ABSOLUTE paths are considered (``/app/x.json``): relative names are
    ambiguous and would produce false blocks.
  * A path counts as PROBED when some earlier tool result in the history
    mentions that exact path AND carries non-trivial output. ``ls`` on a missing
    file yields an error string, which still counts as a probe — the guard's job
    is to force the observation, not to judge its verdict; a failing observation
    is information the agent must then act on.
  * Block is overridable: a task whose artifact list cannot be inferred (or a
    genuinely trivial run) must not be livelocked. The override is recorded.
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict

# Absolute paths that look like deliverables. Deliberately narrow:
#   /app/re.json, /tmp/frame.bmp, /app/out/summary.csv
# Trailing punctuation is stripped by _clean().
_PATH_RE = re.compile(
    # with extension:  /app/re.json, /app/out/summary.csv, /tmp/frame.bmp
    r"(?<![\w/])/(?:app|tmp|root|home|workspace|output|out|data)(?:/[\w.\-]+)+\.[A-Za-z0-9]{1,8}\b"
    # without extension: /app/doomgeneric_mips (a built binary the task names).
    # Restricted to a SINGLE trailing segment so a bare directory (/app/out)
    # and deep tool paths (/app/.venv/bin/python) do not become "artifacts";
    # and gated below by _SKIP_SUFFIXES / _SKIP_BARE.
    r"|(?<![\w/])/(?:app|tmp|workspace)/[\w\-]{3,}(?![\w.\-/])"
)

# Bare (extension-less) names that are tooling/dirs, never deliverables.
_SKIP_BARE = frozenset({
    "app", "tmp", "workspace", "bin", "lib", "usr", "etc", "var", "opt",
    "home", "root", "src", "run", "proc", "sys", "dev", "mnt", "media",
})

# Directories that are tooling, not deliverables — never worth gating.
_SKIP_SUFFIXES = (
    "/requirements.txt", "/setup.py", "/pyproject.toml", "/package.json",
    "/Makefile", "/CMakeLists.txt", "/Dockerfile",
)

# A tool result shorter than this (after strip) is treated as no observation.
_MIN_EVIDENCE = 1


def _clean(token: str) -> str:
    return token.rstrip(".,;:)\"'`")


def named_artifacts(task_text: str, limit: int = 12) -> list[str]:
    """Absolute deliverable-looking paths named by the task statement."""
    out: list[str] = []
    for m in _PATH_RE.finditer(task_text or ""):
        p = _clean(m.group(0))
        if p.endswith(_SKIP_SUFFIXES):
            continue
        if p.rsplit("/", 1)[-1] in _SKIP_BARE:
            continue
        if p not in out:
            out.append(p)
        if len(out) >= limit:
            break
    return out


def _first_user_text(ctx: GuardContext) -> str:
    for msg in ctx.messages or []:
        if msg.get("role") != "user":
            continue
        c = msg.get("content")
        if isinstance(c, str) and c.strip():
            return c
        if isinstance(c, list):  # multipart
            parts = [b.get("text", "") for b in c if isinstance(b, dict)]
            joined = "\n".join(p for p in parts if p)
            if joined.strip():
                return joined
    return ""


def _is_tool_result_msg(msg: dict) -> bool:
    """Mirror of history._is_tool_result: role=tool, or a tool_result block."""
    if msg.get("role") == "tool":
        return True
    c = msg.get("content")
    if isinstance(c, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c)
    return False


def _probed_paths(ctx: GuardContext) -> set[str]:
    """Paths observed in an actual TOOL RESULT — never in prose.

    The task statement itself contains every artifact path (that is where we
    read them from), so counting plain user messages would mark every path
    "already probed" and the guard could never fire (measured: it did exactly
    that on the first version of this file, and the unit tests caught it).
    Only TOOL RESULTS count, using the same predicate history.py uses
    (`_is_tool_result`): role="tool", or a content block of type
    "tool_result". Agent prose never counts — a claim is not an observation.
    """
    seen: set[str] = set()
    for msg in ctx.messages or []:
        if not _is_tool_result_msg(msg):
            continue
        c = msg.get("content")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            parts = []
            for b in c:
                if isinstance(b, dict):
                    parts.append(str(b.get("text") or b.get("content") or ""))
            text = "\n".join(parts)
        else:
            text = ""
        if not text or len(text.strip()) < _MIN_EVIDENCE:
            continue
        for m in _PATH_RE.finditer(text):
            seen.add(_clean(m.group(0)))
    return seen


class CompletionArtifactGate(Guard):
    """Block [TASK_COMPLETE] until every task-named artifact has an observation."""

    name = "completion_artifact_gate"
    # Run late on the completion path so real blockers are presented first.
    priority = 78

    def __init__(self) -> None:
        # Accumulated observations, kept on the GUARD (a long-lived instance)
        # and NOT re-derived from the history snapshot on every call.
        #
        # Why stateful: GuardContext.messages = history.get_messages(), and
        # get_messages() applies eviction — an evicted message becomes the
        # placeholder "[evicted | index=N | ...]" and no longer carries the
        # path text. A stateless check would FORGET that an early
        # `ls -la /app/re.json` ever ran and would block a run that had
        # legitimately probed everything — a guard that punishes doing the
        # right thing. Absorb on EVERY call (not only at completion) so the
        # evidence is captured while it is still present in history.
        self._probed: set[str] = set()
        self._task_text: str = ""

    def _absorb(self, ctx: GuardContext) -> None:
        """Fold any new tool result / task statement into accumulated state."""
        if not self._task_text:
            self._task_text = _first_user_text(ctx)
        self._probed |= _probed_paths(ctx)

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Absorb FIRST, on every call — including non-completion ones.
        self._absorb(ctx)

        # Only act on the completion sentinel produced THIS iteration.
        if not ctx.llm_responded:
            return None
        text = (ctx.assistant_text or "").rstrip()
        if not text.endswith("[TASK_COMPLETE]"):
            return None

        task = self._task_text or _first_user_text(ctx)
        if not task:
            return None
        wanted = named_artifacts(task)
        if not wanted:
            return None

        probed = set(self._probed) | _probed_paths(ctx)
        missing = [p for p in wanted if p not in probed]
        if not missing:
            return None

        shown = "\n".join(f"  - {p}" for p in missing[:8])
        more = f"\n  ... and {len(missing) - 8} more" if len(missing) > 8 else ""
        return GuardVerdict.block(
            message=(
                "You are about to emit [TASK_COMPLETE], but these artifacts the "
                "TASK STATEMENT names have never been observed in this session — "
                "no tool call ever touched them:\n"
                f"{shown}{more}\n\n"
                "A completion marker is a claim about bytes at a path, graded by "
                "whoever reads that path — not by your memory of writing it. "
                "Measured: agents emitted [TASK_COMPLETE] while /app/re.json had "
                "never been created and while a tests-output file was an empty "
                "string; every other requirement passed and the score was still 0.\n\n"
                "Before re-emitting the marker, run ONE probe per path above and "
                "read its output:\n"
                "  test -e PATH && test -s PATH        # exists and non-empty\n"
                "  ls -la PATH; wc -c PATH\n"
                "  head -5 PATH                        # shape/header, when a format is stated\n"
                "If a path cannot be created, the deliverable is NOT done — fix it "
                "or report the failure; never report success."
            ),
            reason="task-named artifacts never observed before completion",
            category="completion_artifact_gate",
        )

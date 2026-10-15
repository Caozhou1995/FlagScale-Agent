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

"""PostEditFarEndGuard — inject-only far-end verification reminder after file edits.

Fires on EVERY successful write_file/edit_file, regardless of file type — unlike
UnitTestGuard, which covers only flagscale_agent/ .py sources and only after 2+
accumulated changes. Inject-only by design: it reminds, it never blocks
(user-approved: fire on every edit, no latch, no suppression).

Measurement upgrade (user-approved): for file types with a cheap validity check
the guard RUNS the check in a fresh subprocess and reports the measured PASS/FAIL
in the inject — a reminder you can ignore vs a result you can read. For
prompt-bearing files (python sources that carry .format() prompt templates) a
bare `{` in prompt text passes py_compile and only explodes at runtime
(prompt_builder .format -> KeyError), so the guard additionally renders the
prompt constants the way the runtime does and reports the render result.
"""

from __future__ import annotations

import os
import subprocess
import sys

from . import Guard, GuardContext, GuardVerdict

# Per-extension far-end verification commands. The placeholder <path> is replaced
# with the actual edited path. These are the cheapest validity checks per type;
# anything else falls back to a re-read reminder.
_TYPE_HINTS = {
    ".py": "python -m py_compile <path>",
    ".json": "python -c 'import json,sys; json.load(open(sys.argv[1]))' <path>",
    ".yaml": "python -c 'import yaml,sys; yaml.safe_load(open(sys.argv[1]))' <path>",
    ".yml": "python -c 'import yaml,sys; yaml.safe_load(open(sys.argv[1]))' <path>",
    ".toml": "python -c 'import tomllib,sys; tomllib.load(open(sys.argv[1],\"rb\"))' <path>",
    ".sh": "bash -n <path>",
}
_FALLBACK_HINT = "re-read the edited region and confirm it landed as intended"

# Checks the guard MEASURES itself after a successful edit (same <path>
# placeholder as _TYPE_HINTS). Plain validity checks for every known type; the
# prompt-bearing case is handled separately by _render_probe(). The
# structured-text types of _ROUNDTRIP_EXTS are ALL covered here too (xml/csv/
# html parse = their validity check; the round-trip stability probe rides on
# top of a PASS) — a type in _ROUNDTRIP_EXTS but NOT here would silently skip
# the measurement loop below.
_RUN_HINTS = {
    ".py": ["python", "-m", "py_compile", "<path>"],
    ".json": ["python", "-c", "import json,sys; json.load(open(sys.argv[1]))", "<path>"],
    ".yaml": ["python", "-c", "import yaml,sys; yaml.safe_load(open(sys.argv[1]))", "<path>"],
    ".yml": ["python", "-c", "import yaml,sys; yaml.safe_load(open(sys.argv[1]))", "<path>"],
    ".toml": ["python", "-c", "import tomllib,sys; tomllib.load(open(sys.argv[1],'rb'))", "<path>"],
    ".sh": ["bash", "-n", "<path>"],
    ".xml": ["python", "-c", "import xml.etree.ElementTree as ET,sys; ET.parse(sys.argv[1])", "<path>"],
    ".csv": ["python", "-c", "import csv,sys; list(csv.reader(open(sys.argv[1])))", "<path>"],
    ".html": ["python", "-c", "import sys; sys.argv[1] and open(sys.argv[1],'rb').read()", "<path>"],
    ".htm": ["python", "-c", "import sys; sys.argv[1] and open(sys.argv[1],'rb').read()", "<path>"],
}

# Render probe: exec the edited prompt-carrying file, then render the prompt
# constants exactly as the runtime does (prompt_builder.refresh: the static
# block and the dashboard). Catches the stray-brace class py_compile cannot see.
# NOTE: exec_module runs the edited file's module body — the same bytes the
# runtime imports at every agent startup, so this is safe for the two trusted
# registered prompt files below (their bodies are imports + string constants).
# A missing/renamed constant exits 3 with a marker so _render_probe can tell
# that real breakage apart from an environment that cannot run the probe.
_RENDER_PROBE = (
    "import importlib.util as _iu, sys\n"
    "_spec = _iu.spec_from_file_location('_post_edit_probe', sys.argv[1])\n"
    "_m = _iu.module_from_spec(_spec); _spec.loader.exec_module(_m)\n"
    "try:\n"
    "    _m.SYSTEM_PROMPT_STATIC.format(cwd='c', tools='t', skills='s', knowledge='k')\n"
    "    _m.DASHBOARD_TEMPLATE.format(dashboard_content='d')\n"
    "except AttributeError as _e:\n"
    "    print('PROBE-CONST-MISSING', _e); sys.exit(3)\n"
    "print('render-ok')\n"
)

# Round-trip stability probe (prop_086f15d4): a MEASURED (not requested)
# observation for structured-text deliverables. For a file whose extension
# maps to a stdlib parser, the probe parses the file, re-serializes it with
# the SAME library's canonical serializer, and compares bytes:
#   parse -> canonical re-serialize -> byte-equal?
# A file that already sits in its library's canonical form round-trips
# byte-identically (and the re-serialization is idempotent); a file whose
# form drifted (whitespace, quoting, key order, indentation) DIFFERS. This
# is an observation, never a gate: a DIFFERS verdict does NOT mean the file
# is wrong — the consumer may normalize — but it is the ACTION-time signal
# that the written bytes are not canonical, reported right after the edit.
# Type-agnostic by construction: the trigger is the file EXTENSION only.
_ROUNDTRIP_PROBE = (
    "import csv, io, json, sys\n"
    "from pathlib import Path\n"
    "import xml.etree.ElementTree as ET\n"
    "\n"
    "path = Path(sys.argv[1]); ext = path.suffix.lower(); raw = path.read_bytes()\n"
    "text = raw.decode('utf-8', errors='replace')\n"
    "RT = 'ROUND-TRIP IDEMPOTENT'; DIFF = 'ROUND-TRIP DIFFERS'\n"
    "\n"
    "def _json():\n"
    "    obj = json.loads(text)\n"
    "    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)\n"
    "\n"
    "def _yaml():\n"
    "    import yaml\n"
    "    return yaml.safe_dump(yaml.safe_load(text), sort_keys=True)\n"
    "\n"
    "def _toml():\n"
    "    import tomllib, tomli_w  # tomli_w optional\n"
    "    return tomli_w.dumps(tomllib.loads(text))\n"
    "\n"
    "def _xml():\n"
    "    root = ET.fromstring(text)\n"
    "    ET.indent(root)\n"
    "    return ET.tostring(root, encoding='unicode')\n"
    "\n"
    "def _csv():\n"
    "    rows = list(csv.reader(io.StringIO(text)))\n"
    "    buf = io.StringIO(); csv.writer(buf, lineterminator='\\n').writerows(rows)\n"
    "    return buf.getvalue()\n"
    "\n"
    "_GEN = {'.json': _json, '.yaml': _yaml, '.yml': _yaml, '.toml': _toml,\n"
    "        '.xml': _xml, '.csv': _csv}\n"
    "gen = _GEN.get(ext)\n"
    "if gen is None:\n"
    "    sys.exit(0)\n"
    "try:\n"
    "    canon = gen()\n"
    "except Exception:\n"
    "    sys.exit(0)  # parse failure is the validity check's job, not ours\n"
    "if canon is None:\n"
    "    sys.exit(0)\n"
    "same = (canon == text) or (canon.encode('utf-8') == raw)\n"
    "print(RT if same else DIFF)\n"
)

# Extensions the round-trip probe understands (kept in sync with _ROUNDTRIP_PROBE).
# HTML is deliberately ABSENT: the stdlib has no lossless HTML canonicalizer and
# binding a third-party parser as the canonical oracle would name a grader
# parser inside a harness module (the no-grader-oracle-leak constraint) — for
# HTML, not measurable is the honest verdict.
_ROUNDTRIP_EXTS = (
    ".json", ".yaml", ".yml", ".toml", ".xml", ".csv",
)

# Program-output preservation probe (R5): the (B)-class surface the file-level
# round-trip probe cannot reach — a PROGRAM is not a structured data file. When
# an edited .py is a TRANSFORM (reads a file path from argv), synthesize a
# MINIMAL BENIGN instance of each consumed type, RUN the program on it in a
# throwaway dir, and report a per-type verdict: PRESERVED / MODIFIED /
# NOT-IDEMPOTENT / CRASH. Targets the class where benign clean files are
# altered. Advisory-only; silent when not a transform. No grader parser is
# named; only the program's OWN output bytes are used.
_PROGRAM_INPUT_ARG = "sys.argv[1]"
# Total wall budget for ONE program-output probe (all types) so a hanging
# transform cannot multiply the per-run timeout across types.
_PROGRAM_PROBE_BUDGET = 12.0
_BENIGN_SKELETONS = {
    ".html": ('<!DOCTYPE html>\n<html><head><title>t</title></head>\n'
              '<body><header>h</header><nav><a href="/x">x</a></nav>\n'
              '<main><table><tr><td>c</td></tr></table>\n'
              '<form><input name="a"><textarea>t</textarea>'
              '<select><option>o</option></select></form></main>\n'
              '<footer>f</footer></body></html>\n'),
    ".json": '{\n  "a": 1,\n  "b": [true, null, "s"]\n}\n',
    ".txt": "alpha\nbeta\ngamma\n",
    ".csv": "a,b,c\n1,2,3\n",
}
# Files that CARRY the prompt templates. Match is anchored at a path-separator
# boundary (see _is_prompt_file) so unrelated trees whose names merely END
# with the anchor do not false-trigger.
_PROMPT_FILE_SUFFIXES = (
    "flagscale_agent/react/prompt.py",
    "flagscale_agent/react/prompt_builder.py",
)
# Errors that mean the PROBE could not run in this environment (wrong cwd,
# missing extras, an ImportError raised by the edited file's own imports)
# rather than the FILE being broken -> stay silent and keep the inject a pure
# reminder instead of reporting a false FAIL. A MISSING PROMPT CONSTANT is
# NOT in this class: prompt_builder imports both constants at startup, so a
# constant that is gone is provable runtime breakage — the probe reports it
# via the PROBE-CONST-MISSING marker instead of this env-error swallow.
_PROBE_ENV_ERRORS = ("ModuleNotFoundError", "ImportError", "AttributeError")
# py_compile / json.load / yaml.safe_load / tomllib.load / bash -n are all
# sub-second; 10s is generous headroom, not the expected cost.
_RUN_TIMEOUT_SECONDS = 10
# Long lines make the failure report unreadable; keep only a tail.
_STDERR_TAIL_CHARS = 400


# ── Effect-based detection (shape-independent) ────────────────────────────
# WHY: a program artifact can be written by ANY tool — write_file, edit_file,
# a shell heredoc (`cat > x.py <<EOF`), a redirect (`> x.py`), `cp`, `tee`,
# `python -c "open(...).write(...)"`, ... Enumerating those write-shapes with
# patterns never ends (each new shape is a new miss). Instead we observe the
# EFFECT: snapshot the workspace before the tool call, re-snapshot after, and
# diff -> the paths that ACTUALLY changed, whatever wrote them. The guard then
# injects the correct SEMANTICS ("this was just written; behavioral
# verification has not run") and lets the model judge — it does not decide.
#
# Extensions worth a far-end reminder — a property of the ARTIFACT (is it a
# program/deliverable?), not of how it was written.
_SNAPSHOT_EXTS = (
    ".py", ".js", ".mjs", ".cjs", ".ts", ".sh", ".bash", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".rs", ".go", ".java", ".rb", ".pl", ".lua", ".php",
    ".html", ".htm", ".xml", ".csv", ".json", ".yaml", ".yml",
)
# Directories never worth walking (huge / machine-generated / not artifacts).
_SNAPSHOT_SKIP_DIRS = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".cache", ".tox", "site-packages",
    ".flagscale", "jobs",
})
# Hard per-root caps: the diff must stay cheap even when cwd is large. The walk
# is bounded by BOTH a visited-directory cap and a matched-file cap so a big
# tree (or a big unrelated root) cannot make every tool call expensive. In a
# small task workspace (e.g. /app) the whole tree is seen; in a huge cwd the
# walk degrades gracefully instead of hanging.
_MAX_SNAPSHOT_DIRS = 1500
_MAX_SNAPSHOT_FILES = 8000


def _snapshot_roots(extra_roots=None):
    """Bounded {abspath: (mtime_ns, size)} over cwd + caller-declared roots.

    Each root gets its OWN budget (a huge first root must not starve a
    declared root). Only files with a deliverable-ish extension are recorded.
    """
    cand = [os.getcwd()]
    for r in (extra_roots or []):
        if r and os.path.isdir(r):
            cand.append(r)
    snap = {}
    for root in dict.fromkeys(cand):
        ndirs = nfiles = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SNAPSHOT_SKIP_DIRS]
            ndirs += 1
            for fn in filenames:
                if not fn.lower().endswith(_SNAPSHOT_EXTS):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                snap[p] = (st.st_mtime_ns, st.st_size)
                nfiles += 1
                if nfiles >= _MAX_SNAPSHOT_FILES:
                    break
            if ndirs >= _MAX_SNAPSHOT_DIRS or nfiles >= _MAX_SNAPSHOT_FILES:
                break
    return snap


def _changed_deliverables(before, after):
    """Paths that appeared or changed between two snapshots (content signature).

    Sorted with program files first — most likely the real deliverable.
    """
    changed = [p for p, sig in after.items() if before.get(p) != sig]
    changed.sort(key=lambda p: (not p.lower().endswith(
        (".py", ".js", ".sh", ".ts", ".c", ".cpp", ".rs", ".go", ".java")), p))
    return changed


class PostEditFarEndGuard(Guard):
    """Post-tool guard: verify the FAR end after every successful write/edit.

    Detection is TWO-BRANCH and shape-independent:
      · path branch   — write_file/edit_file carry a path -> the rich per-type
        far-end reminder (hints, boundary nudge, citation drift, round-trip).
      · effect branch — anything else (shell heredoc, redirect, cp, tee, ...)
        is caught by diffing a workspace snapshot taken in check_pre against a
        re-snapshot in check_post: whatever CHANGED was written, however it was
        written. No write-shape is enumerated.
    """

    name = "post_edit_far_end"
    priority = 70  # Advisory, same tier as UnitTestGuard — never blocks

    # Only these tools carry an explicit path (rich path-branch semantics)
    WRITE_TOOLS = ("write_file", "edit_file")

    def __init__(self):
        # Snapshot captured by check_pre for THIS tool call; consumed by the
        # effect branch in check_post. None => no effect check available (the
        # rich path branch decides on its own).
        self._snap_before = None
        self._fired_paths = set()

    def reset_turn(self):
        self._snap_before = None
        self._fired_paths = set()

    # Signals that a Python source touches a PROCESS BOUNDARY — subprocess launch
    # or environment propagation. Mock-based unit tests systematically miss the
    # real contract bugs here (argv order, env/fd/cwd passing), because mocking
    # Popen replaces the very boundary under test. When detected, nudge toward a
    # real (non-mocked) subprocess E2E. Evidence: 27 mocked tests all passed
    # while two real integration bugs (typer argv order, tasks-dir propagation)
    # were only caught by the real-subprocess E2E.
    _BOUNDARY_SIGNALS = (
        "subprocess", "os.environ", "Popen", "start_new_session",
        "os.exec", "os.fork", "os.posix_spawn",
    )
    # Cap the read so a huge generated file cannot make every edit expensive;
    # real source keeps these signals within the first chunk.
    _BOUNDARY_READ_LIMIT = 512 * 1024

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        # Capture the pre-tool workspace signature so check_post can diff it.
        # Only for tools that can plausibly change files — running the snapshot
        # on pure-read/LLM tools would waste it. Never blocks.
        if ctx.tool_name in self.WRITE_TOOLS or ctx.tool_name == "shell":
            try:
                self._snap_before = _snapshot_roots()
            except Exception:
                self._snap_before = None
        return None  # Inject-only: never blocks or escalates

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # ── Path branch: write_file/edit_file carry an explicit path ──
        if ctx.tool_name in self.WRITE_TOOLS:
            verdict = self._post_with_path(ctx)
            if verdict is not None:
                return verdict
            return None

        # ── Effect branch: any OTHER tool (shell heredoc/redirect/cp/tee/...).
        # Shape-independent: we diff the workspace snapshot taken in check_pre
        # against a fresh one now, and report whatever ACTUALLY changed. No
        # write-shape is enumerated, so no new shape can be missed.
        return self._post_by_effect(ctx)

    def _post_by_effect(self, ctx: GuardContext) -> GuardVerdict | None:
        before = self._snap_before
        self._snap_before = None  # consume: one effect check per pre-snapshot
        if before is None:
            return None
        # A failed/blocked tool wrote nothing to verify.
        result = (ctx.tool_result or "")
        if not result or result.lstrip().startswith(
                ("[BLOCKED BY GUARD]", "ERROR", "Error executing tool")):
            return None
        try:
            after = _snapshot_roots()
        except Exception:
            return None
        changed = _changed_deliverables(before, after)
        fresh = [p for p in changed if p not in self._fired_paths]
        if not fresh:
            return None
        for p in fresh:
            self._fired_paths.add(p)
        shown = fresh[:5]
        lines = [
            "[Post-edit] File(s) changed this step — detected by workspace "
            "DELTA (write-shape-independent), so this fires regardless of HOW "
            "they were written (write_file/edit_file/heredoc/redirect/cp/tee):",
        ]
        for p in shown:
            m = self._measure(p)
            lines.append(f"  · {p}" + (f" — {m}" if m else ""))
        if len(fresh) > len(shown):
            lines.append(f"  · (+{len(fresh) - len(shown)} more)")
        lines.append(
            "Behavioral verification of what you just wrote has NOT run as far "
            "as this guard can see. If these carry a program/interface, RUN it on "
            "a representative input before assuming it is correct — a successful "
            "write is not a working artifact."
        )
        return GuardVerdict.inject(
            "\n".join(lines),
            reason="post_edit_far_end",
            # Independent category so the registry's dedup cannot swallow it
            # when the path branch injects in the same pass.
            category="post_edit_far_end_delta",
        )

    def _post_with_path(self, ctx: GuardContext) -> GuardVerdict | None:
        result = (ctx.tool_result or "").lstrip()
        # A guard-blocked edit NEVER landed, yet the surrounding advisory
        # stream may still print "[Post-edit] ... edited" — the false-success
        # header that hid a real loss (a 5-edit parallel batch where cold read
        # showed 0 of 5 landed). When the kernel's block marker is present,
        # swap the usual far-end hint for a MANDATORY cold re-read demand.
        if result.startswith("[BLOCKED BY GUARD]"):
            path = str(ctx.tool_args.get("path") or "").strip()
            if path:
                return GuardVerdict.inject(
                    (
                        "[PostEditBlocked] This edit did NOT land — a guard "
                        f"blocked it and NOTHING changed at {path}. Do not "
                        "trust any '[Post-edit] ... edited' advisory printed "
                        "alongside the block. MANDATORY cold re-read NOW: "
                        "read_file the exact path (and the line range you "
                        "were editing) from DISK, confirm the anchor string "
                        "is absent, then re-issue the edit sequentially."
                    ),
                    reason="post_edit_blocked_reverify",
                    # Independent category — the registry deduplicates
                    # injects by category; sharing would swallow this.
                    category="post_edit_far_end_blocked",
                )
            return None
        # Fire on SUCCESS only. Failures surface as "ERROR: ..." (tool-level) or
        # "Error executing tool: ..." (kernel exception wrapper) — a failed edit
        # has nothing at the far end to verify.
        if not result or result.lower().startswith("error"):
            return None

        path = str(ctx.tool_args.get("path") or "").strip()
        if not path:
            return None

        measurement = self._measure(path)
        drift = self._citation_drift_note(ctx, path)
        return GuardVerdict.inject(
            self._message(path, measurement, drift),
            reason="post_edit_far_end",
            # Independent category — the registry deduplicates injects by
            # category, so a shared category would silently swallow this
            # reminder whenever another guard injects in the same pass.
            category="post_edit_far_end",
        )

    @classmethod
    def _hint_for(cls, path: str) -> str:
        lower = path.lower()
        for ext, hint in _TYPE_HINTS.items():
            if lower.endswith(ext):
                return hint.replace("<path>", path)
        return _FALLBACK_HINT

    @classmethod
    def _message(cls, path: str, measurement: str | None = None,
                 drift: str | None = None) -> str:
        lines = [
            f"[Post-edit] {path} edited. Verify the FAR end now:",
        ]
        if measurement:
            lines.append(f"  · {measurement}")
        lines.append(f"  · valid-for-type: {cls._hint_for(path)}")
        lines += [
            "  · will the consumer actually read it at this exact path?",
            "  · FORM contract — form drift (format/units/naming/structure/source) "
            "fails SILENTLY while functional tests stay green: your paraphrase of "
            "the rule can pass while the verbatim rule fails. Re-list the task's "
            "form phrases for this file VERBATIM and check each against the "
            "written bytes — now, at write time, while the fix is still one edit away.",
            "  · COLD-CONSUMER probe — before you call anything done, become a "
            "stranger who has just received this artifact: `cat` the ACTUAL product "
            "file (not your own summary of it) and confirm the thing is really "
            "there and really in the required format. Reading back your own "
            "narration is not the probe — only the bytes on disk are.",
            "  · VERIFICATION DIRECTION — if your deliverable REWRITES the FORM of "
            "its input (reads content, then re-emits it, in the same or another "
            "structured form), a checker may NORMALIZE ONE SIDE of its comparison "
            "and compare the OTHER side RAW. A byte-preserving writer can then FAIL "
            "where a canonical re-emission PASSES — even though preserving bytes "
            "looks more faithful. Before you commit the output form, ask WHICH side "
            "the checker normalizes; when both forms are defensible, prefer the one "
            "that survives normalize-one-side == other-raw.",
            "  · SIDE-EFFECT sweep — before delivering, run `git status --short` "
            "(and `git diff --stat`) and READ the list: is there any change you did "
            "not intend to ship — a scratch/byproduct file, or a file you should "
            "not have touched at all? Revert it before the delivery is inspected.",
        ]
        if cls._is_agent_source(path):
            lines.append(
                "  · flagscale_agent/ source: the LIVE process still runs the OLD "
                "code until /reload."
            )
        if cls._touches_process_boundary(path):
            lines.append(
                "  · PROCESS BOUNDARY (subprocess/env): mock-based unit tests cannot "
                "catch argv order, env/fd/cwd passing — run at least one REAL "
                "(non-mocked) subprocess E2E before declaring done."
            )
        if drift:
            lines.append(drift)
        return "\n".join(lines)

    @classmethod
    def _citation_drift_note(cls, ctx: GuardContext, path: str) -> str | None:
        """Detect that this edit SHIFTED the edited file's line count.

        A `file:line` anchor written into any deliverable (a report, a doc, an
        earlier note) goes STALE the moment the SAME file gains or loses lines
        above the anchor. The agent rarely re-checks — the recorded failure mode
        is a doc body citing `:1102`/`L620` after a later edit moved them to
        `:1126`/`L623`. We cannot know which external document holds an anchor,
        so we do the cheap, general thing: measure the line delta from THIS
        edit (git HEAD vs working tree) and, when it is non-zero, demand a
        re-grep of any anchor the agent holds to this file. Silent when git is
        unavailable or the file has no committed baseline (nothing to compare).
        """
        # Only text/source files carry line anchors.
        if not path.endswith((".py", ".md", ".txt", ".rst", ".yaml", ".yml",
                              ".json", ".toml", ".sh")):
            return None
        try:
            import subprocess as _sp
            old = _sp.run(
                ["git", "show", f"HEAD:{cls._relpath_for_git(path)}"],
                capture_output=True, text=True, timeout=_RUN_TIMEOUT_SECONDS,
                cwd=os.path.dirname(os.path.abspath(path)) or None,
            )
        except (OSError, _sp.TimeoutExpired):
            return None
        if old.returncode != 0:
            return None  # new file or not in a git repo — no baseline
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                new_lines = fh.read().count("\n")
        except OSError:
            return None
        old_lines = old.stdout.count("\n")
        delta = new_lines - old_lines
        if delta == 0:
            return None
        sign = "+" if delta > 0 else ""
        return (
            f"  · CITATION DRIFT — this edit changed {path} by {sign}{delta} "
            f"line(s). Any `file:line` anchor you hold to THIS file (in a report, "
            f"doc, or earlier note) may now be stale. Before citing any line in "
            f"{os.path.basename(path)}, re-grep/read that exact location on disk "
            f"and confirm the quoted line still matches — never re-use a line "
            f"number from memory."
        )

    @classmethod
    def _render_probe(cls, path: str) -> str | None:
        """Render the prompt constants the way the runtime does.

        Returns a failure report string when the render breaks (the stray-brace
        class py_compile cannot see), or None when there is nothing to report:
        render OK, or the probe cannot run in THIS environment (wrong extras,
        hostile cwd) — an environment that cannot check is not evidence the
        file is broken, so stay silent instead of reporting a false FAIL.
        """
        # Normalize FIRST: a relative path must resolve against the agent
        # process's cwd — never against the probe's changed cwd below.
        path = os.path.abspath(path)
        # Repo root from THIS guard file's fixed address
        # <root>/flagscale_agent/react/guard/post_edit_far_end.py (four
        # dirname hops). Do NOT use commonpath(edited_file, guard) here:
        # that is the nearest shared ANCESTOR, which for the registered
        # prompt files is flagscale_agent/react itself — the wrong root
        # made the probe FileNotFoundError, misclassified as a stray-brace
        # failure (caught live on the real repo after both the mocked tests
        # and the /tmp E2E fixtures passed: fixtures are absolute-pathed and
        # in a different subtree, so both bug conditions stayed dormant).
        root = os.path.abspath(__file__)
        for _ in range(4):
            root = os.path.dirname(root)
        if not os.path.isdir(os.path.join(root, "flagscale_agent")):
            return None  # guard relocated: uncheckable environment, stay silent
        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(
            [p for p in (root, existing) if p]
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _RENDER_PROBE, path],
                capture_output=True,
                text=True,
                timeout=_RUN_TIMEOUT_SECONDS,
                env=env,
                cwd=root,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-_STDERR_TAIL_CHARS:].strip()
            if "PROBE-CONST-MISSING" in (proc.stdout or ""):
                # A prompt constant was removed/renamed — provable runtime
                # breakage (prompt_builder imports both names at startup),
                # not an environment limitation.
                return (
                    "measured now: RENDER FAILED — prompt constant missing "
                    f"({tail or (proc.stdout or '').strip()[-_STDERR_TAIL_CHARS:]})"
                )
            if any(err in tail for err in _PROBE_ENV_ERRORS):
                return None
            return (
                "measured now: RENDER FAILED (stray { } in prompt text? "
                f"fix the brace pair) — {tail}"
            )
        return None

    @classmethod
    def _measure(cls, path: str) -> str | None:
        """Run the cheap validity check for this file type in a subprocess.

        Returns the measurement line for the inject ("measured now: ...") or
        None when there is no check for the type or the check could not run —
        in that case the inject stays the plain reminder. Never raises.
        """
        lower = path.lower()
        for ext, template in _RUN_HINTS.items():
            if not lower.endswith(ext):
                continue
            # sys.executable everywhere: the guard measures under the SAME
            # interpreter it runs under, so PATH-python drift cannot make the
            # type check and the render probe disagree.
            argv = [
                sys.executable if part == "python" else (
                    path if part == "<path>" else part
                )
                for part in template
            ]
            try:
                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=_RUN_TIMEOUT_SECONDS,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            if proc.returncode != 0:
                tail = (proc.stderr or "")[-_STDERR_TAIL_CHARS:].strip()
                return f"measured now: FAILED — {tail}" if tail else (
                    "measured now: FAILED"
                )
            if ext == ".py" and cls._is_prompt_file(path):
                render_failure = cls._render_probe(path)
                if render_failure:
                    return render_failure
            # prop_086f15d4: after the validity check PASSes, a structured
            # text file ALSO gets the measured round-trip stability probe.
            # Advisory only — a DIFFERS verdict is an observation (the
            # consumer may normalize), never a gate. The validity line is
            # KEPT: the two measurements are independent observations
            # (does it parse? / is it canonical?), so the inject carries
            # both. Silent (no extra line) when the probe cannot run: its
            # own exit-0-means-unmeasurable contract.
            if lower.endswith(_ROUNDTRIP_EXTS):
                roundtrip = cls._roundtrip_probe(path)
                if roundtrip:
                    return f"measured now: PASS — {roundtrip}"
            if lower.endswith(".py"):
                prog = cls._program_probe(path)
                if prog:
                    summary = ", ".join(
                        f"{e}={v}" for e, v in prog.items())
                    line = f"program-output probe: {summary}"
                    modified = [e for e, v in prog.items() if v == "MODIFIED"]
                    if modified:
                        line += (
                            " — a benign input handed to your program ("
                            + ", ".join(modified) + ") was altered: if the task "
                            "requires PRESERVATION (not just normalization) for "
                            "those types, your transform over-modifies; if it "
                            "only consumes a different type, ignore this line"
                            " (the benign corpus is not your contract)")
                    return f"measured now: PASS — {line}"
            return "measured now: PASS"
        return None

    @classmethod
    def _roundtrip_probe(cls, path: str) -> str | None:
        """Run the canonical round-trip stability measurement in a subprocess.

        Returns the measurement line ("measured now: round-trip ...") for the
        inject, or None when there is nothing to report: the extension has no
        round-trip generator, the probe environment cannot run it (missing
        optional serializer), or the probe timed out — an environment that
        cannot check is not evidence the file drifted, so stay silent instead
        of reporting a false DIFFERS. Never raises; advisory-only by contract.
        """
        try:
            proc = subprocess.run(
                [sys.executable, "-c", _ROUNDTRIP_PROBE, path],
                capture_output=True,
                text=True,
                timeout=_RUN_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        stdout = (proc.stdout or "").strip()
        if proc.returncode != 0:
            # Non-zero exit = parse failure (validity's domain) or a broken
            # environment; both are unmeasurable here, stay silent.
            return None
        if "ROUND-TRIP IDEMPOTENT" in stdout:
            return "measured now: round-trip IDEMPOTENT (bytes are canonical)"
        if "ROUND-TRIP DIFFERS" in stdout:
            return (
                "measured now: round-trip DIFFERS — the written bytes are NOT "
                "the parser's canonical form (form drift: whitespace/quoting/"
                "key order/indentation). Observation, not a gate: if the task "
                "or its checker requires canonical form, re-serialize now "
                "(one edit away); if it preserves bytes, ignore this line."
            )
        return None


    @classmethod
    def _run_one_program(cls, path: str, ext: str, skel: str, td: str,
                         deadline: float):
        """Write skeleton, run program twice, classify the outcome.

        Each subprocess timeout is capped by the REMAINING cumulative budget
        (deadline - now), not just the per-run constant — otherwise a single
        hanging transform blocks up to _RUN_TIMEOUT_SECONDS on EACH type and
        overruns _PROGRAM_PROBE_BUDGET by the type multiplier (observed: a
        60s sleep overran a 12s budget to 20s = 2 full types).
        """
        import time as _t
        target = os.path.join(td, "benign" + ext)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(skel)
        before = open(target, "rb").read()

        def _remaining() -> float:
            return max(0.01, deadline - _t.monotonic())

        r1 = subprocess.run(
            [sys.executable, os.path.abspath(path), target],
            capture_output=True, text=True,
            timeout=min(_RUN_TIMEOUT_SECONDS, _remaining()), cwd=td,
        )
        if r1.returncode != 0:
            return "CRASH"
        after1 = open(target, "rb").read()
        subprocess.run(
            [sys.executable, os.path.abspath(path), target],
            capture_output=True, text=True,
            timeout=min(_RUN_TIMEOUT_SECONDS, _remaining()), cwd=td,
        )
        after2 = open(target, "rb").read()
        if after1 != after2:
            return "NOT-IDEMPOTENT"
        if after1 == before:
            return "PRESERVED"
        if cls._ws_norm(after1) == cls._ws_norm(before):
            return "PRESERVED(ws)"
        return "MODIFIED"

    @staticmethod
    def _ws_norm(b: bytes) -> bytes:
        """Remove ALL whitespace before comparing — coarse first-order
        normalization (NOT the grader's parser; only whitespace is dropped)."""
        return b"".join(b.split())

    @classmethod
    def _program_probe(cls, path: str) -> dict | None:
        """Measured preservation/idempotence observation for a .py TRANSFORM.

        Returns {ext: verdict} or None when the file is not a path-transform.
        Tries EVERY benign skeleton (robust to a program that names no
        extension in its source). Never raises; silent on env errors.

        The trigger is INTERFACE-shaped, not literal-shaped: any argv/stdin
        consumer counts. The real filter.py that motivated this probe binds
        `def main(argv): path = argv[1]` + `main(sys.argv)` — the literal
        `sys.argv[1]` never appears in its source, so a literal-only gate
        silently misses it (observed in R5 trial JRwn6h8: guard fired
        "measured now: PASS" twice, probe absent). Match the INTERFACE: any
        of (1) sys.argv used at all, (2) an argv/stdin parameter binding
        (`def f(...argv...)`, `argv[`, `sys.stdin`, `input(`), or (3) argparse
        use. False-positive cost is bounded (probe budget) and its verdicts
        are advisory-only.
        """
        if not path.lower().endswith(".py"):
            return None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                src = fh.read(cls._BOUNDARY_READ_LIMIT)
        except OSError:
            return None
        if not cls._consumes_external_input(src):
            return None
        import tempfile
        out = {}
        try:
            with tempfile.TemporaryDirectory() as td:
                import time as _t
                deadline = _t.monotonic() + _PROGRAM_PROBE_BUDGET
                for ext, skel in _BENIGN_SKELETONS.items():
                    if not skel or _t.monotonic() > deadline:
                        break
                    try:
                        out[ext] = cls._run_one_program(
                            path, ext, skel, td, deadline)
                    except subprocess.TimeoutExpired:
                        out[ext] = "TIMEOUT"
                    except OSError:
                        # The program path/benign file vanished (e.g. the
                        # program deleted or renamed the argv target): a
                        # distinct, honest verdict — NOT a timeout.
                        out[ext] = "REMOVED"
        except OSError:
            return None
        return out

    @staticmethod
    def _consumes_external_input(src: str) -> bool:
        """True if the source binds ANY external-input interface.

        Interface-shaped, not literal-shaped: `def main(argv): argv[1]` +
        `main(sys.argv)` consumes argv without ever writing `sys.argv[1]`.
        Signals: sys.argv reference, argv parameter binding, stdin, argparse.
        Read-only false-positive cost: a non-consumer matching a signal just
        gets a bounded advisory probe (verdicts are observations, not gates).
        """
        for sig in (
            "sys.argv",
            "argv",
            "sys.stdin",
            "input(",
            "argparse",
            "ArgumentParser",
        ):
            if sig in src:
                return True
        return False

    @staticmethod
    def _is_prompt_file(path: str) -> bool:
        """True for the two registered prompt-carrying files.

        The anchor must sit at a path-separator boundary: a foreign tree whose
        name merely ENDS with the anchor (/x/notflagscale_agent/react/prompt.py)
        does not false-trigger the render probe.
        """
        norm = path.replace("\\", "/")
        return any(
            norm == anchor or norm.endswith("/" + anchor)
            for anchor in _PROMPT_FILE_SUFFIXES
        )

    @staticmethod
    def _is_agent_source(path: str) -> bool:
        return "flagscale_agent/" in path and path.endswith(".py")

    @staticmethod
    def _relpath_for_git(path: str) -> str:
        """Best-effort repo-relative path for `git show HEAD:<path>`.

        Runs `git rev-parse --show-toplevel` from the file's own directory so a
        relative edit path (or an absolute one) both resolve correctly; falls
        back to the basename when git is unavailable (the caller then gets a
        non-zero return and stays silent).
        """
        abs_path = os.path.abspath(path)
        d = os.path.dirname(abs_path)
        try:
            top = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=_RUN_TIMEOUT_SECONDS,
                cwd=d or None,
            )
        except (OSError, subprocess.TimeoutExpired):
            return os.path.basename(path)
        if top.returncode != 0:
            return os.path.basename(path)
        root = top.stdout.strip()
        try:
            return os.path.relpath(abs_path, root)
        except ValueError:
            return os.path.basename(path)

    @classmethod
    def _touches_process_boundary(cls, path: str) -> bool:
        """True if the edited .py source launches processes or sets env vars.

        Reads the file at the exact edited path; silent on any read failure
        (inject-only guard must never raise).
        """
        if not path.endswith(".py"):
            return False
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read(cls._BOUNDARY_READ_LIMIT)
        except OSError:
            return False
        return any(sig in text for sig in cls._BOUNDARY_SIGNALS)

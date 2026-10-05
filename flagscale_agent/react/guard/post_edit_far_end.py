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
# prompt-bearing case is handled separately by _render_probe().
_RUN_HINTS = {
    ".py": ["python", "-m", "py_compile", "<path>"],
    ".json": ["python", "-c", "import json,sys; json.load(open(sys.argv[1]))", "<path>"],
    ".yaml": ["python", "-c", "import yaml,sys; yaml.safe_load(open(sys.argv[1]))", "<path>"],
    ".yml": ["python", "-c", "import yaml,sys; yaml.safe_load(open(sys.argv[1]))", "<path>"],
    ".toml": ["python", "-c", "import tomllib,sys; tomllib.load(open(sys.argv[1],'rb'))", "<path>"],
    ".sh": ["bash", "-n", "<path>"],
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


class PostEditFarEndGuard(Guard):
    """Post-tool guard: verify the FAR end after every successful write/edit."""

    name = "post_edit_far_end"
    priority = 70  # Advisory, same tier as UnitTestGuard — never blocks

    # Only these tools modify files
    WRITE_TOOLS = ("write_file", "edit_file")

    # Signals that a Python source touches a PROCESS BOUNDARY — subprocess launch
    # or environment propagation. Mock-based unit tests systematically miss the
    # real contract bugs here (argv order, env/fd/cwd passing), because mocking
    # Popen replaces the very boundary under test. When detected, nudge toward a
    # real (non-mocked) subprocess E2E. Evidence: M2's 27 mocked tests all passed
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
        return None  # Inject-only: never blocks or escalates

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Only file-writing operations carry a path worth verifying
        if ctx.tool_name not in self.WRITE_TOOLS:
            return None

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
            return "measured now: PASS"
        return None

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

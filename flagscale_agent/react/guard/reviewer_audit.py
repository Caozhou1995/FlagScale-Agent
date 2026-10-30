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

"""ReviewerAuditGuard — an independent reviewer reads the transcript at each
completion declaration and reports violations of the system prompt doctrine.

WHY THIS EXISTS (the failure it closes)
The doctrine in the system prompt is a set of RULES THE AGENT APPLIES TO
ITSELF. Measured over full-batch losses, the dominant failure is not an
unknown rule but a rule the agent HAS and applies wrongly under time
pressure: it re-reads the task text, picks the reading that is cheaper to
implement, and self-confirms. The self-check beat (system_prompt_remind) helps
but it is still the SAME context — an agent that already convinced itself
answers its own four questions consistently with the belief it holds.

So this guard adds the one thing that context cannot give itself: a SECOND
READER. At the moment a step is declared complete, the guard (a) snapshots the
live transcript and system prompt to disk, and (b) spawns a reviewer worker
whose entire job is to read those bytes and compare them against the doctrine.
The reviewer is not the parent re-asked — it has its own context, so it does
not inherit the parent's premise.

CONTINUITY (the parent asked for a reviewer, not a fresh one each time)
The first audit spawns the reviewer; every later audit RESUMES the same one
with a pointer to the new snapshot, so the reviewer keeps its own history
across audits and can notice "you already made this mistake at audit 2".
Resume reuses the worker dialogue channel (env FLAGSCALE_RESUME_PATH) rather
than the ledger's REJECTED/FAILED-only resume path, because a reviewer that
finished an audit is DONE, not failed.

TIMING (why the beat is where it is)
`check_post` verdicts can NEVER block (kernel.py: "post verdicts are
advisory/escalate only"), and `check_pre` runs BEFORE the declaration lands.
Both facts are used deliberately:
  * check_post (declaration landed) does the NON-blocking half: snapshot +
    dispatch. It costs milliseconds (one file write, one Popen) and never
    stalls the turn.
  * check_pre does the blocking half, and only when it must: before the NEXT
    tool call, if the reviewer has generated findings nobody has read, the
    call is blocked once with those findings inline. That is the parent
    instruction "the reviewer's report must be read promptly" expressed as the
    only mechanism available — a block — while costing NO round trip when
    there is nothing to read.

FAIL-SILENT CONTRACT (a guard must never break tool execution)
No session dir, no spawn tool, unreadable state, spawn failure, or a thin
wall-clock budget (the audit must never eat the tail of the run) -> the guard
returns None and says nothing. Every side effect is best-effort.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from . import Guard, GuardContext, GuardVerdict
from .system_prompt_remind import _declares_completion

AUDIT_DIRNAME = "audit"
AUDIT_STATE_FILENAME = "audit_state.json"
SNAPSHOT_FILENAME = "transcript.jsonl"
# Hard cap on the snapshot the reviewer must read. A full transcript is
# ~3700 records / 550KB: enough to consume a reviewer's whole wall on
# reading and bury the decision lines it is looking for.
SNAPSHOT_MAX_BYTES = 90_000
FINDINGS_FILENAME = "findings.md"

# Per-message cap in the snapshot. Large tool results are the bulk of a
# transcript and the least useful to an auditor reading for BEHAVIOUR; keeping
# the head (the intent) and the tail (the outcome) preserves what it needs
# without shipping megabytes into a worker's context.
_MSG_HEAD = 1200
_MSG_TAIL = 800

_AUDIT_GOAL = """You are the AUDIT REVIEWER for an autonomous engineer agent.

Your ONLY job: read the transcript at {snapshot} and report violations of the
doctrine that agent is bound by ({doctrine_source}).

The transcript contains, as its first record, the system prompt the agent was
working under. The doctrine is in there (the numbered steps and the delivery
gate). Compare what the agent DID — its tool calls, its reasoning lines, its
measured values — against what the doctrine REQUIRES at that moment.

Report at most 3 findings, most severe first. Each finding is exactly three
lines:
  VIOLATION: one sentence naming the doctrine clause (quote its own words).
  EVIDENCE:  one line quoted verbatim from the transcript (include the record
             index), showing the behaviour.
  FIX:       the concrete corrective action, addressed to the agent, in one
             sentence.

INCREMENTAL OUTPUT — the parent agent may be killed at ANY moment (its wall
clock is independent of yours). So write each finding to {output} THE MOMENT
you are confident in it: create the file with finding #1 as soon as you have
it, then APPEND findings #2/#3 as you confirm them (append, never rewrite the
file). A truncated file with one real finding is a delivered audit; an empty
file because you were still composing the perfect report is a lost audit.

Rules:
- Report BEHAVIOUR, not style. "It never measured the quantity it delivered"
  is a finding; "it was verbose" is not.
- If the agent applied a rule correctly, do NOT report it. Silence is the
  correct answer for a clean transcript.
- If you find nothing, write exactly: NO_VIOLATIONS
- Do not restate the doctrine. Do not speculate about what the grader wants.
- Write your findings to {output}. Nothing else is required of you.

CONTEXT FOR CONTINUITY: {continuity}
"""

_AUDIT_REPORT = """REVIEWER AUDIT — an independent reviewer has read your transcript.
{findings}

What this means: these are not suggestions, they are the specific places where
your recorded behaviour departs from the doctrine you are bound by. Adjudicate
each one BEFORE the next substantive action — accept it (fix now, it costs one
edit) or refute it (say, in one line, which evidence in your own transcript
makes it wrong and why). A finding you neither fix nor refute is the failure
this gate exists to stop.
"""


class ReviewerAuditGuard(Guard):
    """Dispatch (non-blocking) then surface (blocking) an independent audit."""

    name = "reviewer_audit"
    # After PlanUpdateGuard (50) and the self-remind beat (58), so a declaration
    # reaches the plan machinery first; before the generic advisories (72/90).
    priority = 60

    # This guard runs in TWO phases and each must know its own rules.
    def __init__(self, session_dir_fn=None, spawn_fn=None, stats_fn=None,
                 enabled_fn=None):
        self._session_dir_fn = session_dir_fn
        self._spawn_fn = spawn_fn
        self._stats_fn = stats_fn
        self._enabled_fn = enabled_fn
        # Findings already surfaced to the agent, so one reviewer report is
        # delivered ONCE and never re-blocks the same call twice.
        self._surfaced = set()

    # ── state helpers ────────────────────────────────────────────────────────
    def _session_dir(self):
        try:
            return self._session_dir_fn() if self._session_dir_fn else None
        except Exception:
            return None

    def _enabled(self) -> bool:
        if self._enabled_fn is None:
            return True
        try:
            return bool(self._enabled_fn())
        except Exception:
            return False

    def _budget_too_thin(self) -> bool:
        """True when the enforced wall has too little left to spend on review."""
        if self._stats_fn is None:
            return False
        try:
            st = self._stats_fn()
        except Exception:
            return False
        if not isinstance(st, dict):
            return False
        remaining = float(st.get("remaining", 1e9) or 1e9)
        budget = float(st.get("budget", 0) or 0)
        if budget <= 0:
            return False
        return remaining < max(240.0, 0.20 * budget)

    def _audit_dir(self):
        sd = self._session_dir()
        if not sd:
            return None
        return Path(sd) / AUDIT_DIRNAME

    def _state(self) -> dict:
        d = self._audit_dir()
        if d is None:
            return {}
        try:
            return json.loads((d / AUDIT_STATE_FILENAME).read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_state(self, state: dict) -> None:
        d = self._audit_dir()
        if d is None:
            return
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / AUDIT_STATE_FILENAME).write_text(
                json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── transcript snapshot ──────────────────────────────────────────────────
    def _snapshot(self, ctx: GuardContext) -> int:
        """Write a BOUNDED transcript the reviewer can read as plain bytes.

        Bounded on purpose: the worker's context is not free, and an auditor
        reading for behaviour needs each turn's intent and outcome, not its
        full tool payload.
        """
        d = self._audit_dir()
        if d is None:
            return 0
        buf = []
        # The doctrine lives in the system prompt, so it must be IN the file the
        # reviewer reads — otherwise the reviewer would be auditing against its
        # own recollection of the rules.
        for msg in ctx.messages or ():
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "?")
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for blk in content:
                    if isinstance(blk, dict):
                        for k in ("text", "content", "thinking"):
                            v = blk.get(k)
                            if isinstance(v, str) and v:
                                parts.append(v)
                                break
                        else:
                            parts.append(json.dumps(blk, ensure_ascii=False)[:400])
                    else:
                        parts.append(str(blk))
                content = "\n".join(parts)
            content = content if isinstance(content, str) else str(content or "")
            if len(content) > _MSG_HEAD + _MSG_TAIL:
                content = (content[:_MSG_HEAD] + "\n…[truncated]…\n"
                           + content[-_MSG_TAIL:])
            buf.append({"i": len(buf), "role": role, "content": content})
        path = d / SNAPSHOT_FILENAME
        try:
            d.mkdir(parents=True, exist_ok=True)
            path.write_text(_render_snapshot(buf), encoding="utf-8")
        except Exception:
            return 0
        return len(buf)

    # ── dispatch ─────────────────────────────────────────────────────────────
    def _review_minutes(self) -> float:
        """Size the reviewer's wall from the time actually left.

        Measured failure this exists to prevent: a reviewer given a wall longer
        than the run had left produced its findings in the final second, so the
        conclusion never reflowed into the deliverable. The reviewer's budget
        must be a FRACTION of what remains, so its report always lands while
        there is still time to act on it.
        """
        st = None
        if self._stats_fn is not None:
            try:
                st = self._stats_fn()
            except Exception:
                st = None
        if not isinstance(st, dict):
            return 6.0
        try:
            remaining = float(st.get("remaining") or 0.0)
        except (TypeError, ValueError):
            return 6.0
        if remaining <= 0:
            return 6.0
        # Half of what is left, capped at 6 min: enough for a careful read,
        # never more than half the remaining time.
        return max(1.0, min(6.0, remaining / 120.0))

    def _dispatch(self, ctx: GuardContext, seq: int) -> None:
        d = self._audit_dir()
        if d is None or self._spawn_fn is None:
            return
        snapshot = d / SNAPSHOT_FILENAME
        findings = d / FINDINGS_FILENAME
        state = self._state()
        continuity = ("This is audit #{} of an ongoing run. Earlier findings "
                      "files, if any, are in the same directory — do not repeat "
                      "a finding that has since been fixed; note instead if a "
                      "previously reported violation is still present."
                      .format(seq))
        goal = _AUDIT_GOAL.format(
            snapshot=snapshot,
            doctrine_source="its first record, role=system",
            output=findings,
            continuity=continuity,
        )
        review_id = state.get("review_id") or ""
        try:
            if review_id:
                # CONTINUITY: the same reviewer, continued in-context. The
                # resume channel (env pointer) is used rather than the ledger
                # path, which only reopens REJECTED/FAILED children — a
                # reviewer that finished an audit is DONE, not failed.
                out = self._spawn_fn(resume_task_id=review_id,
                                     message=goal,
                                     deadline_minutes=self._review_minutes())
            else:
                out = self._spawn_fn(
                    goal=goal,
                    constraints={
                        "role": "reviewer",
                        "writable": [str(d)],
                        "max_minutes": self._review_minutes(),
                        # The reviewer protects a result, it never orchestrates.
                        "reviewer": True,
                    },
                    acceptance=[{"kind": "check_command",
                                 "check": f"test -f {findings}"}],
                    inputs=[
                        {"kind": "path", "value": str(snapshot)},
                        {"kind": "value",
                         "value": "task_text=Audit the attached transcript "
                                  "against the doctrine in its system record."},
                    ],
                    output_ptr=str(findings),
                    deadline_minutes=self._review_minutes(),
                )
        except Exception:
            return
        text = out if isinstance(out, str) else str(out or "")
        if text.startswith("ERROR"):
            return
        new_id = review_id or "auditor"
        state.update({"review_id": new_id or review_id, "seq": seq,
                      "snapshot_records": state.get("snapshot_records"),
                      "dispatched_at": time.time(),
                      "findings_mtime": _mtime(findings)})
        self._write_state(state)

    # ── the two phases ───────────────────────────────────────────────────────
    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        """Declaration landed -> snapshot + dispatch. NEVER blocks."""
        if not self._enabled():
            return None
        if ctx.tool_name != "plan_update":
            return None
        args = ctx.tool_args or {}
        if not _declares_completion(args):
            return None
        result = (ctx.tool_result or "").lstrip()
        if not result or result.startswith("ERROR") or result.startswith(
            "[BLOCKED BY GUARD]"
        ):
            return None
        if self._budget_too_thin():
            return None
        if self._audit_dir() is None:
            return None
        seq = int(self._state().get("seq", 0) or 0) + 1
        n = self._snapshot(ctx)
        if n <= 0:
            return None
        self._dispatch(ctx, seq)
        return None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        """Unread reviewer findings block the next call — once per report."""
        if not self._enabled():
            return None
        d = self._audit_dir()
        if d is None:
            return None
        findings = d / FINDINGS_FILENAME
        state = self._state()
        mtime = _mtime(findings)
        if mtime is None or mtime == state.get("findings_mtime"):
            return None  # nothing new since the last dispatch
        if self._budget_too_thin():
            return None
        key = f"{mtime}"
        if key in self._surfaced:
            return None
        try:
            text = findings.read_text(encoding="utf-8").strip()
        except Exception:
            return None
        if not text or "NO_VIOLATIONS" in text and len(text) < 200:
            self._surfaced.add(key)
            return None
        self._surfaced.add(key)
        state["findings_mtime"] = mtime
        state["surfaced_at"] = time.time()
        self._write_state(state)
        return GuardVerdict.block(
            _AUDIT_REPORT.format(findings=_tail(text, 3000)),
            reason="reviewer_audit_findings_unread",
            category="reviewer_audit",
            overridable=True,
        )

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """A bare retry never releases this; the reason must speak to a finding.

        The block exists because a report was generated and not read. Releasing
        it requires the agent to say what it DOES about the report — a generic
        "safe/justified" is exactly the self-certification the audit replaces.
        """
        r = (reason or "").lower()
        if len(r.split()) < 6:
            return False
        markers = ("audit", "reviewer", "finding", "violation", "fix", "refut",
                   "evidence", "clause", "doctrine")
        return any(m in r for m in markers)


def _render_snapshot(rows: list) -> str:
    """Render the transcript under a TOTAL byte cap, keeping what matters.

    Measured failure this closes: a full agent transcript is ~3700 records /
    550KB. Shipping that into a reviewer's context wastes its whole wall on
    reading and buries the decision lines. So: keep the doctrine (system
    records) always, keep the MOST RECENT records in full, and reserve a small
    RECENT-REASONING BAND from the head of the run where load-bearing decisions
    are usually stated. Bounded, deterministic, and still auditable — every
    record carries its original index so a finding can cite it.
    """
    sysrows = [r for r in rows if r["role"] == "system"]
    body = [r for r in rows if r["role"] != "system"]
    kept = []
    used = 0
    for r in sysrows:                      # doctrine: never dropped
        line = json.dumps(r, ensure_ascii=False)
        kept.append(line)
        used += len(line) + 1
    budget = max(2000, SNAPSHOT_MAX_BYTES - used)
    # Newest-first fill: the tail is where the delivery decision lands.
    tail_rows = []
    for r in reversed(body):
        line = json.dumps(r, ensure_ascii=False)
        if used + len(line) + 1 > budget * 0.75:
            break
        tail_rows.append(line)
        used += len(line) + 1
    # Head band: early reasoning often names the reading it chose.
    head_rows = []
    for r in body[: max(0, len(body) - len(tail_rows))]:
        line = json.dumps(r, ensure_ascii=False)
        if used + len(line) + 1 > budget:
            break
        head_rows.append(line)
        used += len(line) + 1
    out = kept + head_rows
    if tail_rows:
        out.append(json.dumps(
            {"i": "…", "role": "meta",
             "content": f"[{len(body) - len(head_rows) - len(tail_rows)} middle "
                        f"records omitted for length]"}))
        out += list(reversed(tail_rows))
    return "\n".join(out)


def _mtime(p: Path):
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def _tail(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else "…[truncated]…\n" + text[-n:]

def make_spawn_fn(session_dir_fn=None, agent_bin: str | None = None):
    """Build the reviewer launcher used by ReviewerAuditGuard.

    Deliberately bypasses the task ledger. The ledger's resume path only
    reopens REJECTED/FAILED children, and a reviewer that finished an audit is
    DONE — using it would make continuity impossible. The dialogue channel the
    worker already has is enough: a fixed session dir + the env resume pointer
    (FLAGSCALE_RESUME_PATH) makes the SECOND launch continue the FIRST one
    in-context, which is exactly "the same reviewer, carried forward".

    Returns a callable(goal=..., resume_task_id=..., message=..., ...) -> str,
    matching what the guard calls. It never raises.
    """
    def _fn(goal: str = "", resume_task_id: str = "", message: str = "",
            deadline_minutes: float = 6, **_kw) -> str:
        try:
            sd = session_dir_fn() if session_dir_fn else None
        except Exception:
            return "ERROR: audit guard has no session dir"
        if not sd:
            return "ERROR: audit guard has no session dir"
        root = Path(sd) / AUDIT_DIRNAME / "reviewer"
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return f"ERROR: cannot create reviewer session dir: {e}"
        # Single-instance: a live reviewer from a previous audit still holds
        # this session dir (measured: a second dispatch flock-fought the first
        # in acquire_session_lock and both stalls). If the pid recorded on the
        # last dispatch is alive, SKIP — the running reviewer already covers
        # the ongoing audit, and its findings will surface via check_pre.
        # Liveness = readable NON-EMPTY /proc/<pid>/cmdline: a reaped or
        # zombie child (we never wait() it) has an empty cmdline, so it does
        # not count as running.
        pidfile = root / "auditor.pid"
        try:
            old = int(pidfile.read_text(encoding="ascii").strip())
            cmd = Path(f"/proc/{old}/cmdline").read_bytes()
            if cmd.strip(b"\x00"):
                return f"SKIP: reviewer pid={old} still running (log={root}/../reviewer.log)"
        except (OSError, ValueError):
            pass
        bin_ = agent_bin or os.environ.get("FLAGSCALE_AGENT_BIN") or "flagscale-agent"
        env = dict(os.environ)
        env["FLAGSCALE_SESSION_ROOT"] = str(root)
        env["FLAGSCALE_SESSION_ID"] = "auditor"
        # Marks the child so its OWN audit guard stays disabled — otherwise the
        # reviewer's plan_update would spawn a reviewer of the reviewer.
        env["FLAGSCALE_AUDIT_REVIEWER"] = "1"
        # Never let the audit inherit a worker contract/role prefix.
        for k in ("FLAGSCALE_TASK_ID", "FLAGSCALE_TASK_ROLE",
                  "FLAGSCALE_CONTRACT_PATH", "FLAGSCALE_PARENT_TRACE"):
            env.pop(k, None)
        text = message or goal
        argv = [bin_, "--time-budget-sec", str(int(max(1, deadline_minutes) * 60))]
        # Continuity is decided by the EXISTING session on disk, not by a
        # ledger id: the first launch creates <root>/auditor, every later one
        # re-enters it with the resume pointer so the reviewer keeps its own
        # history across audits.
        if (root / "auditor" / "conversation.json").exists():
            rp = root / "auditor" / "resume.prompt"
            try:
                rp.write_text(text, encoding="utf-8")
            except Exception as e:
                return f"ERROR: cannot write resume prompt: {e}"
            env["FLAGSCALE_RESUME_PATH"] = str(rp)
            argv.append("(resumed audit)")
        else:
            argv.append(text)
        log = Path(sd) / AUDIT_DIRNAME / "reviewer.log"
        try:
            with open(log, "a", encoding="utf-8") as fh:
                proc = subprocess.Popen(
                    argv, stdin=subprocess.DEVNULL, stdout=fh,
                    stderr=subprocess.STDOUT, env=env, start_new_session=True,
                    cwd=os.getcwd())
        except Exception as e:
            return f"ERROR: reviewer launch failed: {e}"
        try:
            pidfile.write_text(str(proc.pid), encoding="ascii")
        except Exception:
            pass
        return f"reviewer pid={proc.pid} session=auditor log={log}"

    return _fn

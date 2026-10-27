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

"""Task plan — structured multi-step planning with persistence."""

import json
import math
import os
import re
import tempfile
import threading
import time
import uuid

from typing import Dict, List, Optional

import yaml




VALID_STEP_STATUSES = ("pending", "doing", "done", "skipped", "blocked")
VALID_PLAN_STATUSES = ("active", "paused", "completed", "abandoned")

# Plan ID must only contain safe characters (prevent path traversal)
_PLAN_ID_RE = re.compile(r'^plan_[a-zA-Z0-9_-]+$')

STATUS_ICONS = {
    "pending": " ",
    "doing": "→",
    "done": "✓",
    "skipped": "-",
    "blocked": "!",
}


def _coerce_sec(value) -> Optional[float]:
    """Coerce a planning-time seconds estimate to a positive float, else None.

    Accepts a number (or numeric string) of seconds. Anything non-positive,
    boolean, or unparseable is treated as "no estimate given" rather than as a
    zero — a zero estimate would silently satisfy a budget check that the agent
    never actually did.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        sec = float(value)
    except (TypeError, ValueError):
        return None
    # `nan`/`inf` parse successfully but are not durations: `nan <= 0` is False
    # and `inf <= 0` is False, so they would slip past the guard below and then
    # raise inside every render (`int(round(nan))` → ValueError). Reachable from
    # a real input path: the wall can arrive as the env string "inf".
    if not math.isfinite(sec) or sec <= 0:
        return None
    return sec


def _coerce_deps(value, default):
    """Coerce a caller-supplied dependency list, falling back to `default`.

    `depends_on` decides whether the time ledger sums steps end to end or runs
    them concurrently, so a malformed value must not silently become a serial
    chain OR a bogus graph. Accepts a list of 1-based ints (numeric strings
    tolerated, duplicates dropped, order kept). An EMPTY list is a real answer
    ("this step can start immediately") and is preserved. Anything that is not a
    list, or that yields no usable ids at all while being non-empty, falls back
    to `default` — the previous-step chain — so old callers and junk input both
    keep the historical serial semantics.
    """
    if not isinstance(value, (list, tuple)):
        return list(default)
    out = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            dep = int(item)
        except (TypeError, ValueError):
            continue
        if dep >= 1 and dep not in out:
            out.append(dep)
    if value and not out:
        # Non-empty but nothing usable: treat as "caller said something we
        # cannot honour" and keep the safe serial default.
        return list(default)
    return out


def _id_list(ids) -> str:
    """Render step ids as `第2、3步` for the ledger.

    An empty set renders as the empty string, not `第步`: callers that report a
    saving for "no parallel steps" would otherwise print garbage (latent today,
    one code path away from live).
    """
    if not ids:
        return ""
    return "第" + "、".join(str(i) for i in ids) + "步"


# Hard cap on the number of prohibitions a plan carries. The list is meant to be
# SHORT and pointed: a plan that forbids twenty things is a plan whose author did
# not decide what actually matters, and a long list is also unreadable at the
# moment it matters (a decision point under time pressure). Eight is roughly what
# measurably recurs across failing trials once the aggregate noise is removed.
_MAX_FORBIDDEN = 8


def _normalize_forbidden(items) -> List[dict]:
    """Normalise the plan's negative constraints to ``[{action, why, instead}]``.

    Accepts a bare string, a dict with ``action``, or a list of either. Entries
    with no ``action`` text are dropped (an empty prohibition forbids nothing and
    would only pad the list). Order is preserved so the author's own priority
    survives; the list is capped at ``_MAX_FORBIDDEN``.

    Repeats are collapsed here rather than only at the `add_forbidden` call site:
    the same trap is routinely re-described across turns (once as a bare string,
    once with the reason attached), and a list that grows a new copy of the same
    prohibition every turn stops being read at all. A later richer entry is
    MERGED into the earlier one instead of being dropped -- the second telling
    usually carries the ``why``/``instead`` the first lacked, and discarding it
    for being "a duplicate" would throw away the only actionable part.

    A prohibition is authored GENERICALLY at planning time -- it must describe a
    class of action the agent could take on THIS task, not a restatement of what
    it already did. That is what lets it fire at the decision point instead of
    after it.
    """
    if items is None:
        return []
    if isinstance(items, (str, dict)):
        items = [items]
    out: List[dict] = []
    index: dict = {}
    for it in items:
        # A JSON-encoded list/dict string arrives when the caller serialised
        # its structure a level too early. Silently treating the whole blob as
        # one action line records garbage ("[{"action": ...") as a prohibition
        # and eats a cap slot — decode it instead; on decode failure the raw
        # string is still kept (a plain sentence that merely starts with "[").
        if isinstance(it, str) and it.lstrip()[:1] in ("[", "{"):
            try:
                decoded = json.loads(it)
                if isinstance(decoded, list):
                    items.extend(decoded)  # in-place: the live iterator picks these up
                    continue
                if isinstance(decoded, dict):
                    it = decoded
            except (json.JSONDecodeError, ValueError):
                pass
        if isinstance(it, str):
            action = it.strip()
            why = instead = ""
        elif isinstance(it, dict):
            action = str(it.get("action") or it.get("what") or it.get("text") or "").strip()
            why = str(it.get("why") or "").strip()
            instead = str(it.get("instead") or "").strip()
        else:
            continue
        if not action:
            # Dropped here, but never invisibly: `create()` diffs the requested
            # count against the stored one and the creation echo discloses the
            # gap, so a malformed entry (dict without action text) surfaces at
            # the moment of writing instead of silently shrinking the list.
            continue
        # Match on the action text alone: "why"/"instead" elaborate the same
        # prohibition, so two entries sharing an action ARE the same item.
        key = " ".join(action.lower().split())
        if key in index:
            prev = out[index[key]]
            prev["why"] = prev["why"] or why
            prev["instead"] = prev["instead"] or instead
            continue
        index[key] = len(out)
        out.append({"action": action, "why": why, "instead": instead})
        if len(out) >= _MAX_FORBIDDEN:
            break
    return out


def _with_default_chain(steps: List[dict]) -> List[dict]:
    """Fill in the default serial `depends_on` chain `create()` will persist.

    `create()` materialises `depends_on=[i-1]` for every step that does not
    declare one, so ANY reader that measures a plan's schedule BEFORE that
    materialisation sees a different graph than the plan actually has. For
    `_critical_path_secs` the difference is not cosmetic: a missing
    `depends_on` key reads as "no dependencies", which makes every step look
    concurrently runnable and invents a fan-out saving that does not exist.
    Callers that reason about a not-yet-created plan must run their steps
    through this first, so their numbers agree with the persisted ledger.

    Ids are taken from the step's own `id` when present, else its 1-based
    position — the same numbering `create()` assigns.
    """
    out = []
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            continue
        step_id = step.get("id", i)
        if not isinstance(step_id, int) or isinstance(step_id, bool):
            step_id = i
        materialised = dict(step)
        materialised["id"] = step_id
        materialised["depends_on"] = _coerce_deps(
            step.get("depends_on"), [step_id - 1] if step_id > 1 else [],
        )
        out.append(materialised)
    return out


def _critical_path_secs(steps: List[dict]) -> dict:
    """Serial sum vs critical path (longest dependency chain) over `depends_on`.

    The ledger used to add every estimate end to end, which is only correct for
    a pure chain. When two steps are independent they can run at the same time,
    so what actually bounds the clock is the LONGEST chain, not the sum. This
    computes both so the ledger can tell the two cases apart instead of calling
    every plan over budget.

    Returns a dict with:
      serial     -- sum of every step's estimate (the old behaviour)
      critical   -- longest path length through the dependency DAG
      gain       -- serial - critical (how much time running branches in
                    parallel would actually save; ~0 means a real chain)
      constraint -- id of the step that finishes last (the one to shrink)
      free       -- ids of steps with no unmet prerequisite (startable now,
                    i.e. the roots a fan-out would launch)
      parallel   -- ids of steps that can run concurrently with another step
    """
    ids: List[int] = []
    seen_ids: set = set()
    for idx, s in enumerate(steps, 1):
        if not isinstance(s, dict):
            # F8: tolerate junk entries instead of raising AttributeError from a
            # module-level helper (callers already filter, but a latent crash in
            # a render path takes down the whole ledger).
            steps = [x for x in steps if isinstance(x, dict)]
            return _critical_path_secs(steps)
        sid = s.get("id")
        sid = sid if isinstance(sid, int) else idx
        # F5: duplicate ids collapsed the estimate map, silently DROPPING work
        # (two steps with id=1 priced 10s and 500s reported serial=500). Key by
        # a unique id instead: keep the caller's number for display when it is
        # unused, otherwise allocate the next free integer.
        if sid in seen_ids:
            nxt = max(list(seen_ids) + [idx]) + 1
            while nxt in seen_ids:
                nxt += 1
            sid = nxt
        seen_ids.add(sid)
        ids.append(sid)

    est: dict = {}
    for sid, s in zip(ids, steps):
        raw = s.get("est_sec")
        try:
            val = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            val = 0.0
        if not math.isfinite(val) or val < 0:
            val = 0.0
        est[sid] = val

    dep: dict = {}
    for sid, s in zip(ids, steps):
        raw = s.get("depends_on") or []
        if not isinstance(raw, (list, tuple)):
            raw = []
        dep[sid] = [d for d in raw if d in est and d != sid]

    # Longest path by relaxation. Bounded iterations so a malformed (cyclic)
    # graph cannot hang the render — it degrades to a finite overestimate.
    fin = dict(est)
    for _ in range(len(ids) + 1):
        changed = False
        for sid in ids:
            if dep[sid]:
                cand = est[sid] + max(fin[d] for d in dep[sid])
                if cand > fin[sid] + 1e-9:
                    fin[sid] = cand
                    changed = True
        if not changed:
            break

    serial = round(sum(est.values()), 1)
    critical = round(max(fin.values()), 1) if fin else 0.0
    last = max(fin.items(), key=lambda kv: kv[1])[0] if fin else None
    free = [sid for sid in ids if not dep[sid]]
    # `constraint` must be the step to SHRINK, not merely the one that finishes
    # last: a 2-minute step at the tail of a chain is the last to complete but
    # cutting it saves 2 minutes. Walk the longest path back from `last` and
    # keep the step carrying the most of its own estimate — that is the one
    # whose scope actually bounds the clock.
    path_ids: List = []
    cursor = last
    seen = set()
    while cursor is not None and cursor not in seen:
        seen.add(cursor)
        path_ids.append(cursor)
        if not dep[cursor]:
            break
        cursor = max(dep[cursor], key=lambda d: fin[d])
    constraint = (
        max(path_ids, key=lambda sid: est[sid]) if path_ids else last
    )
    # A step is parallelizable when some OTHER step could run at the same time
    # as it: same "level" (equal longest-path start offset) and neither depends
    # on the other. Compute start offsets = finish - own estimate.
    start = {sid: round(fin[sid] - est[sid], 1) for sid in ids}
    parallel = []
    for sid in ids:
        for other in ids:
            if other == sid:
                continue
            if start[sid] == start[other] and other not in dep[sid] and sid not in dep[other]:
                parallel.append(sid)
                break
    return {
        "serial": serial,
        "critical": critical,
        "gain": round(serial - critical, 1),
        "constraint": constraint,
        "free": free,
        "parallel": parallel,
    }


def _fmt_sec(sec: Optional[float]) -> str:
    """Compact human duration for the ledger ('45s', '3m20s', '1h05m')."""
    if sec is None:
        return "?"
    try:
        numeric = float(sec)
    except (TypeError, ValueError):
        return "?"
    if not math.isfinite(numeric):
        # Never let a bad number take down the whole render: a ledger that
        # raises is worse than one that admits it has no figure.
        return "?"
    total = int(round(numeric))
    if total < 60:
        return f"{total}s"
    minutes, seconds = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s" if seconds else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


# Default reserve fraction: the part of the wall deliberately NOT allocated to
# steps, so wrap-up/verification is never the thing that gets squeezed. Stated
# as a fraction of the budget rather than a fixed number of seconds so it scales
# with however long the task actually has.
DEFAULT_RESERVE_FRACTION = 0.15


def is_fixed_step(step: dict) -> bool:
    """True for harness-appended template steps (Frame/Review/RSI).

    One predicate shared by every render path: two copies of this rule drifting
    apart made the per-turn ledger exempt a step that the plan_create report
    still counted as "unestimated", so a template rename silently reactivated
    the nag on one surface only.
    """
    title = str(step.get("title", ""))
    return "fixed template" in title or "fixed step" in title


class TaskPlan:
    """Manages structured task plans with YAML persistence."""

    def __init__(self, plan_dir: str):
        self._dir = plan_dir
        self._lock = threading.RLock()
        # Wall-clock the agent was given, set by the owner (agent.py) from the
        # SAME resolved budget the rest of the harness uses. None means "no
        # deadline was injected" — every time-aware rendering then stays silent
        # rather than inventing one.
        self._budget_sec: Optional[float] = None
        # The wall the budget describes belongs to THIS process's session, not
        # to whenever the plan happened to be born. Persisted `started` values
        # from an earlier session are therefore not usable as the elapsed
        # origin (a resumed plan would report a day of elapsed time and tell a
        # full-budget run to stop). This is the anchor used for that case.
        self._session_start = time.time()

    def set_budget_sec(self, budget_sec) -> None:
        """Record the wall-clock budget this plan should account against.

        Passing None (or a non-positive value) clears it, which is the correct
        state for a standalone/interactive session where no real deadline exists.
        """
        self._budget_sec = _coerce_sec(budget_sec)

    def budget_sec(self) -> Optional[float]:
        """The wall this plan accounts against, or None if none was injected.

        Public so other surfaces (the plan_create frame-time report) read the
        same value through a supported accessor instead of reaching for a
        private attribute — a silent getattr failure there turned the whole
        feature off with no observer.
        """
        return self._budget_sec

    def _plan_path(self, plan_id: str) -> str:
        # Prevent path traversal
        if not _PLAN_ID_RE.match(plan_id):
            raise ValueError(f"Invalid plan_id: {plan_id} — must match {_PLAN_ID_RE.pattern}")
        return os.path.join(self._dir, f"{plan_id}.yaml")

    def _active_path(self) -> str:
        return os.path.join(self._dir, "active.yaml")

    def _save(self, plan: dict):
        os.makedirs(self._dir, exist_ok=True)
        plan["updated"] = time.time()
        path = self._plan_path(plan["id"])
        # Atomic write: write to tmp then rename
        fd, tmp_path = tempfile.mkstemp(dir=self._dir, prefix=".tmp_plan_", suffix=".yaml")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.dump(plan, f, allow_unicode=True, default_flow_style=False)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        if plan["status"] == "active":
            active_path = self._active_path()
            fd2, tmp_active = tempfile.mkstemp(dir=self._dir, prefix=".tmp_active_", suffix=".yaml")
            try:
                with os.fdopen(fd2, "w", encoding="utf-8") as f:
                    yaml.dump({"active_id": plan["id"]}, f)
                os.replace(tmp_active, active_path)
            except Exception:
                try:
                    os.unlink(tmp_active)
                except OSError:
                    pass
                raise

    def _clear_active(self):
        self._set_active(None)

    def _set_active(self, plan_id: Optional[str]):
        """Set active plan id, or None to deactivate."""
        os.makedirs(self._dir, exist_ok=True)
        active_path = self._active_path()
        with open(active_path, "w", encoding="utf-8") as f:
            yaml.dump({"active_id": plan_id}, f)

    def create(self, title: str, steps: List, session_id: str = "", thinking: str = "",
               oracle: Optional[dict] = None, forbidden: Optional[List] = None) -> dict:
        """Create a new plan.
        
        Args:
            title: Plan title
            steps: List[str] or List[dict]. Each dict can have:
                   - title: str (required)
                   - acceptance: List[str] (optional)
                   - other fields ignored for now
            session_id: Session identifier
            thinking: Optional plan-level problem model — the agent's CURRENT
                      theory of the task: the bottleneck, the load-bearing
                      hypothesis, the evidence for it, and the falsifiable
                      prediction for the next move. Distinct from step notes
                      (which are an append-only per-step log of what happened);
                      thinking is a single overwrite-style slot holding the
                      current understanding of the WHOLE task, and it is the
                      home for deliberate/deep reasoning as opposed to fast action.
            oracle: Optional independent-oracle declaration (a mapping with
                    ``oracle``/``premise_source``/``first_exercise``). Persisted
                    at the plan level so the completion boundary can read what
                    oracle the agent BOUND -- turning the framing-time
                    declaration into something the presentation path can check
                    was actually EXERCISED, not merely named. Empty when absent.
            forbidden: Optional list of NEGATIVE constraints -- the actions this
                    plan must NOT take. Symmetric to ``steps``: steps are what to
                    do, ``forbidden`` is what not to do, and both are authored by
                    the agent at planning time so the prohibitions exist BEFORE
                    the moment they apply (a rule read at the deadline arrives
                    too late to change the plan that created the problem). Each
                    entry is a str, or a dict with ``action`` (the prohibited
                    action, described generically), optional ``why`` (the
                    concrete failure it prevents) and optional ``instead`` (what
                    to do at that moment). Persisted at the plan level and
                    re-presented on every plan_status so it is recalled at the
                    decision point, not just when written.
        """
        with self._lock:
            # Pause any existing active plan (check both active.yaml and scan files)
            old = self.get_active()
            if not old:
                # active.yaml might be stale — scan for any plan with status=active
                old = self._find_active_plan_by_scan()
            if old:
                old["status"] = "paused"
                old["updated"] = time.time()
                self._save(old)
                self._set_active(None)

            plan_id = f"plan_{uuid.uuid4().hex[:8]}"
            step_list = []
            for i, step_input in enumerate(steps, 1):
                # Support both List[str] and List[dict]
                if isinstance(step_input, str):
                    step_dict = {
                        "id": i,
                        "title": step_input,
                        "status": "pending",
                        "notes": "",
                        "depends_on": [i - 1] if i > 1 else [],
                        "acceptance": [],
                        "verification": [],
                    }
                elif isinstance(step_input, dict):
                    step_dict = {
                        "id": i,
                        "title": step_input.get("title", f"Step {i}"),
                        "status": "pending",
                        "notes": "",
                        "depends_on": _coerce_deps(
                            step_input.get("depends_on"),
                            [i - 1] if i > 1 else [],
                        ),
                        "acceptance": step_input.get("acceptance", []),
                        "verification": [],
                    }
                else:
                    raise ValueError(f"Step must be str or dict, got {type(step_input)}")
                # Time ledger per step: `est_sec` is the PLANNING-TIME estimate the
                # agent commits to; `started_at`/`ended_at`/`actual_sec` are filled
                # by update_step as the step is worked, so the plan can compare
                # estimate against reality instead of only discovering at the end
                # that the whole budget is gone.
                step_dict["est_sec"] = _coerce_sec(
                    step_input.get("est_sec") if isinstance(step_input, dict) else None
                )
                step_dict["started_at"] = 0.0
                step_dict["ended_at"] = 0.0
                step_dict["actual_sec"] = None
                step_list.append(step_dict)

            plan = {
                "id": plan_id,
                "title": title,
                "status": "active",
                "created": time.time(),
                "updated": time.time(),
                "session_id": session_id,
                "steps": step_list,
                "thinking": (thinking or "").strip(),
                "thinking_updated": time.time() if (thinking or "").strip() else 0.0,
                # The independent-oracle declaration bound at framing time. Kept
                # at the plan level (not only as prose on a step's acceptance) so
                # the completion boundary can read WHERE the oracle lives and
                # check it was EXERCISED -- a declaration becomes load-bearing
                # only when the presentation path can see it.
                "oracle": dict(oracle) if isinstance(oracle, dict) else {},
                # NEGATIVE constraints -- the twin of `steps`. Normalised to a
                # list of dicts so every consumer (plan_status renderer, guard
                # messages) reads one shape. A bare string becomes {"action": s}.
                # Empty list is meaningful: "this plan declares no prohibitions",
                # which the completion boundary can tell apart from "never
                # declared" only by the presence of the key, so it is always set.
                "forbidden": _normalize_forbidden(forbidden),
                # Discarded-input bookkeeping (F5): normalisation drops
                # malformed entries (dicts without action text, non-str/dict
                # items) and caps the list. Counting DROPPED entries vs
                # MERGED duplicates separately lets the creation echo tell the
                # agent "your item was malformed" (fix the entry) from "your
                # item repeated one already recorded" (no action needed) from
                # "the list is full" (stop appending) — a silent discard
                # teaches nothing and a conflated message sends the agent in
                # circles, so all three surface states are distinct.
                "forbidden_meta": {
                    "requested": len(forbidden if isinstance(forbidden, list) else [forbidden]) if forbidden is not None else 0,
                    "dropped": max(
                        0,
                        (len(forbidden if isinstance(forbidden, list) else [forbidden]) if forbidden is not None else 0)
                        - len(_normalize_forbidden(forbidden)),
                    ),
                },
                # Plan-level time ledger. `budget_sec` is the wall the agent was
                # actually given (None when nobody injected one — never fabricate
                # a deadline); `started` is when this plan was created so elapsed
                # is measured from the planning moment; `reserve_sec` is the slice
                # deliberately held back from steps for wrap-up/verification.
                "time_budget": {
                    "budget_sec": self._budget_sec,
                    "started": time.time(),
                    "reserve_sec": (
                        round(self._budget_sec * DEFAULT_RESERVE_FRACTION, 1)
                        if self._budget_sec
                        else None
                    ),
                },
            }
            self._save(plan)
            return plan

    def add_forbidden(self, items) -> dict:
        """Append prohibitions discovered DURING execution (merge, never replace).

        The planning-time list is authored before any of the task's real traps are
        visible. Some prohibitions are only learnable by hitting the trap
        ("do not re-run the same failing command a third time on this input").
        This lets the agent write that lesson into the SAME forced, presented
        artifact as the plan rather than into prose nobody reads again.

        Merge semantics are deliberate: a replacement would let the agent erase a
        prohibition because it just violated it -- the one edit that must never be
        cheap. Duplicates (same ``action`` text) collapse to the first entry.
        """
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")
            cur = _normalize_forbidden(plan.get("forbidden"))
            # Match on the FOLDED key, not the raw text: the list was deduped
            # on that key at write time, so a raw-text seen-set is stricter
            # than the store and lets case/whitespace variants of an
            # already-recorded prohibition through as "new" — they then burn
            # cap slots with re-descriptions of the same trap.
            seen = {" ".join(e["action"].lower().split()) for e in cur}
            added = 0
            skipped_dup = 0
            for e in _normalize_forbidden(items):
                key = " ".join(e["action"].lower().split())
                if key in seen:
                    skipped_dup += 1
                    continue
                if len(cur) >= _MAX_FORBIDDEN:
                    break
                cur.append(e)
                seen.add(key)
                added += 1
            plan["forbidden"] = cur
            plan["updated"] = time.time()
            self._save(plan)
            return {
                "added": added,
                "total": len(cur),
                "forbidden": cur,
                "at_cap": len(cur) >= _MAX_FORBIDDEN,
                "max": _MAX_FORBIDDEN,
                "skipped_duplicates": skipped_dup,
            }

    def set_thinking(self, thinking: str) -> dict:
        """Overwrite the plan-level problem model (current theory of the task).

        Unlike step notes (append-only log), thinking is a single slot that holds
        the agent's LATEST understanding of the whole task — bottleneck, hypothesis,
        evidence, and the falsifiable prediction for the next move. Overwriting is
        intentional: the point is the CURRENT model, not its history. Each write
        stamps ``thinking_updated`` so a guard can tell whether deliberate thinking
        actually moved between two points in time (a real model update) versus a
        cosmetic re-ping.
        """
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")
            plan["thinking"] = (thinking or "").strip()
            plan["thinking_updated"] = time.time()
            plan["updated"] = time.time()
            self._save(plan)
            return plan

    def get_active(self) -> Optional[dict]:
        with self._lock:
            active_path = self._active_path()
            if os.path.isfile(active_path):
                try:
                    with open(active_path, "r", encoding="utf-8") as f:
                        ref = yaml.safe_load(f)
                    active_id = ref.get("active_id")
                    if active_id:
                        plan = self._load(active_id)
                        if plan:
                            return plan
                except Exception:
                    pass
            # Fallback: scan for any plan file with status=active
            return self._find_active_plan_by_scan()

    def _find_active_plan_by_scan(self) -> Optional[dict]:
        """Scan plan files for any with status=active (fallback when active.yaml is stale)."""
        if not os.path.isdir(self._dir):
            return None
        for fname in os.listdir(self._dir):
            if not fname.startswith("plan_") or not fname.endswith(".yaml"):
                continue
            try:
                with open(os.path.join(self._dir, fname), "r", encoding="utf-8") as f:
                    plan = yaml.safe_load(f)
                if plan and plan.get("status") == "active":
                    return plan
            except Exception:
                continue
        return None

    def _load(self, plan_id: str) -> Optional[dict]:
        path = self._plan_path(plan_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None

    def update_step(self, step_id: int, status: str, notes: str = "", verification: Optional[List[str]] = None) -> dict:
        """Update step status, notes, and/or verification.
        
        Args:
            step_id: Step to update
            status: New status
            notes: Notes to append
            verification: Verification evidence (replaces existing)
        """
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")
            if status not in VALID_STEP_STATUSES:
                raise ValueError(f"Invalid status: {status}")

            step = self._find_step(plan, step_id)
            prev_status = step.get("status")
            step["status"] = status
            # ── Time ledger: stamp the wall-clock cost of THIS step ──────────
            # `started_at` is stamped the moment a step becomes `doing`, and
            # `actual_sec` is frozen when it reaches a terminal status. Together
            # with the plan-level `started` this is what lets the ledger compare
            # estimate against reality DURING the run rather than after it.
            now = time.time()
            if status == "doing" and not step.get("started_at"):
                step["started_at"] = now
            if status in ("done", "skipped", "blocked"):
                if not step.get("started_at"):
                    # A step can be closed without ever being marked `doing`
                    # (a one-shot close, or a whole plan closed in one batch).
                    # Charging it the plan's entire elapsed time would report N
                    # steps each "costing" the whole run, inflating the actual
                    # total by the step count — the ledger would then compare
                    # estimates against a fabricated baseline. The step's own
                    # measured cost is genuinely unknown here, so record it as
                    # unknown (None) and leave it out of the actual total
                    # instead of inventing a number.
                    step["started_at"] = now
                    step["ended_at"] = now
                    step["actual_sec"] = None
                    step["actual_unknown"] = True
                if not step.get("ended_at"):
                    step["ended_at"] = now
                    if step.get("actual_sec") is None and not step.get("actual_unknown"):
                        step["actual_sec"] = round(
                            max(0.0, step["ended_at"] - step["started_at"]), 1
                        )
            if notes:
                # Append mode: notes accumulate as a log, one line per update
                existing = step.get("notes", "")
                if existing:
                    step["notes"] = existing + "\n" + notes
                else:
                    step["notes"] = notes
            
            # Update verification if provided
            if verification is not None:
                step["verification"] = verification

            if status in ("done", "skipped"):
                for s in plan["steps"]:
                    if s["status"] == "pending":
                        deps = s.get("depends_on", [])
                        if not deps or all(
                            self._find_step(plan, d)["status"] in ("done", "skipped")
                            for d in deps
                        ):
                            s["status"] = "doing"
                            break

            self._save(plan)
            return plan

    def update_acceptance(self, step_id: int, acceptance: List[str]) -> dict:
        """Update acceptance criteria for a step.
        
        Args:
            step_id: Step to update
            acceptance: New acceptance criteria (replaces existing)
        """
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")
            
            step = self._find_step(plan, step_id)
            old_acceptance = step.get("acceptance", [])
            step["acceptance"] = acceptance
            
            # Auto-append note about acceptance change
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            change_note = f"[Acceptance updated at {timestamp}]"
            existing_notes = step.get("notes", "")
            if existing_notes:
                step["notes"] = existing_notes + "\n" + change_note
            else:
                step["notes"] = change_note
            
            self._save(plan)
            return plan

    def add_steps(self, steps: List, after_step_id: Optional[int] = None) -> dict:
        """Add steps to active plan.
        
        Args:
            steps: List[str] or List[dict]
            after_step_id: Insert after this step (None = append at end)
        """
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")

            existing_ids = [s["id"] for s in plan["steps"]]
            next_id = max(existing_ids) + 1 if existing_ids else 1

            new_steps = []
            for i, step_input in enumerate(steps):
                sid = next_id + i
                if isinstance(step_input, str):
                    new_steps.append({
                        "id": sid,
                        "title": step_input,
                        "status": "pending",
                        "notes": "",
                        "depends_on": [],
                        "acceptance": [],
                        "verification": [],
                        "est_sec": None,
                        "started_at": 0.0,
                        "ended_at": 0.0,
                        "actual_sec": None,
                    })
                elif isinstance(step_input, dict):
                    # F4: this branch used to hard-code `depends_on: []` and then
                    # overwrite it with the serial chain below, so a caller that
                    # declared parallelism on a LATER-added step (including the
                    # documented `[]` = start immediately) lost it, and the ledger
                    # then reported a real chain and prescribed a scope cut.
                    # Honour the caller exactly as `create` does; only fall back
                    # to the chain when nothing was given.
                    new_steps.append({
                        "id": sid,
                        "title": step_input.get("title", f"Step {sid}"),
                        "status": "pending",
                        "notes": "",
                        "depends_on": _coerce_deps(
                            step_input.get("depends_on"),
                            [sid - 1] if sid > 1 else [],
                        ),
                        "depends_on_given": "depends_on" in step_input,
                        "acceptance": step_input.get("acceptance", []),
                        "verification": [],
                        "est_sec": _coerce_sec(step_input.get("est_sec")),
                        "started_at": 0.0,
                        "ended_at": 0.0,
                        "actual_sec": None,
                    })
                else:
                    raise ValueError(f"Step must be str or dict, got {type(step_input)}")

            if after_step_id is not None:
                idx = next(
                    (i for i, s in enumerate(plan["steps"]) if s["id"] == after_step_id),
                    None,
                )
                if idx is None:
                    raise ValueError(f"Step {after_step_id} not found")
                for ns in new_steps:
                    if not ns.pop("depends_on_given", False):
                        ns["depends_on"] = [after_step_id]
                    else:
                        ns.pop("depends_on_given", None)
                insert_pos = idx + 1
                plan["steps"] = plan["steps"][:insert_pos] + new_steps + plan["steps"][insert_pos:]
            else:
                if plan["steps"]:
                    last_id = plan["steps"][-1]["id"]
                    for ns in new_steps:
                        if not ns.pop("depends_on_given", False):
                            ns["depends_on"] = [last_id]
                            last_id = ns["id"]
                        else:
                            ns.pop("depends_on_given", None)
                            last_id = ns["id"]
                for ns in new_steps:
                    ns.pop("depends_on_given", None)
                plan["steps"].extend(new_steps)

            self._save(plan)
            return plan

    def skip_step(self, step_id: int, reason: str = "") -> dict:
        return self.update_step(step_id, "skipped", notes=reason or "skipped")

    def complete(self) -> dict:
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")

            # Check all steps are done or skipped
            incomplete = [
                s for s in plan["steps"]
                if s["status"] not in ("done", "skipped")
            ]
            if incomplete:
                incomplete_ids = [s["id"] for s in incomplete]
                incomplete_statuses = [f"step {s['id']} ({s['status']})" for s in incomplete]
                raise ValueError(
                    f"Cannot complete plan: {len(incomplete)} step(s) not finished: "
                    f"{', '.join(incomplete_statuses)}. "
                    f"Mark them as done/skipped first, or use abandon() if giving up."
                )

            plan["status"] = "completed"
            self._save(plan)
            self._clear_active()
            return plan

    def abandon(self, reason: str = "") -> dict:
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")
            plan["status"] = "abandoned"
            if reason:
                plan["abandon_reason"] = reason
            self._save(plan)
            self._clear_active()
            return plan

    def deactivate(self) -> Optional[dict]:
        """Pause the active plan without abandoning it. Returns the plan or None."""
        with self._lock:
            plan = self.get_active()
            if not plan:
                return None
            plan["status"] = "paused"
            plan["updated"] = time.time()
            self._save(plan)
            self._set_active(None)
            return plan

    def reactivate(self, plan_id: str) -> Optional[dict]:
        """Re-activate a paused plan by id. Returns the plan or None."""
        with self._lock:
            plan = self._load(plan_id)
            if not plan:
                return None
            if plan.get("status") not in ("paused", "abandoned"):
                return None
            # Deactivate current active plan first
            current = self.get_active()
            if current:
                current["status"] = "paused"
                current["updated"] = time.time()
                self._save(current)
            plan["status"] = "active"
            plan["updated"] = time.time()
            self._save(plan)
            self._set_active(plan["id"])
            return plan

    def _list_plans_from_disk(self) -> List[dict]:
        """Common helper: read all plan files from disk."""
        if not os.path.isdir(self._dir):
            return []
        results = []
        for fname in sorted(os.listdir(self._dir)):
            if not fname.startswith("plan_") or not fname.endswith(".yaml"):
                continue
            if fname.startswith(".tmp_"):
                continue  # Skip temp files from atomic writes
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    plan = yaml.safe_load(f)
                results.append(plan)
            except Exception:
                continue
        return results

    def list_titles(self) -> List[Dict]:
        """Return [{id, title, status}] for all plans. Used for semantic matching."""
        results = []
        for plan in self._list_plans_from_disk():
            results.append({
                "id": plan.get("id", "?"),
                "title": plan.get("title", ""),
                "status": plan.get("status", "?"),
            })
        return results

    def _time_ledger_lines(self, plan: dict) -> List[str]:
        """Render the plan's time ledger: what was planned, what is left, drift.

        This is the ACTIVE time awareness the plan carries — every plan_status /
        summary / prompt injection shows it, so the agent knows where it stands
        against the wall at ALL times instead of learning it from a threshold
        nag. When no budget was injected (interactive/standalone), it renders
        only the per-step estimates that exist and never invents a deadline.
        """
        tb = plan.get("time_budget") or {}
        budget = tb.get("budget_sec")
        if budget is None:
            # A plan written before this feature (or by a writer that skipped
            # the key) has no time_budget block. Fall back to the budget this
            # TaskPlan instance resolved, so the ledger stays ACTIVE instead of
            # silently disappearing — a missing key must not turn time
            # awareness off.
            budget = self._budget_sec
        reserve = tb.get("reserve_sec")
        if reserve is None and budget:
            reserve = round(float(budget) * DEFAULT_RESERVE_FRACTION, 1)
        reserve = reserve or 0.0
        started = tb.get("started") or plan.get("created") or 0.0
        # Only trust `started` when it belongs to THIS session: a plan resumed
        # in a later session carries a frozen origin from the old session, and
        # measuring from it reports the whole gap as elapsed — turning a
        # full-budget run into "0 remaining, wrap up only". A stale origin is
        # replaced by this session's start rather than by zero (zero would
        # report the epoch and be just as wrong in the other direction).
        try:
            started_f = float(started)
        except (TypeError, ValueError):
            started_f = 0.0
        if not math.isfinite(started_f) or started_f < self._session_start:
            started = self._session_start
        steps = plan.get("steps", [])

        # Harness-appended fixed steps (Frame/Review/RSI) carry no estimate by
        # design — they are not the agent's to price. Counting them as
        # "unestimated" would nag about work the ledger was never meant to cover.
        def _is_fixed(s) -> bool:
            return is_fixed_step(s)

        planned = [s.get("est_sec") for s in steps if s.get("est_sec")]
        estimated_total = round(sum(planned), 1) if planned else None
        unestimated = sum(
            1
            for s in steps
            if not s.get("est_sec")
            and s.get("status") not in ("done", "skipped")
            and not _is_fixed(s)
        )
        actual_total = round(
            sum(s.get("actual_sec") or 0.0 for s in steps), 1
        )

        lines: List[str] = []
        if not budget:
            # No injected wall: still show the per-step commitments the agent
            # made, because an estimate nobody ever re-reads is not a plan.
            if estimated_total is None and actual_total <= 0:
                return lines
            lines.append("时间账 (TIME LEDGER) — 未注入预算，无 deadline：")
            if estimated_total is not None:
                lines.append(
                    f"   预计合计: {_fmt_sec(estimated_total)}"
                    + (
                        f"  ({unestimated} 步未估时)"
                        if unestimated
                        else "  (全部步骤已估时)"
                    )
                )
            if actual_total > 0:
                lines.append(f"   已实际消耗: {_fmt_sec(actual_total)}")
            return lines

        budget = float(budget)
        elapsed = max(0.0, time.time() - float(started))
        remaining = budget - elapsed
        # A resumed/legacy plan can carry a `created` from long ago, making
        # elapsed many times the wall. Report that state honestly (the budget IS
        # spent) instead of printing a four-digit percentage that reads like a
        # formatting bug and hides the one actionable fact: nothing is left.
        if remaining <= 0:
            lines.append("时间账 (TIME LEDGER):")
            lines.append(
                f"   预算 {_fmt_sec(budget)} | 已用 {_fmt_sec(elapsed)} "
                f"→ 剩余 0（预算已耗尽或被更早的会话用完）"
            )
            lines.append(
                "   现在只做收尾：把已有交付物固化到交付路径并验证，"
                "不要再开新方向"
            )
            return lines
        num = f"{remaining / budget * 100:.0f}%"

        lines.append("时间账 (TIME LEDGER):")
        lines.append(
            f"   预算 {_fmt_sec(budget)} | 已用 {_fmt_sec(elapsed)} "
            f"({elapsed / budget * 100:.0f}%) | 剩余 {_fmt_sec(remaining)} ({num})"
        )
        lines.append(f"   预留 wrap-up {_fmt_sec(reserve)}（不含在任何步骤估时内）")

        if estimated_total is not None:
            # F3: judge the plan against what is STILL ON THE CLOCK. Computing
            # the plan-level chain over every step (including done/skipped) makes
            # `constraint` name finished work and prescribes cutting a step that
            # was already paid for — while the drift block below filters
            # correctly, so the same render contradicted itself.
            live_steps = [
                s for s in steps if s.get("status") not in ("done", "skipped")
            ] or steps
            cp = _critical_path_secs(live_steps)
            live_est = round(
                sum(s.get("est_sec") or 0.0 for s in live_steps), 1
            )
            # F2: an unestimated step was coerced to 0.0, so a plan with an
            # unestimated prefix was told it FITS on the strength of the work it
            # had not yet priced. Any unestimated non-terminal step makes the
            # verdict unknown — say so instead of certifying a number.
            verdict_known = unestimated == 0
            lines.append(
                f"   计划估时合计 {_fmt_sec(estimated_total)}"
                + (f" + 预留 {_fmt_sec(reserve)}" if reserve else "")
            )
            # Serial sum is only the answer for a pure chain. Report the
            # critical path whenever it differs, so a plan is not called over
            # budget when independent branches could run at the same time.
            if cp["gain"] > 0.5:
                lines.append(
                    f"   关键路径 {_fmt_sec(cp['critical'])}（串行相加 {_fmt_sec(cp['serial'])}）"
                    f" — 并行可省 {_fmt_sec(cp['gain'])}："
                    + _id_list(cp["parallel"])
                    + " 互不依赖，可同时推进（把其中一条分支交给 worker，"
                    "主线程走最长链）"
                )
                bounded = cp["critical"]
            else:
                bounded = live_est
                if len(live_steps) > 2:
                    lines.append(
                        f"   关键路径 = 串行相加 = {_fmt_sec(cp['serial'])}"
                        f"（依赖链是真的，没有可并行的支线；瓶颈在第 "
                        f"{cp['constraint']} 步）"
                    )
            # Judge the LIVE remainder against the REMAINING wall — not the
            # absolute total against the original budget. Mixed units made one
            # render report "over budget by 5m35s" while the drift block below
            # reported "9m25s to spare" for the same plan: finished steps were
            # being charged to a clock that no longer had to pay for them.
            usable_now = remaining - reserve
            # Label follows the clock being judged: with nothing closed yet the
            # remaining wall IS the original budget, so say "over budget";
            # once steps are closed the honest word is "over the remaining
            # allowance" (charging finished steps to a clock that no longer
            # pays for them is the mixed-unit bug).
            nothing_closed = actual_total <= 0
            over_label = (
                f"⚠ 计划已超预算 {_fmt_sec(bounded - usable_now)}"
                if nothing_closed
                else f"⚠ 计划已超剩余可用 {_fmt_sec(bounded - usable_now)}"
            )
            if bounded > usable_now:
                lines.append(
                    over_label
                    + f"（{'关键路径' if cp['gain'] > 0.5 else '剩余估时'} {_fmt_sec(bounded)}"
                    f" > {'预算-预留' if nothing_closed else '剩余-预留'} {_fmt_sec(usable_now)}）"
                )
                if cp["gain"] > 0.5:
                    lines.append(
                        "   → 先扇出：把互不依赖的支线并行做掉再判断。"
                        "只有并行后仍然超，才动范围"
                    )
                else:
                    lines.append(
                        f"   → 并行救不了（依赖链是真的）：砍第 {cp['constraint']} 步的范围，"
                        "或换更便宜的方法，而不是压缩后续每一步"
                    )
            else:
                lines.append(
                    f"   计划余量 {_fmt_sec(usable_now - bounded)}"
                    + (
                        " （按已知估时可完成；未估时的步骤未计入，补上再判断）"
                        if not verdict_known
                        else " （按计划走可完成）"
                    )
                )
        if unestimated:
            lines.append(
                f"   ⚠ {unestimated} 个未完成步骤还没有估时 — 未估时的步骤等于没有计划，"
                f"现在就补 est_sec（剩余可用 {_fmt_sec(remaining - reserve)}，"
                "按这个数给每步分配）"
            )
        elif estimated_total is None:
            # Nothing estimated at all, but a real wall exists: say where the
            # agent stands anyway, because "how much is left" is exactly the
            # fact it needs in order to produce the estimates.
            lines.append(
                f"   未估时的计划无法判断可行性；剩余可用 {_fmt_sec(remaining - reserve)}，"
                "先给每步分配再开工"
            )
        if actual_total > 0:
            lines.append(f"   已实际消耗（已关闭步骤）: {_fmt_sec(actual_total)}")

        # Drift check: are the remaining steps still affordable with what's left?
        pending_est = [
            s.get("est_sec") or 0.0
            for s in steps
            if s.get("status") not in ("done", "skipped") and s.get("est_sec")
        ]
        if pending_est:
            # The same trap as the plan-level total: summing the REMAINING
            # steps end to end overstates what is needed when independent
            # branches can run together. Measure what is actually still on the
            # clock, and separate "needs more time" from "needs a fan-out".
            pending_steps = [
                s for s in steps
                if s.get("status") not in ("done", "skipped")
            ]
            cp_pending = _critical_path_secs(pending_steps)
            need = round(sum(pending_est), 1)
            need_critical = cp_pending["critical"]
            usable = remaining - reserve
            if cp_pending["gain"] > 0.5 and need_critical <= usable < need:
                lines.append(
                    f"   剩余步骤串行需 {_fmt_sec(need)} > 可用 {_fmt_sec(usable)}，"
                    f"但关键路径只要 {_fmt_sec(need_critical)} → 可完成。"
                    " 并行做掉"
                    + _id_list(cp_pending["parallel"])
                    + "：把一条支线交给 worker，主线程走最长链"
                )
            elif need_critical > usable:
                lines.append(
                    f"   ⚠ 漂移: 剩余步骤关键路径还需 {_fmt_sec(need_critical)}"
                    + (f"（串行 {_fmt_sec(need)}）" if need != need_critical else "")
                    + f"，可用只有 {_fmt_sec(usable)}（=剩余-预留）"
                    f" → 差 {_fmt_sec(need_critical - usable)}。"
                )
                if cp_pending["gain"] > 0.5:
                    lines.append(
                        "   先扇出并行支线"
                        + _id_list(cp_pending["parallel"])
                        + "；仍不够再砍范围"
                    )
                else:
                    lines.append(
                        "   依赖链是真的，并行救不了：现在砍范围/换更便宜的方法/"
                        "放弃最贵的一步，不要等余量耗尽"
                    )
            else:
                lines.append(
                    f"   剩余步骤估时需要 {_fmt_sec(need)}，可用 {_fmt_sec(usable)}"
                    f" → 尚余 {_fmt_sec(usable - need)}"
                )
        return lines

    def summary(self) -> str:
        plan = self.get_active()
        if not plan:
            return "No active plan."
        return self._format_plan(plan)

    def _forbidden_lines(self, plan: dict) -> list:
        """Render the agent's own written prohibitions into the plan context.

        Placement is the whole mechanism: this sits right after the time ledger
        and BEFORE the step list, so every turn the agent reads "what I said I
        must not do" beside "how much clock is left" -- the two facts that
        together stop the failure mode where a nearly-out-of-time agent polishes
        what it already has instead of securing a deliverable.

        Deliberately NOT gated on the clock: a prohibition must be in view
        BEFORE the state that would trip it, not only once it is too late.
        """
        items = _normalize_forbidden(plan.get("forbidden"))
        lines: List[str] = []
        if not items:
            lines.append("不做清单 (forbidden) — 尚未声明：")
            lines.append(
                "   计划只写了要做什么。没有阴性约束，时间紧张时就没有防线。"
                "现在用 plan_update(action='add_forbidden') 写下本任务里明确不该做的事"
                "（每条：不该做的动作 + 为什么 + 那种时候改做什么）。"
            )
            lines.append("")
            return lines
        lines.append("不做清单 (forbidden) — 我自己写下的、本任务中不该做的事：")
        for i, e in enumerate(items, 1):
            seg = f"   {i}. ✗ {e.get('action', '')}"
            if e.get("why"):
                seg += f"（因为：{e['why']}）"
            if e.get("instead"):
                seg += f" → 那时改为：{e['instead']}"
            lines.append(seg)
        lines.append("")
        return lines

    def context_for_prompt(self) -> str:
        plan = self.get_active()
        if not plan:
            return ""
        lines = []
        ledger = self._time_ledger_lines(plan)
        if ledger:
            lines.extend(ledger)
            lines.append("")
        lines.extend(self._forbidden_lines(plan))
        thinking = (plan.get("thinking") or "").strip()
        if thinking:
            lines.append("当前问题模型 (thinking):")
            for t_line in thinking.split("\n"):
                lines.append(f"   {t_line}")
            lines.append("")
        for s in plan["steps"]:
            icon = STATUS_ICONS.get(s["status"], " ")
            line = f"{s['id']}. [{icon}] {s['title']}"
            # Per-step time commitment, visible right next to the title so the
            # agent reads "this step was supposed to cost X" every single turn.
            est = s.get("est_sec")
            actual = s.get("actual_sec")
            if est or actual:
                bits = []
                if est:
                    bits.append(f"估{_fmt_sec(est)}")
                if actual is not None:
                    bits.append(f"实{_fmt_sec(actual)}")
                    if est:
                        drift = actual - est
                        bits.append(
                            f"{'超出' if drift > 0 else '省'}{_fmt_sec(abs(drift))}"
                        )
                line += f"  ⏱ {' / '.join(bits)}"
            lines.append(line)
            
            # Show acceptance criteria
            acceptance = s.get("acceptance", [])
            if acceptance:
                lines.append("   验收:")
                for item in acceptance:
                    lines.append(f"     • {item}")
            
            # Show verification evidence
            verification = s.get("verification", [])
            if verification:
                lines.append("   产出:")
                for item in verification:
                    lines.append(f"     ✓ {item}")
            
            # Show notes
            if s.get("notes"):
                for note_line in s["notes"].split("\n"):
                    lines.append(f"   📝 {note_line}")
        return (
            f'<active-plan title="{plan["title"]}">\n'
            + "\n".join(lines)
            + "\n</active-plan>"
        )

    def _format_plan(self, plan: dict) -> str:
        lines = [f"Plan: {plan['title']} [{plan['status']}]"]
        ledger = self._time_ledger_lines(plan)
        if ledger:
            lines.extend(ledger)
        # The negative half of the plan rides the SAME surfaces as the steps:
        # plan_status/reactivate/plan_create all print summary(), so rendering
        # the prohibitions here is what makes the list reachable on demand.
        # (context_for_prompt has no production caller — the live per-turn path
        # builds its own plan block in agent.py _build_plan_context.)
        lines.extend(self._forbidden_lines(plan))
        thinking = (plan.get("thinking") or "").strip()
        if thinking:
            lines.append("  当前问题模型 (thinking):")
            for t_line in thinking.split("\n"):
                lines.append(f"     {t_line}")
        for s in plan["steps"]:
            icon = STATUS_ICONS.get(s["status"], " ")
            line = f"  {s['id']}. [{icon}] {s['title']}"
            est = s.get("est_sec")
            actual = s.get("actual_sec")
            if est or actual:
                bits = []
                if est:
                    bits.append(f"估{_fmt_sec(est)}")
                if actual is not None:
                    bits.append(f"实{_fmt_sec(actual)}")
                    if est:
                        drift = actual - est
                        bits.append(
                            f"{'超出' if drift > 0 else '省'}{_fmt_sec(abs(drift))}"
                        )
                line += f"  ⏱ {' / '.join(bits)}"
            lines.append(line)
            
            # Show acceptance criteria
            acceptance = s.get("acceptance", [])
            if acceptance:
                lines.append("      验收:")
                for item in acceptance:
                    lines.append(f"        • {item}")
            
            # Show verification evidence
            verification = s.get("verification", [])
            if verification:
                lines.append("      产出:")
                for item in verification:
                    lines.append(f"        ✓ {item}")
            
            # Show notes
            if s.get("notes"):
                for note_line in s["notes"].split("\n"):
                    lines.append(f"      📝 {note_line}")
        
        done = sum(1 for s in plan["steps"] if s["status"] in ("done", "skipped"))
        lines.append(f"Progress: {done}/{len(plan['steps'])}")
        return "\n".join(lines)

    @staticmethod
    def _find_step(plan: dict, step_id: int) -> dict:
        for s in plan["steps"]:
            if s["id"] == step_id:
                return s
        raise ValueError(f"Step {step_id} not found")

    def list_plans(self) -> List[dict]:
        plans = []
        for plan in self._list_plans_from_disk():
            steps = plan.get("steps", [])
            plans.append({
                "id": plan.get("id", "?"),
                "title": plan.get("title", ""),
                "status": plan.get("status", "?"),
                "done": sum(1 for s in steps if s.get("status") in ("done", "skipped")),
                "total": len(steps),
                "created": plan.get("created", 0),
            })
        return plans

    def clear_completed(self) -> int:
        if not os.path.isdir(self._dir):
            return 0
        count = 0
        for fname in os.listdir(self._dir):
            if not fname.startswith("plan_") or not fname.endswith(".yaml"):
                continue
            path = os.path.join(self._dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    plan = yaml.safe_load(f)
                if plan.get("status") in ("completed", "abandoned"):
                    os.remove(path)
                    count += 1
            except Exception:
                continue
        return count


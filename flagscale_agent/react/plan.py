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


class TaskPlan:
    """Manages structured task plans with YAML persistence."""

    def __init__(self, plan_dir: str):
        self._dir = plan_dir
        self._lock = threading.RLock()

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

    def create(self, title: str, steps: List[str], session_id: str = "") -> dict:
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
            for i, desc in enumerate(steps, 1):
                step_list.append({
                    "id": i,
                    "title": desc,
                    "status": "pending",
                    "notes": "",
                    "depends_on": [i - 1] if i > 1 else [],
                })

            plan = {
                "id": plan_id,
                "title": title,
                "status": "active",
                "created": time.time(),
                "updated": time.time(),
                "session_id": session_id,
                "steps": step_list,
            }
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

    def update_step(self, step_id: int, status: str, notes: str = "") -> dict:
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")
            if status not in VALID_STEP_STATUSES:
                raise ValueError(f"Invalid status: {status}")

            step = self._find_step(plan, step_id)
            step["status"] = status
            if notes:
                # Append mode: notes accumulate as a log, one line per update
                existing = step.get("notes", "")
                if existing:
                    step["notes"] = existing + "\n" + notes
                else:
                    step["notes"] = notes

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

    def add_steps(self, steps: List[str], after_step_id: Optional[int] = None) -> dict:
        with self._lock:
            plan = self.get_active()
            if not plan:
                raise ValueError("No active plan")

            existing_ids = [s["id"] for s in plan["steps"]]
            next_id = max(existing_ids) + 1 if existing_ids else 1

            new_steps = []
            for i, desc in enumerate(steps):
                sid = next_id + i
                new_steps.append({
                    "id": sid,
                    "title": desc,
                    "status": "pending",
                    "notes": "",
                    "depends_on": [],
                })

            if after_step_id is not None:
                idx = next(
                    (i for i, s in enumerate(plan["steps"]) if s["id"] == after_step_id),
                    None,
                )
                if idx is None:
                    raise ValueError(f"Step {after_step_id} not found")
                for ns in new_steps:
                    ns["depends_on"] = [after_step_id]
                insert_pos = idx + 1
                plan["steps"] = plan["steps"][:insert_pos] + new_steps + plan["steps"][insert_pos:]
            else:
                if plan["steps"]:
                    last_id = plan["steps"][-1]["id"]
                    for ns in new_steps:
                        ns["depends_on"] = [last_id]
                        last_id = ns["id"]
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

    def summary(self) -> str:
        plan = self.get_active()
        if not plan:
            return "No active plan."
        return self._format_plan(plan)

    def context_for_prompt(self) -> str:
        plan = self.get_active()
        if not plan:
            return ""
        lines = []
        for s in plan["steps"]:
            icon = STATUS_ICONS.get(s["status"], " ")
            line = f"{s['id']}. [{icon}] {s['title']}"
            lines.append(line)
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
        for s in plan["steps"]:
            icon = STATUS_ICONS.get(s["status"], " ")
            line = f"  {s['id']}. [{icon}] {s['title']}"
            lines.append(line)
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


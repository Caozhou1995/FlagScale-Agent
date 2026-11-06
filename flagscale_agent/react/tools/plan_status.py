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

"""Plan status tool — show current plan progress."""

from flagscale_agent.react.tools.base import Tool



class PlanStatusTool(Tool):
    name = "plan_status"
    description = (
        "Show the current task plan and progress. "
        "Use at the start of a turn to check where you left off, "
        "or after completing steps to see what's next. Closed steps render as "
        "one line (title + first evidence) — pass step_id to read ONE step "
        "verbatim, including its full notes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "step_id": {
                "type": "integer",
                "description": (
                    "Optional. Read this one step in full (acceptance, all "
                    "evidence, full untruncated notes)."
                ),
            }
        },
    }

    def __init__(self, task_plan):
        self._plan = task_plan

    def execute(self, **kwargs) -> str:
        step_id = kwargs.get("step_id")
        if step_id in (None, ""):
            return self._plan.summary()
        try:
            return self._plan.summary(step_id=int(step_id))
        except (TypeError, ValueError):
            return f"ERROR: step_id must be an integer, got {step_id!r}."
        except Exception as e:
            return f"ERROR: {e}"

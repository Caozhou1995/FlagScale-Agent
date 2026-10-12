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

"""Demand self-reference suppression tests removed.

These tests covered the reviewer/diverger demand suppression that keyed off
the spawned worker's contract while reading the PlanGuard plan_create-time
inject. That inject was removed — the plan-framing and reviewer disciplines
now live in the plan standard template's fixed steps (tools/plan_create.py
`_FRAME_STEP_ACCEPTANCE` / `_REVIEW_STEP_ACCEPTANCE`) and in prompt-level
discipline, so there is no parent-side inject demand to suppress.
"""

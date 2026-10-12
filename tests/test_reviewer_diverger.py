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

"""Diverger machinery tests removed.

The PlanGuard plan_create-time "[Plan] Framing the plan." inject, the
per-framing divergence demand, the diverger dispatch gate
(`_diverger_pending` / `_DIVERGER_INJECT` / `_DIVERGER_ENFORCE`), the
diverger goal auto-tag, and the diverger block re-arm logic were all removed.
The divergence check now lives as one acceptance criterion of the plan
standard template's fixed FIRST step (see tools/plan_create.py
`_FRAME_STEP_ACCEPTANCE`), which the retained plan step_done evidence gate
enforces — one systematic home, no separate guard.

Reviewer-deadline coverage lives in test_reviewer_deadline_guard.py; the
reviewer demand is prompt-level discipline (react/prompt.py).
"""

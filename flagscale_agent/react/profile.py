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

"""WorkerProfile — parameterizes WorkerAgent behavior by persona.

A WorkerProfile is NOT a separate Agent class. It's a configuration that tells
a single WorkerAgent class which skills to load, which Checklist to use, and
which scene constraints to activate.

Adding a new scenario = registering a new WorkerProfile. No new class needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkerProfile:
    """Defines a worker persona — not a separate class."""

    name: str
    description: str
    skills: list[str]  # Skills to preload


# ── Registered profiles ────────────────────────────────────────────────────

PROFILES: dict[str, WorkerProfile] = {
    "general": WorkerProfile(
        name="general",
        description="General-purpose tasks: shell operations, file inspection, simple Q&A, cleanup",
        skills=[],
    ),
    "model-migration": WorkerProfile(
        name="model-migration",
        description="Migrate models from source frameworks to FlagScale/Megatron-Core",
        skills=["train-model-porter", "train-config", "train-data-prep"],
    ),
    "training-reproduce": WorkerProfile(
        name="training-reproduce",
        description="Reproduce training results from papers/reference implementations",
        skills=["train-reproduce", "train-config", "train-run"],
    ),
    "train-env-setup": WorkerProfile(
        name="train-env-setup",
        description="Detect hardware, create environments, install dependencies",
        skills=["train-env-setup", "workspace-layout"],
    ),
    "inference-deploy": WorkerProfile(
        name="inference-deploy",
        description="Deploy inference services with vllm/sglang (coming soon)",
        skills=["train-config"],  # future: inference-specific skills
    ),
}

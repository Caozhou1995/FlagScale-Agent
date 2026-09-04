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

"""Tests for the task-level time-budget health block and the _health_judge
kwarg-forwarding fix (regression: the LLM judge was silently dead because
shell.py passed kwargs _health_judge did not accept)."""

from flagscale_agent.react.judge import Judge


class CapturingProvider:
    """Records the FULL prompt of each call, returns a fixed response."""

    def __init__(self, response='{"kill": false, "reason": ""}'):
        self.response = response
        self.prompts = []

    def chat(self, messages, tools=None):
        self.prompts.append(messages[-1]["content"])
        return {"content": self.response}


class TestTaskBudgetBlock:
    def test_no_budget_block_is_empty(self):
        # No budget summary -> byte-identical to pre-budget prompt (backward compat).
        assert Judge._build_task_budget_block("") == ""
        assert Judge._build_task_budget_block(None) == ""
        assert Judge._build_task_budget_block("   \n ") == ""

    def test_budget_block_contains_summary_text(self):
        block = Judge._build_task_budget_block(
            "cumulative task time elapsed 50m of total budget 1h00m (83% used)"
        )
        assert "50m" in block and "83% used" in block
        assert "budget" in block.lower()

    def test_budget_block_distinct_from_per_command(self):
        # Must frame a TASK-LEVEL, cumulative, whole-task judgment and explicitly
        # contrast with the per-command rate/ETA criterion.
        low = Judge._build_task_budget_block("x").lower()
        assert "cumulative" in low
        assert "task-level" in low or "task level" in low
        assert "per-command" in low or "this command" in low

    def test_budget_block_prefers_advisory_not_kill(self):
        low = Judge._build_task_budget_block("x").lower()
        assert "advisory" in low or "kill=false" in low
        assert "method-class" in low or "method class" in low

    def test_budget_block_excludes_hw_net_limits(self):
        low = Judge._build_task_budget_block("x").lower()
        assert "hardware" in low and "network" in low
        assert "configurable" in low

    def test_budget_block_forbids_shrinking_task(self):
        low = Judge._build_task_budget_block("x").lower()
        assert "never" in low
        # must forbid lowering quality/accuracy or cutting required input
        assert "quality" in low or "accuracy" in low
        assert "requirement" in low or "input" in low

    def test_budget_block_is_generic_no_task_specifics(self):
        block = Judge._build_task_budget_block("x").lower()
        for leaked in ("fasttext", "caffe", "cifar", "yelp", "epoch"):
            assert leaked not in block


class TestHealthBudgetInjection:
    def test_health_injects_budget_into_prompt(self):
        provider = CapturingProvider()
        judge = Judge(provider)
        judge.health(
            "run something", "progress 5%", "3m", True, 0,
            task_budget="cumulative task time elapsed 50m of total budget 1h",
        )
        prompt = provider.prompts[0]
        assert "cumulative task time elapsed 50m" in prompt
        assert "Task-level time budget" in prompt

    def test_health_without_budget_omits_block(self):
        provider = CapturingProvider()
        judge = Judge(provider)
        judge.health("run something", "progress 5%", "3m", True, 0)
        prompt = provider.prompts[0]
        assert "Task-level time budget" not in prompt

    def test_health_accepts_shell_loop_kwargs(self):
        # Regression: shell.py forwards command_history + container_resources.
        # judge.health must accept them without TypeError.
        provider = CapturingProvider()
        judge = Judge(provider)
        result = judge.health(
            "cmd", "out", "5m", True, 0,
            command_history="ran 3x, killed 2x",
            container_resources="8GB RAM, 4 CPU",
            task_budget="50m of 1h",
        )
        assert isinstance(result, dict)

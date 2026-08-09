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

"""Tests for simplified Judge — classify, health, caching."""

import pytest

from flagscale_agent.react.judge import Judge, _CLASSIFY_PROMPTS, _HEALTH_PROMPT


class MockProvider:
    """Returns controlled JSON responses in sequence."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append(messages[-1]["content"][:200])
        if self.responses:
            return {"content": self.responses.pop(0)}
        return {"content": '{"real": false}'}


# ── Judge.classify ────────────────────────────────────────────────────────


class TestJudgeClassify:
    def test_classify_calls_provider(self):
        provider = MockProvider(['{"real": true}'])
        judge = Judge(provider)
        result = judge.classify("is_fatal", {"command": "rm -rf /"})
        assert result is True
        assert len(provider.calls) == 1

    def test_classify_returns_false(self):
        provider = MockProvider(['{"real": false}'])
        judge = Judge(provider)
        result = judge.classify("is_dangerous", {"command": "ls"})
        assert result is False

    def test_classify_uses_cache(self):
        provider = MockProvider(['{"real": true}', '{"real": false}'])
        judge = Judge(provider)
        r1 = judge.classify("is_fatal", {"command": "rm -rf /"})
        r2 = judge.classify("is_fatal", {"command": "rm -rf /"})
        assert r1 is True
        assert r2 is True  # cached
        assert len(provider.calls) == 1  # only 1 LLM call

    def test_classify_different_context_not_cached(self):
        provider = MockProvider(['{"real": true}', '{"real": false}'])
        judge = Judge(provider)
        judge.classify("is_fatal", {"command": "rm -rf /"})
        judge.classify("is_fatal", {"command": "ls"})
        assert len(provider.calls) == 2

    def test_classify_returns_default_on_parse_failure(self):
        provider = MockProvider(["not json at all"])
        judge = Judge(provider)
        result = judge.classify("is_dangerous", {"command": "something"}, default=False)
        assert result is False

    def test_classify_unknown_category_returns_default(self):
        provider = MockProvider()
        judge = Judge(provider)
        result = judge.classify("nonexistent_category", {}, default="fallback")
        assert result == "fallback"

    def test_reset_turn_clears_cache(self):
        provider = MockProvider(['{"real": true}', '{"real": false}'])
        judge = Judge(provider)
        judge.classify("is_fatal", {"command": "rm -rf /"})
        judge.reset_turn()
        judge.classify("is_fatal", {"command": "rm -rf /"})
        assert len(provider.calls) == 2  # cache was cleared


# ── Judge.health ──────────────────────────────────────────────────────────


class TestJudgeHealth:
    def test_health_returns_dict(self):
        provider = MockProvider(['{"kill": false, "reason": "", "next_check_seconds": 30}'])
        judge = Judge(provider)
        result = judge.health("python train.py", "loss: 2.3", "5m", True, 0)
        assert result["kill"] is False

    def test_health_kill_decision(self):
        provider = MockProvider(['{"kill": true, "reason": "stalled", "next_check_seconds": 10}'])
        judge = Judge(provider)
        result = judge.health("python train.py", "", "30m", False, 5)
        assert result["kill"] is True
        assert "stalled" in result["reason"]

    def test_health_default_on_failure(self):
        provider = MockProvider(["garbage"])
        judge = Judge(provider)
        result = judge.health("cmd", "out", "1m", True, 0)
        assert result == {"kill": False}


# ── Prompts exist ─────────────────────────────────────────────────────────


class TestPrompts:
    def test_required_prompts_exist(self):
        assert "is_fatal" in _CLASSIFY_PROMPTS
        assert "is_dangerous" in _CLASSIFY_PROMPTS

    def test_health_prompt_exists(self):
        assert _HEALTH_PROMPT
        assert "{command}" in _HEALTH_PROMPT

    def test_prompts_have_context_placeholder(self):
        for name, prompt in _CLASSIFY_PROMPTS.items():
            assert "{command}" in prompt, f"Prompt {name} missing {{command}} placeholder"

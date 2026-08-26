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

    def test_prompts_have_placeholders(self):
        for name, prompt in _CLASSIFY_PROMPTS.items():
            assert "{" in prompt, f"Prompt {name} missing placeholders"

    def test_health_prompt_has_stalled_progress_kill_criterion(self):
        # Regression: a brute-force search ran with a progress counter that never
        # advanced; the health judge had no kill criterion for a stalled compute
        # loop and waited it out until the hard timeout. The kill criterion must be
        # observable (a non-advancing counter) and must distinguish itself from the
        # patient COMPILING/INSTALLING cases.
        low = _HEALTH_PROMPT.lower()
        assert "progress counter" in low or "progress indicator" in low
        assert "not advanc" in low or "not moved" in low
        # must explicitly contrast with the patient no-counter cases
        assert "compiling" in low or "installing" in low

    def test_health_prompt_requires_actionable_kill_reason(self):
        # Regression: after a kill, the agent relaunched near-identical variants
        # (attack2 -> attack3 -> attack4) because the kill reason only described
        # the symptom. The prompt must instruct the judge to write a reason that
        # redirects to an alternative METHOD CLASS, without prescribing a command.
        low = _HEALTH_PROMPT.lower()
        assert "actionable" in low
        # names a class/direction of alternative, not a specific command
        assert "class of alternative" in low or "class of" in low
        assert "do not prescribe the exact command" in low

    def test_health_prompt_has_slow_rate_kill_criterion(self):
        # Regression: a long-running job advanced (counter moving, metric even
        # improving) but at a rate whose projected completion far exceeded the
        # allowed runtime, so it was doomed to hit an external timeout with no
        # output. The old prompt judged this healthy and waited it out. The judge
        # must have a kill criterion for progress that IS advancing but whose
        # rate/ETA cannot finish within budget.
        low = _HEALTH_PROMPT.lower()
        assert "eta" in low
        # must key off rate, not just a frozen counter
        assert "rate" in low
        # must exclude genuine hardware / network hard limits (not the agent's fault)
        assert "hardware" in low and "network" in low
        # must attribute the killable case to a configurable choice the agent owns
        assert "configurable" in low
        # must contrast with the stalled-counter criterion above it
        assert "advancing" in low

    def test_health_prompt_prefers_advisory_when_uncertain_hw_vs_config(self):
        # The slow-rate criterion must NOT fire on hardware-bound jobs. When it is
        # ambiguous whether slowness is hardware or a config choice, the prompt
        # should degrade to a non-killing advisory that prompts the agent to
        # reconsider, rather than killing a legitimately hardware-bound job.
        low = _HEALTH_PROMPT.lower()
        assert "cannot tell" in low or "when in doubt" in low or "when uncertain" in low
        assert "advisory" in low or "kill=false" in low

    def test_health_returns_slow_rate_kill(self):
        # Behavioral: the judge surfaces a rate/ETA-based kill decision produced by
        # the provider (parsing + passthrough of the redirect reason).
        provider = MockProvider([
            '{"kill": true, "reason": "rate too low for budget; slowness is a '
            'configurable choice, reconsider settings", "next_check_seconds": 10}'
        ])
        judge = Judge(provider)
        out = "Progress: 10.0% ETA: long"
        result = judge.health("some-long-running-command", out, "8m", True, 0)
        assert result["kill"] is True
        assert "configurable" in result["reason"].lower()


# ── Expectation anchor (declared-vs-observed) ─────────────────────────────


class CapturingProvider:
    """Records the FULL prompt of each call, returns a fixed response."""

    def __init__(self, response='{"kill": false, "reason": "", "next_check_seconds": 30}'):
        self.response = response
        self.prompts = []

    def chat(self, messages, tools=None):
        self.prompts.append(messages[-1]["content"])
        return {"content": self.response}


class TestExpectationAnchor:
    def test_no_anchor_block_is_empty(self):
        # No declared expectation -> block is empty string, so the health prompt
        # is byte-identical to the pre-anchor version (no leakage, no behavior
        # change for commands run without a plan).
        assert Judge._build_expectation_block("") == ""
        assert Judge._build_expectation_block(None) == ""
        assert Judge._build_expectation_block("   \n  ") == ""

    def test_anchor_block_contains_declared_text(self):
        block = Judge._build_expectation_block("finish in ~10min, acc>=0.62")
        assert "finish in ~10min, acc>=0.62" in block
        assert "anchor" in block.lower()

    def test_anchor_block_frames_both_axes_and_prefers_advisory(self):
        # The anchor must be tested along TIME and QUALITY axes, treated as a
        # hypothesis (not ground truth), and drift should trigger an advisory that
        # questions the METHOD-CLASS rather than an outright kill.
        low = Judge._build_expectation_block("some declared goal").lower()
        assert "time" in low and "quality" in low
        assert "method-class" in low or "method class" in low
        assert "advisory" in low or "kill=false" in low
        # framed as a testable hypothesis, can also mean the anchor was wrong
        assert "hypothesis" in low
        assert "anchor was wrong" in low or "may equally mean" in low

    def test_anchor_is_generic_no_task_specifics(self):
        # The block frames the mechanism generically; concrete task wording only
        # enters via the operator's own declared anchor, never hardcoded.
        block = Judge._build_expectation_block("x").lower()
        for leaked in ("fasttext", "words/sec", "yelp", "epoch", "loss"):
            assert leaked not in block

    def test_health_injects_anchor_into_prompt(self):
        provider = CapturingProvider()
        judge = Judge(provider)
        judge.health(
            "run something", "progress 5%", "3m", True, 0,
            expectation="expect completion in ~5min",
        )
        prompt = provider.prompts[0]
        assert "expect completion in ~5min" in prompt
        assert "Declared expectation" in prompt

    def test_health_without_anchor_omits_block(self):
        provider = CapturingProvider()
        judge = Judge(provider)
        judge.health("run something", "progress 5%", "3m", True, 0)
        prompt = provider.prompts[0]
        assert "Declared expectation" not in prompt

    def test_health_anchor_defaults_to_empty(self):
        # Backward compatibility: existing callers that don't pass expectation
        # get the no-anchor prompt.
        provider = CapturingProvider()
        judge = Judge(provider)
        judge.health("cmd", "out", "1m")
        assert "Declared expectation" not in provider.prompts[0]


# ── Live resource signals (silent-compute vs hung) ────────────────────────


class TestActivityBlock:
    def test_no_activity_block_is_empty(self):
        # No live-signal summary -> block is empty string, so the health prompt
        # is byte-identical to the pre-activity version (no leakage, no behavior
        # change for callers that don't supply resource signals).
        assert Judge._build_activity_block("") == ""
        assert Judge._build_activity_block(None) == ""
        assert Judge._build_activity_block("   \n  ") == ""

    def test_activity_block_contains_signal_text(self):
        block = Judge._build_activity_block(
            "CPU 190%, memory 512 MB, live child processes 2 of 2"
        )
        assert "CPU 190%" in block
        assert "live child processes 2 of 2" in block

    def test_activity_block_distinguishes_hung_from_silent(self):
        # The block must frame the two cases: HUNG (no output + no CPU/child)
        # vs HEALTHY-SILENT (no output but sustained CPU/children), and forbid
        # killing the silent-compute case or inventing a monitoring deadline.
        low = Judge._build_activity_block("some signal").lower()
        assert "hung" in low
        assert "silent" in low
        assert "do not kill" in low
        # must reject the invented per-command monitoring window / deadline
        assert "monitoring window" in low or "deadline" in low

    def test_activity_block_is_generic_no_task_specifics(self):
        # Generic mechanism only; concrete numbers enter via the caller's own
        # signal string, never hardcoded task wording.
        block = Judge._build_activity_block("x").lower()
        for leaked in ("fasttext", "words/sec", "yelp", "epoch", "ngram", "verbose"):
            assert leaked not in block

    def test_health_injects_activity_into_prompt(self):
        provider = CapturingProvider()
        judge = Judge(provider)
        judge.health(
            "run something", "", "6m", False, 8,
            activity="CPU 200%, memory 900 MB, live child processes 3 of 3",
        )
        prompt = provider.prompts[0]
        assert "CPU 200%" in prompt
        assert "Live resource signals" in prompt

    def test_health_without_activity_omits_block(self):
        provider = CapturingProvider()
        judge = Judge(provider)
        judge.health("run something", "", "6m", False, 8)
        prompt = provider.prompts[0]
        assert "Live resource signals" not in prompt

    def test_health_activity_defaults_to_empty(self):
        # Backward compatibility: callers that don't pass activity get the
        # no-activity prompt.
        provider = CapturingProvider()
        judge = Judge(provider)
        judge.health("cmd", "out", "1m")
        assert "Live resource signals" not in provider.prompts[0]


# ── Silence-is-not-a-stall + anti-downgrade prompt rules ──────────────────


class TestSilenceAndAntiDowngradePrompt:
    def test_prompt_has_silence_is_not_a_stall_section(self):
        low = _HEALTH_PROMPT.lower()
        assert "silence is not by itself a stall" in low
        # a positive sign of being stuck is required before reading silence as
        # a stall; silence alone is never a kill trigger
        assert "positive sign" in low
        assert "kill trigger" in low

    def test_prompt_forbids_inventing_monitoring_window(self):
        low = _HEALTH_PROMPT.lower()
        assert "monitoring window" in low or "deadline" in low

    def test_kill_reason_forbids_downgrading_the_task(self):
        # The kill reason must never tell the agent to shrink the work, lower
        # the task's quality target, or emit fake output to look alive; it may
        # only redirect to a faster METHOD that still meets every requirement.
        low = _HEALTH_PROMPT.lower()
        assert "never" in low
        assert "faster method" in low or "cheaper" in low
        assert "requirement" in low

    def test_prompt_stays_generic_no_task_specifics(self):
        low = _HEALTH_PROMPT.lower()
        for leaked in ("fasttext", "words/sec", "yelp", "ngram"):
            assert leaked not in low

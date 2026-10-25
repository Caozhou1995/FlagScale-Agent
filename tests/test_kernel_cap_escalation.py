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

"""Tests for the escalating recovery after a runtime thinking cap.

Measured failure this closes (write-compressor, v12.2 batch 2026-10-26
04-03-20, trial EK83dj8): the in-stream wall-clock cap DID fire (two
pure-thinking streams of 325.3s and 316.6s were cut), but the ONE-SHOT
convergence nudge was ignored — the model re-read the SAME ~53KB of reasoning
re-injected by the resume path and spent another 316.6s thinking. The trial
total: 4 LLM round-trips, ZERO tool calls, 707s, verifier 3 failed (no
/app/data.comp). The saturation branch already carried the fix (bounded recap
+ repeat counter + HARD STOP) but this endpoint never trips `was_saturated`
(out_tok stays under 95% of max_output_tokens), so the escalation has to be
driven by the cap count itself.

Behavior contract:
  - 1st cap in a turn -> convergence nudge with a BOUNDED reasoning tail
    (<= 4000 chars), never the whole stream, and no HARD STOP wording
  - 2nd+ cap in the same turn -> HARD STOP nudge demanding exactly one tool
    call that produces state on disk
  - a response with real visible output resets the streak (the model broke out)
  - the streak is per-turn (initialised in run_turn)
"""

import types
from unittest.mock import MagicMock

from flagscale_agent.react.kernel import AgentKernel, KernelDeps
from flagscale_agent.react.history import HistoryManager
from flagscale_agent.react.guard import GuardRegistry


class _Harness:
    """Real HistoryManager driven through AgentKernel.run_turn."""

    def __init__(self):
        self.history = HistoryManager(max_context_tokens=64000)
        self.history.append({"role": "user", "content": "do the task"})
        self.registry = GuardRegistry()
        self.config = types.SimpleNamespace(
            max_iterations=20, max_continuations=5, mode="auto", _turn_count=0,
        )
        self.call_count = {"n": 0}

    def build(self, llm_responses):
        def call_llm_fn(messages, schemas):
            idx = min(self.call_count["n"], len(llm_responses) - 1)
            self.call_count["n"] += 1
            return llm_responses[idx], {"input_tokens": 1, "output_tokens": 1}

        provider = MagicMock()
        provider.format_assistant_message.side_effect = lambda resp: {
            "role": "assistant", "content": resp.get("content") or "",
        }

        deps = KernelDeps(
            provider=provider,
            history=self.history,
            tool_registry=MagicMock(),
            judge=MagicMock(),
            guard_registry=self.registry,
            config=self.config,
            display=MagicMock(),
            get_schemas_fn=lambda: [],
            inject_message_fn=lambda msg: self.history.append(
                {"role": "user", "content": msg}),
            append_tool_results_fn=lambda results: None,
            format_tool_result_fn=lambda tid, r: {},
            execute_tools_fn=lambda tcs: ["ok"] * len(tcs),
            is_context_limit_error_fn=lambda e: False,
            call_llm_fn=call_llm_fn,
        )
        return AgentKernel(deps)

    def user_messages(self):
        return [m.get("content") or "" for m in self.history.messages
                if m.get("role") == "user"]


_DONE = {"content": "[TASK_COMPLETE]", "tool_calls": []}


def _capped(long_reasoning=False):
    """A runtime-capped reasoning-only response."""
    text = ("reasoning step. " * 400) if long_reasoning else "short reasoning"
    return {"content": None, "tool_calls": None, "thinking": text,
            "reasoning_only": True, "thinking_capped": True}


class TestCapRecoveryEscalation:
    def test_first_cap_uses_bounded_tail_not_whole_stream(self):
        """The 1st cap must inject a BOUNDED tail of the reasoning (the whole
        stream re-fed is what kept the v12.2 model thinking), and must NOT
        use the HARD STOP wording reserved for repeat offenders."""
        h = _Harness()
        kernel = h.build([_capped(long_reasoning=True), _DONE])
        result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        caps = [m for m in h.user_messages() if "previous_reasoning" in m]
        assert len(caps) == 1
        body = caps[0].split("<previous_reasoning>", 1)[1]
        body = body.split("</previous_reasoning>", 1)[0].strip()
        assert len(body) <= 4000, len(body)
        assert "HARD STOP" not in caps[0]
        # Still asks for a concrete step (unchanged contract)
        assert "Land the next concrete step NOW" in caps[0]

    def test_second_cap_in_same_turn_escalates_to_hard_stop(self):
        """Two caps with no visible output in between -> the 2nd nudge is the
        HARD STOP demanding exactly one state-producing tool call."""
        h = _Harness()
        kernel = h.build([_capped(), _capped(), _DONE])
        result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        hard = [m for m in h.user_messages() if "HARD STOP" in m]
        assert len(hard) == 1
        assert "exactly ONE tool call" in hard[0]
        assert "write_file" in hard[0]
        assert "reasoning has now hit the runtime cap 2 times" in hard[0]

    def test_visible_output_resets_the_streak(self):
        """A response carrying real text breaks the loop: a later cap starts
        the escalation over (1st-level nudge, not HARD STOP)."""
        h = _Harness()
        kernel = h.build([
            _capped(),
            {"content": "Here is progress: wrote a file.", "tool_calls": None,
             "thinking": "x", "reasoning_only": False},
            _capped(),
            _DONE,
        ])
        result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        assert len([m for m in h.user_messages() if "HARD STOP" in m]) == 0
        assert kernel._cap_text_streak == 0

    def test_streak_is_reinitialised_per_turn(self):
        """White-box: the counter lives per turn, so a fresh run_turn starts
        clean even if a prior turn left it dirty."""
        h = _Harness()
        kernel = h.build([_DONE])
        kernel._cap_text_streak = 7
        result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        assert kernel._cap_text_streak == 0

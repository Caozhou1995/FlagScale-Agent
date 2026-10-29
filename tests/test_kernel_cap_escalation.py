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
        assert "Your next message MUST contain exactly ONE tool call that" in caps[0]
        # The first-cap directive must FORCE one action, not offer a menu:
        # "either call a tool ... or write your answer" is the wording that let
        # the model keep deriving (measured in the long-derivation failure
        # family: 4.5k-line trajectory, 2 read-only shell calls, no artifact).
        assert "either call a tool to" not in caps[0]
        assert "write_file" in caps[0]
        assert "imperfect file on disk" in caps[0]

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

    def test_prose_progress_does_not_reset_the_streak(self):
        """A response carrying real TEXT but no tool call does NOT clear the
        cap streak. Measured (round-2 rerun 2026-10-29 19-12-23,
        regex-chess__nqvTfjB): cap #1 -> first-rung recap -> agent emitted
        visible prose -> cap #2 found the streak reset and served the SAME
        first-rung recap, so the HARD STOP rung was never reached and the
        deliverable was never written. Prose is not progress."""
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
        # The second cap must escalate, and the counter must NOT be zeroed by
        # mere visible output.
        assert len([m for m in h.user_messages() if "HARD STOP" in m]) == 1
        assert kernel._cap_text_streak == 2

    def test_state_producing_call_resets_the_streak(self):
        """A SUCCESSFUL state-producing call (write_file/edit_file) is the one
        event that clears the streak: that is evidence of progress, and a later
        cap then legitimately restarts the escalation at rung 1."""
        h = _Harness()
        kernel = h.build([
            _capped(),
            {"content": "", "tool_calls": [
                {"id": "t1", "name": "write_file", "arguments": {}}],
             "thinking": "x", "reasoning_only": False},
            _capped(),
            _DONE,
        ])
        kernel.run_turn()
        # 1st cap set streak=1, the write_file cleared it, 2nd cap set it to 1.
        assert kernel._cap_text_streak == 1
        assert len([m for m in h.user_messages() if "HARD STOP" in m]) == 0

    def test_streak_is_reinitialised_per_turn(self):
        """White-box: the counter lives per turn, so a fresh run_turn starts
        clean even if a prior turn left it dirty."""
        h = _Harness()
        kernel = h.build([_DONE])
        kernel._cap_text_streak = 7
        result = kernel.run_turn()
        assert result.stop_reason == "explicit_signal"
        assert kernel._cap_text_streak == 0

    def test_hard_stop_arms_the_structural_channel_constraint(self):
        """The HARD STOP rung must not rely on wording alone.

        Measured (round 3, regex-chess 2026-10-29 19-34-50): three token caps
        in one turn, acts(grep -c '⚡') stayed at 2, and after EVERY cap the
        agent wrote "I need to stop reasoning and act" and then emitted another
        2000+ characters of design prose. Compliance talk, no action. So the
        HARD STOP branch must also arm the structural channel constraint, which
        arrives at the provider as force_tool=True on the NEXT request.
        """
        h = _Harness()
        kernel = h.build([_capped(), _capped(), _DONE])
        seen = []

        def call_llm_fn(messages, schemas, force_tool=False):
            seen.append(force_tool)
            idx = min(h.call_count["n"], 2)
            h.call_count["n"] += 1
            return [_capped(), _capped(), _DONE][idx], {
                "input_tokens": 1, "output_tokens": 1}

        kernel.deps.call_llm_fn = call_llm_fn
        kernel.run_turn()
        # Call 1 (before any cap) is not forced; the call AFTER the HARD STOP
        # rung fired (cap #2) carries force_tool=True.
        assert seen[0] is False
        assert True in seen, seen

    def test_first_cap_does_not_arm_the_structural_constraint(self):
        """The first rung stays advisory: a cooperative model that was merely
        long-winded must not be jammed into a forced tool call."""
        h = _Harness()
        kernel = h.build([_capped(), _DONE])
        seen = []

        def call_llm_fn(messages, schemas, force_tool=False):
            seen.append(force_tool)
            idx = min(h.call_count["n"], 1)
            h.call_count["n"] += 1
            return [_capped(), _DONE][idx], {
                "input_tokens": 1, "output_tokens": 1}

        kernel.deps.call_llm_fn = call_llm_fn
        kernel.run_turn()
        assert seen == [False, False], seen

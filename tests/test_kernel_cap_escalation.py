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
        assert "Your next message MUST contain exactly ONE tool call" in caps[0]
        # The first-cap directive must FORCE one action, not offer a menu:
        # "either call a tool ... or write your answer" is the wording that let
        # the model keep deriving (measured in the long-derivation failure
        # family: 4.5k-line trajectory, 2 read-only shell calls, no artifact).
        assert "either call a tool to" not in caps[0]
        # Convergence must be defined as LANDING A FILE, not as "a shorter
        # thought" (user directive 2026-11-05): the schemelike trial spent six
        # 3000-token caps deriving in-stream before it finally wrote. The
        # directive now names the write as the default and the probe as the
        # exception.
        assert "CONVERGE means LAND A FILE" in caps[0]
        assert "`write_file`" in caps[0]
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

        def call_llm_fn(messages, schemas, force_tool=False, **kw):
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
        """Contract updated (round 7, regex-chess RK7GQoK): a PURE-THINKING
        first cap (no visible text) MUST arm the structural channel on the
        next call. The soft rung was measured dead on this population:
        cap-retry 1 soft nudge -> 172.5s more pure thinking -> idle-retry 2
        (structural, force_tool only) -> 240.3s more thinking; 612s = 68% of
        a 900s budget gone with zero tool calls. The force_tool kwarg AND the
        thinking_cap_override squeeze must both arrive on the next call.

        A cooperative model is NOT harmed: this branch is only entered when
        the response carried no visible text at all; a model that answers a
        nudge emits visible prose and never re-enters it."""
        h = _Harness()
        kernel = h.build([_capped(), _DONE])
        seen = []

        def call_llm_fn(messages, schemas, force_tool=False,
                        thinking_cap_override=None):
            seen.append((force_tool, thinking_cap_override))
            idx = min(h.call_count["n"], 1)
            h.call_count["n"] += 1
            return [_capped(), _DONE][idx], {
                "input_tokens": 1, "output_tokens": 1}

        kernel.deps.call_llm_fn = call_llm_fn
        kernel.run_turn()
        # Call 2 (after the capped retry) must carry force_tool=True AND the
        # 3000-token squeeze — the old silent degrade (mock without **kw
        # raising TypeError) is what let this regression hide: the degrade
        # branch dropped the constraint and the assertion still saw False.
        assert seen[1] == (True, 3000), seen


def test_production_path_forwards_force_tool_to_provider():
    """Regression: the REAL call path (call_llm_fn=None → agent's
    _call_llm_stream) previously used a 2-arg fallback lambda and a 2-arg
    _call_llm_stream signature, so the 3-arg forced call raised TypeError and
    the degrade branch silently DROPPED the constraint — the flag never
    reached provider.chat_stream even though every provider accepted it.

    Measured (round 4, regex-chess 2026-10-29 20-12-42): kernel armed
    _force_tool_next on the HARD STOP rung, the stream then ran as plain
    chat_stream, and the model kept emitting design prose with acts=0.
    """
    import inspect
    from flagscale_agent.react.agent import WorkerAgent

    # The production stream entry must accept and forward the flag.
    sig = inspect.signature(WorkerAgent._call_llm_stream)
    assert "force_tool" in sig.parameters, sig
    # And it must pass the flag down to the provider's chat_stream — but only
    # on the armed path: the unforced call keeps its exact 2-arg form
    # (regression: unconditional forwarding broke 13 mock-lambda tests).
    src = inspect.getsource(WorkerAgent._call_llm_stream)
    assert "force_tool=True" in src, "forced call no longer carries the flag"

    # The kernel fallback lambda must accept **kwargs (a 2-arg lambda would
    # make the forced call raise TypeError and drop the flag).
    from flagscale_agent.react import kernel as kernel_mod
    ksrc = inspect.getsource(kernel_mod)
    assert "lambda m, s, **kw:" in ksrc, "fallback lambda still drops kwargs"


def test_force_tool_stands_down_both_client_caps():
    """Measured (round 5, regex-chess trial gfpT6ap): the endpoint forces
    thinking ON server-side, so a forced call still thinks FIRST — and the
    client-side token cap + in-stream wall-clock cap both abort the stream
    during that thinking, so the mandatory tool call is never reached and
    the enforcement is toothless (two caps fired, acts stayed at 2). When
    force_tool is armed, the client caps must stand down; the thinking is
    bounded by the server-side max_tokens instead."""
    import inspect
    from flagscale_agent.react.agent import WorkerAgent

    src = inspect.getsource(WorkerAgent._call_llm_stream)
    # Token cap stands down under the flag (but a squeeze override WINS —
    # see test_agent_applies_override_over_force_tool_exemption).
    assert "cap_budget = 0" in src and "force_tool" in src, src[:400]
    # Silence watchdog stands down under the flag (and the thread start and
    # the event-level check share the same computed limit).
    assert "_wd_limit = 0 if force_tool" in src, src[:400]
    assert "if _wd_limit > 0:" in src, "watchdog thread start not gated on _wd_limit"
    assert "and _wd_limit > 0" in src, "event-level wall check not gated on _wd_limit"


def test_hard_stop_squeezes_the_thinking_budget():
    """Squeeze mechanism (user directive 2026-10-30): when the escalation
    ladder reaches HARD STOP and still no write/edit has landed, the next
    streams must run with a SQUEEZED thinking budget (3000) instead of the
    configured one — measured round 5 (regex-chess): 11 token caps at 16384,
    1 real action, 0 TASK_COMPLETE. The tool_choice channel was toothless
    (server forces thinking ON; caps abort the stream mid-thinking), so the
    budget itself must shrink until state is produced.

    Contract updated (round 6, trial SbvVjVR): the override is FORWARDED on
    the forced call and consumed (cleared) — it must not linger on the
    kernel instance, because a lingering copy would re-squeeze every later
    call even after the agent-side branch stops seeing it."""
    h = _Harness()
    kernel = h.build([_capped(), _capped(), _DONE])
    seen = {}
    real_call = kernel.deps.call_llm_fn

    def _spy_call(messages, schemas, **kw):
        seen.update(kw)
        return real_call(messages, schemas, **kw)

    kernel.deps.call_llm_fn = _spy_call
    kernel.run_turn()
    assert seen.get("thinking_cap_override") == 3000, seen
    # Consumed on use: the attribute must be cleared after forwarding.
    assert getattr(kernel, "_thinking_cap_override", None) is None, \
        kernel.__dict__.get("_thinking_cap_override")


def test_state_producing_call_releases_the_squeeze():
    """Once a write/edit finally produces state on disk, the squeeze must be
    RELEASED (override cleared) so later reasoning gets the normal budget —
    the squeeze is a lever for one failure mode, not a permanent handicap."""
    h = _Harness()
    kernel = h.build([
        _capped(), _capped(),
        {"content": "", "tool_calls": [
            {"id": "t1", "name": "write_file", "arguments": {}}],
         "thinking": "x", "reasoning_only": False},
        _DONE,
    ])
    kernel.run_turn()
    assert kernel._thinking_cap_override is None, kernel._thinking_cap_override
    # And the streak reset that comes with the same evidence still holds.
    assert kernel._cap_text_streak == 0


def test_agent_applies_override_over_force_tool_exemption():
    """Ordering contract: the squeeze override WINS over the force_tool cap
    exemption. A squeeze that gets exempted away re-creates the
    unlimited-thinking failure it was built to fix (the exemption sets
    cap_budget=0, which disables the cap entirely)."""
    import inspect
    from flagscale_agent.react.agent import WorkerAgent

    src = inspect.getsource(WorkerAgent._call_llm_stream)
    # The override branch must precede the exemption branch in the source.
    assert "_thinking_cap_override" in src, src[:400]
    _i_override = src.index("_thinking_cap_override")
    _i_exempt = src.index("cap_budget = 0")
    assert _i_override < _i_exempt, \
        "override must be checked BEFORE the force_tool exemption"
    # The squeeze value must come from the CALL kwarg (forwarded by the
    # kernel on the forced call), NOT from getattr(self, ...) — the HARD
    # STOP rung lives on the KERNEL instance, so reading the agent instance
    # always saw None and the squeeze printed "0 token cap" (round 6,
    # trial SbvVjVR). The old broken form was `_sq = getattr(self, ...)`.
    assert "getattr(self, '_thinking_cap_override'" not in src, \
        "override must be read from the call kwarg, not the agent instance"

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

"""Tests for token cache display and evict summary improvements."""

import io
import os
from unittest.mock import MagicMock, patch

import pytest

from flagscale_agent.react import display


class TestLlmDoneWithCache:
    """Fix 1: llm_done displays cache breakdown when caching is active."""

    def _capture_llm_done(self, **kwargs):
        """Capture llm_done output."""
        buf = io.StringIO()
        os.environ["NO_COLOR"] = "1"
        try:
            with patch.object(display, '_print', side_effect=lambda *a, **kw: buf.write(str(a[0]) if a else "")):
                display.llm_done(**kwargs)
        finally:
            os.environ.pop("NO_COLOR", None)
        return buf.getvalue()

    def test_no_cache_no_extra_display(self):
        """Without cache tokens, display is unchanged."""
        out = self._capture_llm_done(elapsed=1.5, input_tokens=1000, output_tokens=50)
        assert "↑1,000" in out
        assert "↓50" in out
        assert "cache" not in out

    def test_cache_read_displayed(self):
        """Cache read tokens shown in parentheses."""
        out = self._capture_llm_done(
            elapsed=2.0, input_tokens=73, output_tokens=100,
            cache_read_tokens=114000
        )
        assert "↑73" in out
        assert "cache:114k" in out

    def test_cache_creation_displayed(self):
        """Cache creation tokens shown."""
        out = self._capture_llm_done(
            elapsed=1.0, input_tokens=500, output_tokens=200,
            cache_creation_tokens=2000
        )
        assert "↑500" in out
        assert "new:2,000" in out

    def test_both_cache_read_and_creation(self):
        """Both cache read and creation shown together."""
        out = self._capture_llm_done(
            elapsed=3.0, input_tokens=73, output_tokens=50,
            cache_read_tokens=100000, cache_creation_tokens=5000
        )
        assert "cache:100k" in out
        assert "new:5,000" in out
        # Format: ↑73(cache:100k+new:5,000)
        assert "+" in out

    def test_none_cache_values_ignored(self):
        """None values for cache don't trigger display."""
        out = self._capture_llm_done(
            elapsed=1.0, input_tokens=500, output_tokens=100,
            cache_read_tokens=None, cache_creation_tokens=None
        )
        assert "cache" not in out
        assert "new:" not in out

    def test_zero_cache_values_ignored(self):
        """Zero values treated same as None — no cache display."""
        out = self._capture_llm_done(
            elapsed=1.0, input_tokens=500, output_tokens=100,
            cache_read_tokens=0, cache_creation_tokens=0
        )
        # 0 is falsy, so should not trigger cache display
        assert "cache" not in out


class TestAnthropicProviderCacheUsage:
    """Fix 1: Anthropic provider yields cache tokens in usage event."""

    @pytest.fixture
    def provider(self):
        with patch("flagscale_agent.react.providers.anthropic_provider.anthropic") as mock_mod:
            mock_client = MagicMock()
            mock_mod.Anthropic.return_value = mock_client
            from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider(model="claude-test", api_key="test-key")
            p._mock_client = mock_client
            return p

    def _make_stream_mock(self, input_tokens=100, output_tokens=50,
                          cache_read=None, cache_create=None):
        """Create a mock stream context that yields no events but has final usage."""
        mock_usage = MagicMock()
        mock_usage.input_tokens = input_tokens
        mock_usage.output_tokens = output_tokens
        mock_usage.cache_read_input_tokens = cache_read
        mock_usage.cache_creation_input_tokens = cache_create

        mock_final = MagicMock()
        mock_final.usage = mock_usage

        mock_stream = MagicMock()
        mock_stream.__iter__ = MagicMock(return_value=iter([]))
        mock_stream.get_final_message.return_value = mock_final

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_stream)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        return mock_ctx

    def test_usage_includes_cache_read(self, provider):
        mock_ctx = self._make_stream_mock(
            input_tokens=73, output_tokens=50, cache_read=114000
        )
        provider._mock_client.messages.stream.return_value = mock_ctx

        events = list(provider.chat_stream(
            messages=[{"role": "user", "content": "hi"}], tools=[]
        ))
        usage_events = [e for e in events if e.get("type") == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0]["input_tokens"] == 73
        assert usage_events[0]["cache_read_input_tokens"] == 114000

    def test_usage_includes_cache_creation(self, provider):
        mock_ctx = self._make_stream_mock(
            input_tokens=500, output_tokens=100, cache_create=3000
        )
        provider._mock_client.messages.stream.return_value = mock_ctx

        events = list(provider.chat_stream(
            messages=[{"role": "user", "content": "hi"}], tools=[]
        ))
        usage_events = [e for e in events if e.get("type") == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0]["cache_creation_input_tokens"] == 3000
        assert "cache_read_input_tokens" not in usage_events[0]

    def test_no_cache_fields_when_absent(self, provider):
        mock_ctx = self._make_stream_mock(input_tokens=1000, output_tokens=200)
        provider._mock_client.messages.stream.return_value = mock_ctx

        events = list(provider.chat_stream(
            messages=[{"role": "user", "content": "hi"}], tools=[]
        ))
        usage_events = [e for e in events if e.get("type") == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0]["input_tokens"] == 1000
        assert "cache_read_input_tokens" not in usage_events[0]
        assert "cache_creation_input_tokens" not in usage_events[0]


class TestEvictSummaryShortMessageSkip:
    """Fix 2: Short/empty messages skip LLM summarization."""

    @pytest.fixture
    def agent_with_mock(self):
        """Create a minimal agent mock with _generate_evict_summaries accessible."""
        with patch("flagscale_agent.react.providers.anthropic_provider.anthropic") as mock_mod:
            mock_client = MagicMock()
            mock_mod.Anthropic.return_value = mock_client
            from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(model="claude-test", api_key="test-key")
            provider._mock_client = mock_client

            # Create a minimal agent-like object with the method
            class FakeAgent:
                pass
            agent = FakeAgent()
            agent.provider = provider
            agent._evict_summary = MagicMock()
            agent.history = MagicMock()

            # Bind the method
            from flagscale_agent.react.agent import WorkerAgent
            import types
            agent._generate_evict_summaries = types.MethodType(
                WorkerAgent._generate_evict_summaries, agent
            )
            return agent

    def test_short_message_no_llm_call(self, agent_with_mock):
        """Messages ≤150 chars should NOT trigger LLM calls."""
        agent = agent_with_mock
        messages = [
            (10, "user", None, "继续完善09"),
            (11, "assistant", "read_file", "Read a file."),
            (12, "user", None, ""),  # empty
        ]
        agent._generate_evict_summaries(messages)

        # No LLM call should have been made
        agent.provider._mock_client.messages.create.assert_not_called()

        # But summaries should still be stored
        assert agent._evict_summary.add.call_count == 3
        assert agent.history.update_evict_placeholder.call_count == 3

    def test_long_message_triggers_llm(self, agent_with_mock):
        """Messages >150 chars should trigger LLM summarization."""
        agent = agent_with_mock
        long_content = "x" * 200  # Over 150 char threshold

        # Mock LLM response
        agent.provider._mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(type="text", text="Summary of long content")]
        )

        messages = [(20, "user", None, long_content)]
        agent._generate_evict_summaries(messages)

        # LLM should have been called
        agent.provider._mock_client.messages.create.assert_called_once()

    def test_mixed_short_and_long(self, agent_with_mock):
        """Mix of short and long: only long ones trigger LLM."""
        agent = agent_with_mock
        messages = [
            (5, "user", None, "short msg"),           # short, no LLM
            (6, "assistant", "shell", "a" * 200),     # long, needs LLM
            (7, "user", None, "another short one"),   # short, no LLM
        ]

        agent.provider._mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(type="text", text="Summary")]
        )

        agent._generate_evict_summaries(messages)

        # Only 1 LLM call (for index 6)
        assert agent.provider._mock_client.messages.create.call_count == 1
        # All 3 summaries stored
        assert agent._evict_summary.add.call_count == 3

    def test_empty_content_gets_role_fallback(self, agent_with_mock):
        """Empty content uses role+tool_name as summary."""
        agent = agent_with_mock
        messages = [(99, "assistant", "write_file", "")]
        agent._generate_evict_summaries(messages)

        # Check the summary stored
        call_args = agent._evict_summary.add.call_args
        summary = call_args[0][2]  # 3rd positional arg is summary
        assert "assistant" in summary or "write_file" in summary


class TestEvictSummaryPromptQuality:
    """Fix 3: Summary prompt wording discourages hallucination."""

    def test_prompt_contains_factual_instruction(self):
        """Verify the summary prompt tells LLM to be factual, not interpretive."""
        import inspect
        from flagscale_agent.react.agent import WorkerAgent
        source = inspect.getsource(WorkerAgent._generate_evict_summaries)
        # New prompt should contain factual instruction
        assert "factually" in source
        assert "Do NOT infer intent" in source
        # Old problematic phrasing should be gone
        assert "Focus on WHAT was done/discussed" not in source

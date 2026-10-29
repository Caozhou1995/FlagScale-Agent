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

"""Structural channel enforcement: when the reasoning-escalation ladder has
exhausted what wording can do, the NEXT request must be sent with a
provider-level "must call a tool" constraint, so a prose-only reply is not a
legal completion at all.

Measured motivation (round 3, regex-chess): three token caps in one turn, and
after each one the agent wrote "I need to stop reasoning and act" and then
emitted another 2000+ characters of design prose. Wording is acknowledged and
ignored; only the channel constraint cannot be ignored, because the provider
itself rejects a text-only completion.
"""

import pytest

from flagscale_agent.react.providers.openai_provider import OpenAIProvider


# ── provider level: the wire payload actually carries the constraint ──


@pytest.fixture
def openai_provider(monkeypatch):
    monkeypatch.setattr(
        "flagscale_agent.react.providers.openai_provider.OpenAI",
        lambda **kwargs: object(),
    )
    return OpenAIProvider(model="m", api_key="k")


TOOLS = [{"type": "function", "function": {"name": "shell", "parameters": {}}}]


def test_openai_force_tool_sets_tool_choice_required(openai_provider):
    kwargs = openai_provider._build_kwargs([], TOOLS, force_tool=True)
    assert kwargs["tool_choice"] == "required"


def test_openai_without_force_tool_omits_tool_choice(openai_provider):
    kwargs = openai_provider._build_kwargs([], TOOLS)
    assert "tool_choice" not in kwargs


def test_openai_force_tool_without_tools_omits_tool_choice(openai_provider):
    """No tools means no legal tool call -- emitting tool_choice would make the
    request unsatisfiable, so it must be omitted rather than sent."""
    kwargs = openai_provider._build_kwargs([], None, force_tool=True)
    assert "tool_choice" not in kwargs


def test_openai_chat_stream_signature_accepts_force_tool(openai_provider):
    kwargs = openai_provider._build_kwargs([], TOOLS, stream=True, force_tool=True)
    assert kwargs["tool_choice"] == "required"
    assert kwargs["stream"] is True


def test_openai_chat_passes_force_tool_through(openai_provider, monkeypatch):
    captured = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)

            class _Msg:
                content = "ok"
                tool_calls = None

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]

            return _Resp()

    class _FakeChat:
        completions = _FakeCompletions()

    openai_provider._client = type("C", (), {"chat": _FakeChat()})()
    openai_provider.chat([], TOOLS, force_tool=True)
    assert captured["tool_choice"] == "required"

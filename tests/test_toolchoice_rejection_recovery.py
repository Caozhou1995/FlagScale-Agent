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

"""A gateway rejection of `tool_choice` must degrade the call, not kill the turn.

Measured 2026-11-26, write-compressor trial F6LwKJb (job 2026-11-26__18-11-11):
at 73% of a 900s budget the endgame window armed force_tool, the request went
out carrying tool_choice={"type":"any"}, and the gateway answered

    Error code: 400 - {'error': {'type': 'invalid_request_error',
        'message': 'Thinking mode does not support this tool_choice ...'}}

That 400 is not the reasoning-replay shape, so the provider re-raised it, the
agent's generic handler printed "LLM call failed" and broke out of the turn.
Turn 1 = 660.7s of work, ~4 minutes of budget still on the clock, reward 0.0.

The rejection is NOT deterministic: 44 probes of the identical shape (thinking
enabled at 2048/16384, tool_choice any, stream=true, temperature=0, history
carrying thinking blocks) all returned HTTP 200 against the same gateway. Only
some upstreams behind the balancer forbid the combination — the same per-request
coin flip already documented for the reasoning_content replay rejection. So
"probe first, then decide what to send" cannot work; the ONLY routing-independent
recovery is to degrade the rejected call and retry it, which is what these tests
pin.

Two readers, because a fix that passes one and fails the other is the classic
self-check: (1) the payload the gateway receives must lose tool_choice and keep
everything else; (2) the caller must receive a real completion, not an exception.
"""

import sys
import types

import pytest

import flagscale_agent.react.providers.anthropic_provider as anthropic_mod
from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider
from flagscale_agent.react.providers.openai_provider import OpenAIProvider


REJECTION = (
    "Error code: 400 - {'error': {'type': 'invalid_request_error', "
    "'message': 'Thinking mode does not support this tool_choice "
    "(request_id: 6a349b90-a496-4a98-815a-edd4ab4379fe)'}, 'type': 'error'}"
)

TOOLS = [{"name": "shell", "description": "run", "input_schema": {"type": "object"}}]


def _bad_request(msg=REJECTION):
    """Build the SDK exception class shape the provider catches."""
    exc_cls = getattr(anthropic_mod.anthropic, "BadRequestError", None)
    assert exc_cls is not None, "anthropic.BadRequestError must exist to catch it"
    try:
        return exc_cls(msg, response=_FakeResponse(), body=None)
    except TypeError:
        exc = exc_cls.__new__(exc_cls)
        Exception.__init__(exc, msg)
        return exc


class _FakeResponse:
    status_code = 400
    headers = {}
    request = None

    def json(self):
        return {"error": {"message": "Thinking mode does not support this tool_choice"}}


class _FakeUsage:
    input_tokens = 11
    output_tokens = 7


class _FakeBlock:
    def __init__(self, btype, text=None, name=None, input=None, tid=None):
        self.type = btype
        self.text = text
        self.name = name
        self.input = input
        self.id = tid


class _FakeMessage:
    def __init__(self, blocks):
        self.content = blocks
        self.usage = _FakeUsage()


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(anthropic_mod.anthropic, "Anthropic", lambda **kw: object())
    p = AnthropicProvider(model="m", api_key="k", base_url="https://gw.example",
                          thinking_budget=16384)
    # The degrade path sleeps between attempts; the test measures bytes, not
    # wall clock, so keep it instant.
    p._retry_backoff = lambda i: None
    return p


def _capture_create(provider, replies):
    """Install a fake `messages.create` that yields `replies` in order.

    A reply that is an exception instance is raised (the gateway rejecting);
    anything else is returned (the gateway accepting).
    """
    sent = []
    queue = list(replies)

    class _Messages:
        def create(self, **kwargs):
            sent.append(dict(kwargs))
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

    provider._client = types.SimpleNamespace(messages=_Messages())
    return sent


def _reminder(capsys):
    return capsys.readouterr().err


# ── reader 1: the payload that finally reaches the gateway ──


def test_chat_drops_tool_choice_and_retries_without_it(provider, capsys):
    good = _FakeMessage([_FakeBlock("tool_use", name="shell", input={}, tid="t1")])
    sent = _capture_create(provider, [_bad_request(), good])

    out = provider.chat([{"role": "user", "content": "hi"}], TOOLS, force_tool=True)

    assert len(sent) == 2, f"expected one rejected call + one retry, got {len(sent)}"
    assert sent[0]["tool_choice"] == {"type": "any"}, "first call must be untouched"
    assert sent[0].get("thinking", {}).get("type") != "disabled", \
        "first call must not pre-emptively disable thinking"
    # Stage 1: channel forcing DROPPED, reasoning KEPT — measured 2026-11-27 on
    # api.deepseek.com/anthropic that the endpoint refuses tool_choice while
    # thinking is ON, and accepts thinking:enabled + no tool_choice (200,
    # blocks=['thinking','tool_use']). Reasoning is the lever that decides what
    # gets written, so it is the one that must survive first.
    assert sent[1].get("thinking", {}).get("type") != "disabled", \
        "stage 1 must keep the reasoning channel"
    assert "tool_choice" not in sent[1], \
        "stage 1 drops the forcing -- that is the lever the gateway objects to"
    assert sent[1]["tools"] == TOOLS, "tools themselves are legal and must survive"
    assert sent[1]["model"] == sent[0]["model"]
    assert sent[1]["max_tokens"] == sent[0]["max_tokens"]
    # reader 2: the caller gets a completion, not an exception
    assert out["tool_calls"][0]["name"] == "shell"
    # the degradation must be visible in the transcript, not silent
    assert "tool_choice" in _reminder(capsys)


def test_chat_stream_degrades_on_rejected_open(provider, capsys):
    """The streaming open path is the one this trial actually died in."""
    import anthropic as real_anthropic

    accepted = _FakeMessage([_FakeBlock("text", text="ok")])
    sent = []
    attempts = {"n": 0}

    class _StreamCtx:
        def __enter__(self):
            events = [types.SimpleNamespace(
                type="content_block_delta",
                delta=types.SimpleNamespace(type="text_delta", text="ok"))]
            return iter(events)

        def __exit__(self, *exc):
            return False

    class _Messages:
        def stream(self, **kwargs):
            sent.append(dict(kwargs))
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _bad_request()
            ctx = _StreamCtx()
            ctx.get_final_message = lambda: accepted
            return ctx

    provider._client = types.SimpleNamespace(messages=_Messages())

    events = list(provider.chat_stream(
        [{"role": "user", "content": "hi"}], TOOLS, force_tool=True))

    assert len(sent) == 2, f"expected rejected open + retry, got {len(sent)}"
    assert sent[0]["tool_choice"] == {"type": "any"}
    assert sent[1].get("thinking", {}).get("type") != "disabled", \
        "stage 1 keeps the reasoning channel"
    assert "tool_choice" not in sent[1], "stage 1 drops the forcing"
    assert any(e.get("type") == "done" for e in events), "stream must complete"
    assert "tool_choice" in _reminder(capsys)


def test_retry_still_failing_raises_the_rejection(provider):
    """If even the degraded shape is refused, tool_choice was not the objection
    -- that must surface, not be swallowed into an empty completion."""
    sent = _capture_create(provider, [_bad_request() for _ in range(8)])
    with pytest.raises(Exception) as ei:
        provider.chat([{"role": "user", "content": "hi"}], TOOLS, force_tool=True)
    assert "tool_choice" in str(ei.value).lower()
    # escalation: stage 1 drops only the forcing and keeps reasoning; stage 2
    # restores the caller's forcing and gives up reasoning instead. One lever
    # has to go on an endpoint that refuses the pair.
    assert sent[1].get("thinking", {}).get("type") != "disabled"
    assert "tool_choice" not in sent[1]
    assert sent[-1]["thinking"] == {"type": "disabled"}
    assert sent[-1]["tool_choice"] == {"type": "any"}, \
        "the terminal shape keeps the caller's channel constraint, drops reasoning"
    assert len(sent) > 1


def test_stage1_without_tool_use_restores_the_forcing(provider, capsys):
    """Dropping the forcing is only safe if the tool call still arrives.

    Measured on api.deepseek.com/anthropic: with a tool schema present the
    unforced model still emits `tool_use`. But that must not be assumed — a
    stage-1 reply carrying only prose would silently turn a force_tool call
    into text. This pins the guard: no `tool_use` on the unforced attempt must
    escalate to the shape that restores forcing, and the caller still gets its
    tool call.
    """
    prose = _FakeMessage([_FakeBlock("text", text="I would rather explain.")])
    good = _FakeMessage([_FakeBlock("tool_use", name="shell", input={}, tid="t9")])
    sent = _capture_create(provider, [_bad_request(), prose, good])

    out = provider.chat([{"role": "user", "content": "hi"}], TOOLS, force_tool=True)

    assert len(sent) == 3, f"rejected + unforced(no tool_use) + forced, got {len(sent)}"
    # attempt 1: reasoning kept, forcing dropped
    assert "tool_choice" not in sent[1]
    assert sent[1].get("thinking", {}).get("type") != "disabled"
    # attempt 2: the guard escalates to stage 2 — forcing AND reasoning released
    assert sent[2]["tool_choice"] == {"type": "any"}, \
        "the retry must restore the forcing the caller asked for"
    assert sent[2]["thinking"] == {"type": "disabled"}
    # reader 2: the caller still gets the tool call it could not proceed without
    assert out["tool_calls"][0]["name"] == "shell"


def test_stage1_with_tool_use_is_accepted_without_escalation(provider):
    """The common case must stay cheap: one degrade, reasoning preserved."""
    good = _FakeMessage([_FakeBlock("tool_use", name="shell", input={}, tid="t1")])
    sent = _capture_create(provider, [_bad_request(), good])

    out = provider.chat([{"role": "user", "content": "hi"}], TOOLS, force_tool=True)

    assert len(sent) == 2, "must not escalate when the unforced call did its job"
    assert "tool_choice" not in sent[1]
    assert sent[1].get("thinking", {}).get("type") != "disabled", \
        "reasoning survives on the accepted path"
    assert out["tool_calls"][0]["name"] == "shell"


# ── discrimination: the recovery must be narrow ──


def test_reasoning_replay_rejection_is_not_this_error(provider):
    """A different 400 must keep going down its own recovery road."""
    exc = _bad_request("The `reasoning_content` in the thinking mode must be "
                       "passed back to the API")
    assert not provider._is_toolchoice_thinking_error(exc)


def test_unrelated_tool_use_pairing_400_is_not_this_error(provider):
    exc = _bad_request("messages.1: `tool_use` ids were found without "
                       "`tool_result` blocks immediately after")
    assert not provider._is_toolchoice_thinking_error(exc)


def test_degrade_stages_are_ordered_least_destructive_first(provider):
    """Stage 1 must keep the REASONING; only a later stage disables it.

    Ordering rationale, measured 2026-11-27 on api.deepseek.com/anthropic:
    the endpoint refuses tool_choice while thinking is ON, and accepts
    thinking:enabled + no tool_choice (200, blocks=['thinking','tool_use']).
    The reversible lever is therefore the FORCING, not the reasoning — the
    reasoning is what decides which bytes get written.
    """
    original = {"model": "m", "max_tokens": 9, "tools": TOOLS,
                "tool_choice": {"type": "any"}, "stream": True}
    s1 = provider._degrade_for_toolchoice(original, 1)
    assert "tool_choice" not in s1, "stage 1 drops the lever the gateway objects to"
    assert s1.get("thinking", {}).get("type") != "disabled", \
        "stage 1 must not disable reasoning"
    assert original.get("thinking") is None, "must not mutate caller's dict"

    s2 = provider._degrade_for_toolchoice(original, 2)
    assert s2["tool_choice"] == {"type": "any"}, \
        "stage 2 restores the forcing the caller asked for"
    assert s2["thinking"] == {"type": "disabled"}
    assert {k: v for k, v in s2.items() if k != "thinking"} == {
        "model": "m", "max_tokens": 9, "tools": TOOLS,
        "tool_choice": {"type": "any"}, "stream": True}, \
        "stage 2 changes only the thinking field"
    assert original["tool_choice"] == {"type": "any"}, "must not mutate caller's dict"


# ── the constraint still fires on a gateway that accepts it ──


def test_accepted_request_still_carries_tool_choice(provider):
    """Guard against 'fixing' the 400 by dropping enforcement everywhere."""
    good = _FakeMessage([_FakeBlock("tool_use", name="shell", input={}, tid="t1")])
    sent = _capture_create(provider, [good])
    provider.chat([{"role": "user", "content": "hi"}], TOOLS, force_tool=True)
    assert len(sent) == 1, "a healthy gateway must see exactly one call"
    assert sent[0]["tool_choice"] == {"type": "any"}


def test_unforced_request_never_carries_tool_choice(provider):
    good = _FakeMessage([_FakeBlock("text", text="hi")])
    sent = _capture_create(provider, [good])
    provider.chat([{"role": "user", "content": "hi"}], TOOLS)
    assert "tool_choice" not in sent[0]


# ── the openai provider carries the same degradation ──


def test_openai_chat_stream_degrades_on_tool_choice_rejection(monkeypatch):
    monkeypatch.setattr(
        "flagscale_agent.react.providers.openai_provider.OpenAI",
        lambda **kwargs: object())
    p = OpenAIProvider(model="m", api_key="k")
    sent = []

    class _Completions:
        def create(self, **kwargs):
            sent.append(dict(kwargs))
            if len(sent) == 1:
                raise Exception("Thinking mode does not support this tool_choice")
            return iter([])  # empty chunk stream is fine for this assertion

    p._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Completions()))

    list(p.chat_stream([{"role": "user", "content": "hi"}], TOOLS, force_tool=True))

    assert sent[0]["tool_choice"] == "required"
    assert "tool_choice" not in sent[1]


def test_openai_string_tool_choice_500_is_also_degraded(monkeypatch):
    """Measured on tokenhub: the STRING form is refused with HTTP 500
    'invalid Claude tool_choice: cannot unmarshal string ...'. Same degradation."""
    monkeypatch.setattr(
        "flagscale_agent.react.providers.openai_provider.OpenAI",
        lambda **kwargs: object())
    p = OpenAIProvider(model="m", api_key="k")
    sent = []

    class _Completions:
        def create(self, **kwargs):
            sent.append(dict(kwargs))
            if len(sent) == 1:
                raise Exception('invalid Claude tool_choice: json: cannot unmarshal '
                                'string into Go value of type map[string]interface {}')
            return iter([])

    p._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Completions()))
    list(p.chat_stream([{"role": "user", "content": "hi"}], TOOLS, force_tool=True))
    assert "tool_choice" not in sent[1]

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

"""An UNATTRIBUTABLE 400 must not end the turn.

Measured 2026-11-26, twice in one batch, on payloads the same gateway accepts
moments later (22 probes over max_tokens 1k..200k, cache_control on system and
messages, synthesized empty thinking blocks, text-only assistant turns, inline
temperature: all HTTP 200):

    ⚠ LLM call failed: Error code: 400 - {'error': {'type': '400001',
      'message': 'The request is invalid: the request was rejected by an
      internal MaaS component. Please check the request body, required fields,
      and request format. ...'}}

    ⚠ LLM call failed: Error code: 400 - {'error': {'type':
      'invalid_request_error', 'message': 'Thinking mode does not support this
      tool_choice ...'}}

Both bodies name either no field or a field the request may legally carry, so
neither is a verdict on our bytes: they are properties of whichever upstream the
balancer picked. Treating them as permanent is what printed "LLM call failed"
and broke the turn — so the provider must reshape and resend instead.

The test set is deliberately two-sided: the recovery must fire on an opaque 400
AND must stay out of the way of the 400s that genuinely are our fault (context
limit, broken tool_use/tool_result pairing), because resending those only burns
the wall clock.
"""

import types

import pytest

import flagscale_agent.react.providers.anthropic_provider as anthropic_mod
from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider


MAAS_BODY = (
    "Error code: 400 - {'error': {'type': '400001', 'message': 'The request is "
    "invalid: the request was rejected by an internal MaaS component. Please "
    "check the request body, required fields, and request format. "
    "(request id: 202611261200463860374218268d9d6sXk4nxHG)'}, 'type': 'error'}"
)

TOOLS = [{"name": "shell", "description": "run", "input_schema": {"type": "object"}}]


class _FakeResponse:
    status_code = 400
    headers = {}
    request = None

    def json(self):
        return {"error": {"message": "invalid"}}


def _bad_request(msg):
    exc_cls = anthropic_mod.anthropic.BadRequestError
    try:
        return exc_cls(msg, response=_FakeResponse(), body=None)
    except TypeError:
        exc = exc_cls.__new__(exc_cls)
        Exception.__init__(exc, msg)
        return exc


class _Usage:
    input_tokens = 3
    output_tokens = 2


class _Block:
    def __init__(self, t, text=None, name=None, input=None, tid=None):
        self.type, self.text, self.name = t, text, name
        self.input, self.id = input, tid


class _Message:
    def __init__(self, blocks):
        self.content = blocks
        self.usage = _Usage()


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(anthropic_mod.anthropic, "Anthropic", lambda **kw: object())
    p = AnthropicProvider(model="m", api_key="k", base_url="https://gw.example",
                          thinking_budget=16384, temperature=0.0)
    p._retry_backoff = lambda i: None
    return p


def _capture(provider, replies):
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


# ── the recovery fires ──


def test_opaque_maas_400_is_retried_not_fatal(provider, capsys):
    good = _Message([_Block("tool_use", name="shell", input={}, tid="t1")])
    sent = _capture(provider, [_bad_request(MAAS_BODY), good])

    out = provider.chat([{"role": "user", "content": "hi"}], TOOLS, force_tool=True)

    assert len(sent) == 2, f"opaque 400 must be resent, got {len(sent)} call(s)"
    assert out["tool_calls"][0]["name"] == "shell", "caller must get a completion"
    assert "llm-degrade" in capsys.readouterr().err, "degradation must be traceable"


def test_retry_drops_decoration_but_keeps_the_conversation(provider):
    """Stage 1 strips cache_control; the messages and tools must survive."""
    good = _Message([_Block("text", text="ok")])
    sent = _capture(provider, [_bad_request(MAAS_BODY), good])
    provider.chat([{"role": "user", "content": "hi"}], TOOLS, force_tool=True)

    def has_cache_control(kw):
        s = kw.get("system")
        if isinstance(s, list) and any("cache_control" in b for b in s
                                       if isinstance(b, dict)):
            return True
        for m in kw.get("messages") or []:
            c = m.get("content")
            if isinstance(c, list) and any("cache_control" in b for b in c
                                           if isinstance(b, dict)):
                return True
        return False

    assert has_cache_control(sent[0]), "first call is untouched"
    assert not has_cache_control(sent[1]), "stage 1 removes cache_control"
    assert sent[1]["tools"] == TOOLS
    assert sent[1]["messages"], "the conversation must never be dropped"
    assert sent[1]["tool_choice"] == {"type": "any"}, \
        "stage 1 must not cost the channel constraint"


def test_generic_recovery_escalates_and_still_reports_failure(provider):
    """If every shape is refused, the LAST rejection must surface."""
    sent = _capture(provider, [_bad_request(MAAS_BODY) for _ in range(8)])
    with pytest.raises(Exception) as ei:
        provider.chat([{"role": "user", "content": "hi"}], TOOLS, force_tool=True)
    assert "MaaS" in str(ei.value) or "400" in str(ei.value)
    assert len(sent) == 1 + provider._MAX_GENERIC_400_RETRIES, \
        f"one original call + {provider._MAX_GENERIC_400_RETRIES} reshaped retries, got {len(sent)}"
    # The last retry is STAGE 2: it RESTORES the caller's forcing and disables
    # thinking (see _degrade_for_generic_400's docstring: on an endpoint that
    # refuses the pair, the channel constraint is the lever that must survive).
    # Asserting `tool_choice not in sent[-1]` encoded the superseded ordering,
    # where the last body carried neither lever — that is the shape measured to
    # write artifacts on a reasoning-disabled call.
    assert sent[-1]["tool_choice"] == {"type": "any"}, \
        "the last escalation must restore the caller's channel constraint"
    assert sent[-1].get("thinking") == {"type": "disabled"}, \
        "the last escalation must give up reasoning, not the constraint"
    last_msgs = sent[-1]["messages"]
    for m in last_msgs:
        c = m.get("content")
        if isinstance(c, list):
            assert not any("cache_control" in b for b in c if isinstance(b, dict))


def test_stream_open_on_opaque_400_recovers(provider, capsys):
    """The streaming open is where the reported failures actually died."""
    import io

    sent = []
    state = {"n": 0}

    class _Ctx:
        def __enter__(self):
            return iter([types.SimpleNamespace(
                type="content_block_delta",
                delta=types.SimpleNamespace(type="text_delta", text="ok"))])

        def __exit__(self, *a):
            return False

    class _Messages:
        def stream(self, **kwargs):
            sent.append(dict(kwargs))
            state["n"] += 1
            if state["n"] == 1:
                raise _bad_request(MAAS_BODY)
            ctx = _Ctx()
            ctx.get_final_message = lambda: _Message([_Block("text", text="ok")])
            return ctx

    provider._client = types.SimpleNamespace(messages=_Messages())
    events = list(provider.chat_stream(
        [{"role": "user", "content": "hi"}], TOOLS, force_tool=True))

    assert len(sent) == 2
    assert any(e.get("type") == "done" for e in events), "stream must complete"
    assert "llm-degrade" in capsys.readouterr().err


# ── the recovery stays OUT of the way ──
#
# These are the 400s that ARE our fault: resending cannot repair them, so
# retrying would only burn the wall clock the agent needs to land work.


def test_context_limit_400_is_not_retried(provider):
    sent = _capture(provider, [_bad_request(
        "Error code: 400 - the prompt is too long: maximum context length "
        "exceeded")])
    with pytest.raises(Exception):
        provider.chat([{"role": "user", "content": "hi"}], TOOLS, force_tool=True)
    assert len(sent) == 1, "a context-limit error needs eviction, not a resend"


def test_broken_tool_pairing_400_is_not_retried(provider):
    sent = _capture(provider, [_bad_request(
        "Error code: 400 - messages.1: `tool_use` ids were found without "
        "`tool_result` blocks immediately after")])
    with pytest.raises(Exception):
        provider.chat([{"role": "user", "content": "hi"}], TOOLS, force_tool=True)
    assert len(sent) == 1, "a malformed history cannot be fixed by resending"


def test_non_400_errors_are_untouched(provider):
    sent = _capture(provider, [_bad_request(MAAS_BODY)])
    exc = RuntimeError("boom")
    assert not provider._is_generic_retryable_400(exc)
    assert len(sent) == 0


def test_a_healthy_call_still_sends_one_request(provider):
    good = _Message([_Block("text", text="ok")])
    sent = _capture(provider, [good])
    provider.chat([{"role": "user", "content": "hi"}], TOOLS, force_tool=True)
    assert len(sent) == 1
    assert sent[0]["tool_choice"] == {"type": "any"}


# ── the same generality for the OpenAI-compatible provider ──
#
# This SDK has no BadRequestError subclass in the provider's catch, so the
# classification is by status + text. Both directions are pinned here too: the
# opaque body must be retried, a context-limit must not.


def _openai_provider(monkeypatch):
    monkeypatch.setattr(
        "flagscale_agent.react.providers.openai_provider.OpenAI",
        lambda **kw: object())
    from flagscale_agent.react.providers.openai_provider import OpenAIProvider
    return OpenAIProvider(model="m", api_key="k")


class _StatusError(Exception):
    """Shaped like an openai APIStatusError: carries .status_code."""

    def __init__(self, msg, status_code=400):
        super().__init__(msg)
        self.status_code = status_code


def test_openai_opaque_400_is_retried(monkeypatch):
    p = _openai_provider(monkeypatch)
    sent = []

    class _Completions:
        def create(self, **kwargs):
            sent.append(dict(kwargs))
            if len(sent) == 1:
                raise _StatusError(MAAS_BODY)
            return iter([])

    p._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Completions()))
    list(p.chat_stream([{"role": "user", "content": "hi"}], TOOLS, force_tool=True))
    assert len(sent) == 2, "opaque 400 on stream open must be reshaped, not raised"
    assert sent[1]["messages"], "conversation must survive the degrade"


def test_openai_context_limit_is_not_retried(monkeypatch):
    p = _openai_provider(monkeypatch)
    sent = []

    class _Completions:
        def create(self, **kwargs):
            sent.append(dict(kwargs))
            raise _StatusError("Error code: 400 - maximum context length exceeded")

    p._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Completions()))
    with pytest.raises(Exception):
        list(p.chat_stream([{"role": "user", "content": "hi"}], TOOLS, force_tool=True))
    assert len(sent) == 1


def test_openai_non_400_is_not_treated_as_generic(monkeypatch):
    p = _openai_provider(monkeypatch)
    assert not p._is_generic_retryable_rejection(_StatusError("boom", status_code=403))

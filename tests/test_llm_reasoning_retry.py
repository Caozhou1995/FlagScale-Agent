"""Fault-injection tests for the reasoning-replay retry ladder.

Background (measured 2026-11-26 on a tokenhub anthropic-compatible gateway):
the same request bytes returned HTTP 400 "The `reasoning_content` in the
thinking mode must be passed back to the API" on some requests and 200 on
others, so a rejection is a per-request coin flip rather than a verdict on our
payload. These tests pin the recovery properties that follow from that:

  * a run of consecutive rejections is survivable, not fatal;
  * recovery does not require the caller to change anything;
  * a BadRequestError that is NOT this specific rejection must never be
    swallowed by the ladder (masking real errors would be worse than the 400).

Only `_create_with_reasoning_retry` is exercised: no network, no client.
"""
import sys
import types

import pytest

sys.path.insert(0, "flagscale_agent")

from react.providers.anthropic_provider import AnthropicProvider  # noqa: E402


class _StubMessages:
    """Rejects the first `reject_first` calls, then returns a sentinel."""

    def __init__(self, reject_first, rejection=None):
        self.calls = 0
        self.reject_first = reject_first
        self.rejection = rejection or _replay_error()
        self.payloads = []

    def create(self, **kwargs):
        self.calls += 1
        self.payloads.append(kwargs)
        if self.calls <= self.reject_first:
            raise self.rejection
        return ("ACCEPTED", self.calls)


def _replay_error():
    return _make(
        "Error code: 400 - {'error': {'type': 'invalid_request_error', "
        "'message': 'The `reasoning_content` in the thinking mode must be "
        "passed back to the API.'}}")


import httpx


def _make(text):
    """Build the real SDK exception so the provider's except-clause matches."""
    import anthropic

    request = httpx.Request("POST", "https://gateway.invalid/v1/messages")
    response = httpx.Response(400, request=request, json={"error": {"message": text}})
    return anthropic.BadRequestError(message=text, response=response, body=None)


def _provider():
    prov = AnthropicProvider.__new__(AnthropicProvider)
    prov.model = "stub"
    prov.max_tokens = 64
    prov.thinking_budget = 0
    prov._is_third_party = True
    # the ladder must not sleep in tests
    prov._retry_backoff = staticmethod(lambda i: None)
    prov._client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=None))
    return prov


def _kwargs():
    return {
        "model": "stub",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "placeholder"},
        ],
    }


def test_is_reasoning_replay_error_matches_only_the_gateway_rejection():
    prov = _provider()
    assert prov._is_reasoning_replay_error(_replay_error()) is True
    other = _make("Error code: 400 - invalid_request_error: max_tokens too large")
    assert prov._is_reasoning_replay_error(other) is False


@pytest.mark.parametrize("reject_first", [1, 2, 5, 12, 40])
def test_ladder_recovers_from_a_run_of_rejections(reject_first):
    """A burst of identical-verdict rejections must not kill the turn.

    `reject_first=40` is the case the user hit in practice (five consecutive
    user-visible failures under the old four-attempt ladder).
    """
    prov = _provider()
    stub = _StubMessages(reject_first=reject_first)
    prov._client.messages.create = stub.create

    result = prov._create_with_reasoning_retry(_kwargs(), attempts=50)

    assert result == ("ACCEPTED", reject_first + 1)
    assert stub.calls == reject_first + 1


def test_ladder_gives_up_after_the_configured_attempts():
    prov = _provider()
    stub = _StubMessages(reject_first=10_000)
    prov._client.messages.create = stub.create

    with pytest.raises(Exception) as excinfo:
        prov._create_with_reasoning_retry(_kwargs(), attempts=7)

    assert "reasoning_content" in str(excinfo.value)
    assert stub.calls == 7


def test_first_attempt_is_byte_identical_and_later_attempts_alternate_shape():
    """Attempt 0 must not alter the caller's payload, and retries must
    alternate the two shapes so no single upstream's preference can reject
    every attempt.

    Contract note (measured, not assumed): the ladder guarantees
    odd retries = every reasoning block stripped, even retries = the caller's
    payload untouched. Producing a payload that CARRIES synthesized thinking is
    not this ladder's job -- `_build_kwargs` -> `_passthrough_thinking` does it
    upstream, so on a healthy path the untouched payload already carries a
    thinking block and the two attempts differ exactly as intended.
    """
    prov = _provider()
    stub = _StubMessages(reject_first=3)
    prov._client.messages.create = stub.create

    prov._create_with_reasoning_retry(_kwargs(), attempts=50)

    # attempt 0: caller's bytes, untouched (healthy gateway unchanged)
    assert stub.payloads[0]["messages"][1]["content"] == "placeholder"
    # attempt 1 (odd): stripped -> block list, never a thinking block
    shape1 = stub.payloads[1]["messages"][1]["content"]
    assert isinstance(shape1, list)
    assert all(b.get("type") != "thinking" for b in shape1)
    # attempt 2 (even): the caller's payload again, unmodified
    assert stub.payloads[2]["messages"][1]["content"] == "placeholder"
    # so the two retry shapes genuinely differ -> one bad upstream cannot
    # reject the whole ladder
    assert shape1 != stub.payloads[2]["messages"][1]["content"]


def test_normalised_shape_carries_thinking_when_build_kwargs_supplied_it():
    """The other half of the alternation: if the payload handed to the ladder
    already carries a thinking block (the normal production shape), the even
    attempt keeps it, so the ladder really alternates present/absent."""
    prov = _provider()
    stub = _StubMessages(reject_first=3)
    prov._client.messages.create = stub.create

    kwargs = _kwargs()
    kwargs["messages"] = [
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
            {"type": "text", "text": "the answer"},
        ]},
    ]

    prov._create_with_reasoning_retry(kwargs, attempts=50)

    odd = stub.payloads[1]["messages"][1]["content"]
    even = stub.payloads[2]["messages"][1]["content"]
    assert all(b.get("type") != "thinking" for b in odd), "odd retry must strip"
    assert any(b.get("type") == "thinking" for b in even), "even retry must keep"


def test_non_replay_bad_request_is_not_swallowed():
    """A different 400 must surface immediately: the ladder recovers from this
    one gateway quirk only, it is not a general error muffler."""
    prov = _provider()
    stub = _StubMessages(
        reject_first=10_000,
        rejection=_make("Error code: 400 - invalid_request_error: bad tool schema"),
    )
    prov._client.messages.create = stub.create

    with pytest.raises(Exception) as excinfo:
        prov._create_with_reasoning_retry(_kwargs(), attempts=50)

    assert "bad tool schema" in str(excinfo.value)
    assert stub.calls == 1, "a non-replay 400 must not be retried at all"

"""Thinking blocks must go on the wire ONLY when the endpoint signed them.

MEASURED 2026-11-28 on flagos (the backend the write-compressor trial ran on):
the signature arrives as ONE trailing ``signature_delta``, after every
``thinking_delta``. So any early exit from the event loop — notably the runtime
thinking cap in agent.py, which ``break``s out — keeps the real thinking TEXT
and loses the SIGNATURE.

Two defects followed from that, and both are pinned here:

  BUG A (write side of the response): ``format_assistant_message`` persisted
  ``{"type":"thinking","thinking":<47791 real chars>,"signature":""}``. That
  trial's conversation.json carries five such blocks (47791 / 63619 / 307 / 178
  / 6154 chars) and every one has signature == "". A real-text/empty-signature
  block asserts a provenance it does not have.

  BUG B (replay side): ``_passthrough_thinking`` DELETED every real reasoning
  block and substituted an empty one, discarding the endpoint's own signed
  blocks. The vendor doc for thinking-mode tool continuation asks for the full
  assistant content array to be replayed as-is.

Both are checked with the CONSUMER's own comparison where one exists: the
replayed block list is asserted field by field, not by stringification.
"""

from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider


def _provider(base_url="https://api.flagos.net"):
    return AnthropicProvider(model="DeepSeek-V4.1-Flash", api_key="x",
                             base_url=base_url)


# ── BUG A: format_assistant_message ────────────────────────────────────────

def test_capped_thinking_without_signature_is_not_persisted_as_a_block():
    """The cap-truncated response: real text, no signature -> NO thinking block.

    This is the exact object the live trial produced. Persisting it as a
    thinking block shipped an unsigned claim of endpoint provenance.
    """
    resp = {"thinking": "x" * 47791, "signature": None,
            "content": None, "tool_calls": [{"id": "t1", "name": "n", "arguments": {}}]}
    msg = _provider().format_assistant_message(resp)

    types = [b["type"] for b in msg["content"]]
    assert "thinking" not in types, (
        "an unsigned thinking block must never be persisted; the text is "
        "re-delivered by the kernel's <previous_reasoning> channel instead")
    assert types == ["tool_use"], "the rest of the turn must be untouched"


def test_signed_thinking_round_trips_verbatim():
    """Positive bound: a block the endpoint signed survives unchanged."""
    resp = {"thinking": "real reasoning", "signature": "a642a99f4afa4f338c92e7ec8e2ee8ae",
            "content": "the answer", "tool_calls": None}
    msg = _provider().format_assistant_message(resp)

    assert msg["content"][0] == {
        "type": "thinking",
        "thinking": "real reasoning",
        "signature": "a642a99f4afa4f338c92e7ec8e2ee8ae",
    }, "a signed block must be carried verbatim, signature included"
    assert msg["content"][1] == {"type": "text", "text": "the answer"}


def test_empty_signature_string_is_treated_as_unsigned():
    """'' and None are the same fact: no signature.

    Ablation guard: a check written as `"signature" in response` would pass the
    live trial's bytes (the key exists, holding "") and read green while
    shipping the defect.
    """
    for sig in ("", None):
        msg = _provider().format_assistant_message(
            {"thinking": "t", "signature": sig, "content": "c", "tool_calls": None})
        assert [b["type"] for b in msg["content"]] == ["text"], sig


def test_thinking_absent_entirely_leaves_turn_unchanged():
    """A turn with no thinking at all keeps its old shape exactly."""
    msg = _provider().format_assistant_message(
        {"content": "c", "tool_calls": None})
    assert msg["content"] == [{"type": "text", "text": "c"}]


# ── BUG B: _passthrough_thinking ──────────────────────────────────────────

def _replay(provider, content):
    return provider._passthrough_thinking(
        [{"role": "user", "content": "q"},
         {"role": "assistant", "content": content}])


def test_replay_keeps_a_signed_block_instead_of_replacing_it():
    """The endpoint's own signed block must survive replay verbatim.

    Ablation: the pre-fix implementation returned {"thinking":"","signature":""}
    here and the two assertions below go red.
    """
    signed = {"type": "thinking", "thinking": "why I chose LZ77",
              "signature": "970ff9bfd8284d3c9e4531eaebfdf3a0"}
    out = _replay(_provider(), [signed, {"type": "tool_use", "id": "t", "name": "n", "input": {}}])

    blocks = out[1]["content"]
    kept = [b for b in blocks if b.get("type") == "thinking"]
    assert len(kept) == 1, "exactly one reasoning slot"
    assert kept[0]["thinking"] == "why I chose LZ77", "real text must survive"
    assert kept[0]["signature"] == "970ff9bfd8284d3c9e4531eaebfdf3a0", \
        "the signature is the proof of provenance; dropping it is the bug"


def test_replay_drops_an_unsigned_block_and_falls_back_to_empty():
    """Unsigned real text is not replayable; the gateway still gets a slot."""
    out = _replay(_provider(), [
        {"type": "thinking", "thinking": "x" * 47791, "signature": ""},
        {"type": "text", "text": "answer"},
    ])
    blocks = out[1]["content"]
    thinking = [b for b in blocks if b.get("type") == "thinking"]
    assert thinking == [{"type": "thinking", "thinking": "", "signature": ""}], \
        "unsigned text is dropped, one empty slot satisfies the gateway"
    assert {"type": "text", "text": "answer"} in blocks, "real answer untouched"


def test_replay_synthesizes_a_slot_when_the_turn_has_none():
    """A text-only (e.g. evicted/placeholder) turn still gains one slot."""
    out = _replay(_provider(), "evicted assistant turn placeholder")
    blocks = out[1]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "", "signature": ""}
    assert blocks[1] == {"type": "text", "text": "evicted assistant turn placeholder"}


def test_official_endpoint_is_untouched():
    """Byte-identical for api.anthropic.com: the whole routine must no-op.

    Guards against 'fix the third-party path' silently rewriting official
    request bytes, which neither needs nor tolerates a synthesized block.
    """
    msgs = [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "t", "signature": ""}]}]
    out = _provider("https://api.anthropic.com")._passthrough_thinking(msgs)
    assert out is msgs, "official path must return the caller's object unchanged"


# ── the two bugs together: the live trial's actual bytes ──────────────────

def test_live_trial_shape_never_reaches_the_wire():
    """End-to-end over the shape the trial produced.

    A capped turn (real text, no signature) is persisted WITHOUT a thinking
    block, so replay has nothing unsigned to carry — and the request that goes
    out contains no real-text/empty-signature block anywhere.
    """
    prov = _provider()
    capped = prov.format_assistant_message(
        {"thinking": "y" * 6154, "signature": None,
         "content": None, "tool_calls": [{"id": "t", "name": "n", "arguments": {}}]})
    out = prov._passthrough_thinking([capped])

    for msg in out:
        for b in msg["content"]:
            if isinstance(b, dict) and b.get("type") == "thinking":
                assert not (b.get("thinking") and not (b.get("signature") or "").strip()), \
                    "real thinking text must never ride on an empty signature"

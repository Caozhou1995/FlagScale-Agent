"""Thinking blocks must never be ERASED by the harness, and never FABRICATED.

This file replaces the earlier contract ("a thinking block goes on the wire only
when the endpoint signed it"). That rule was measured wrong on BOTH of its
premises on 2026-11-28:

  1. "Dropping it loses nothing — the kernel re-injects the reasoning via
     <previous_reasoning>."  False. That re-injection exists ONLY on the runtime
     cap branch (kernel.py:1041/1083/1099/1739) and carries a TRUNCATED tail,
     not the full stream. On the ordinary path nothing re-injects it.
  2. "A real-text/empty-signature block risks a rejection."  False, measured.
     Replaying {"type":"thinking","thinking":<real text>,"signature":""} returns
     HTTP 200 on BOTH tokenhub and flagos — identical to a real or fake
     signature (three arms, all 200).

The old rule had a live victim: tokenhub emits NO signature_delta at all
(n=5, no such event in the stream), so "unsigned means drop" destroyed 100% of
its reasoning on every persist and every replay.

New contract, split into two independent assertions:
  A. CONTENT IS NEVER DISCARDED — real thinking text is persisted and replayed.
  B. THE SIGNATURE IS NEVER FABRICATED — an endpoint that issued none yields
     signature=="", and a real signature is carried verbatim.

Both are checked field by field on the consumer's own block objects (the
replayed block list / the formatted message), not by stringification.
"""

from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider


def _provider(base_url="https://api.flagos.net"):
    return AnthropicProvider(model="DeepSeek-V4.1-Flash", api_key="x",
                             base_url=base_url)


def _blocks(msg):
    return msg["content"] if isinstance(msg["content"], list) else []


def _thinking(msg):
    return [b for b in _blocks(msg) if b.get("type") == "thinking"]


# ── A. content is never discarded ─────────────────────────────────────────

def test_unsigned_thinking_is_persisted_not_dropped():
    """The tokenhub shape: real thinking text, no signature at all.

    Ablation: the pre-fix code required a non-empty signature and emitted no
    thinking block here, erasing the text.
    """
    resp = {"thinking": "real reasoning text", "signature": "",
            "content": "the answer", "tool_calls": None}
    blocks = _thinking(_provider().format_assistant_message(resp))

    assert len(blocks) == 1, "the thought must survive even unsigned"
    assert blocks[0]["thinking"] == "real reasoning text"
    assert {"type": "text", "text": "the answer"} in _blocks(
        _provider().format_assistant_message(resp))


def test_capped_thinking_without_signature_is_still_persisted():
    """The exact object the live trial produced: cap `break`s before the
    trailing signature_delta, so signature is None while thinking is the full
    real stream. The text must be kept (only the signature is honestly empty).
    """
    resp = {"thinking": "x" * 47791, "signature": None,
            "content": None, "tool_calls": [{"id": "t1", "name": "n", "arguments": {}}]}
    msg = _provider().format_assistant_message(resp)
    types = [b["type"] for b in _blocks(msg)]

    assert types == ["thinking", "tool_use"], "capped thought text must survive"
    assert _thinking(msg)[0]["thinking"] == "x" * 47791


def test_replay_keeps_an_unsigned_block_with_its_text():
    """Replay must not swap real text for an empty slot.

    Ablation: the pre-fix implementation returned {"thinking":"","signature":""}
    here — the erasure bug.
    """
    prov = _provider()
    out = prov._passthrough_thinking([
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "why I chose LZ77", "signature": ""},
            {"type": "tool_use", "id": "t", "name": "n", "input": {}}]}])

    blocks = out[1]["content"]
    kept = [b for b in blocks if b.get("type") == "thinking"]
    assert len(kept) == 1, "exactly one reasoning slot"
    assert kept[0]["thinking"] == "why I chose LZ77", "real text must survive"


# ── B. the signature is never fabricated ──────────────────────────────────

def test_signature_is_empty_when_the_endpoint_issued_none():
    """'' and None are the same fact: no signature. Never invent one."""
    for sig in ("", None):
        msg = _provider().format_assistant_message(
            {"thinking": "t", "signature": sig, "content": "c", "tool_calls": None})
        blocks = _thinking(msg)
        assert len(blocks) == 1, sig
        assert blocks[0]["signature"] == "", sig


def test_real_signature_is_carried_verbatim():
    """Positive bound: a block the endpoint signed survives unchanged."""
    sig = "a642a99f4afa4f338c92e7ec8e2ee8ae"
    msg = _provider().format_assistant_message(
        {"thinking": "real reasoning", "signature": sig,
         "content": "the answer", "tool_calls": None})
    assert msg["content"][0] == {"type": "thinking", "thinking": "real reasoning",
                                 "signature": sig}
    assert msg["content"][1] == {"type": "text", "text": "the answer"}


def test_replay_keeps_a_signed_block_verbatim():
    signed = {"type": "thinking", "thinking": "why I chose LZ77",
              "signature": "970ff9bfd8284d3c9e4531eaebfdf3a0"}
    out = _replay(_provider(), [signed,
                                {"type": "tool_use", "id": "t", "name": "n", "input": {}}])
    kept = [b for b in out[1]["content"] if b.get("type") == "thinking"]
    assert kept == [signed], "a signed block must round-trip field for field"


# ── C. fallback slot only when there is no text at all ────────────────────

def test_replay_synthesizes_a_slot_only_when_no_text_exists():
    """A text-only (e.g. evicted/placeholder) turn still gains one slot, so a
    gateway demanding the reasoning field keeps working."""
    out = _replay(_provider(), "evicted assistant turn placeholder")
    blocks = out[1]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "", "signature": ""}
    assert blocks[1] == {"type": "text", "text": "evicted assistant turn placeholder"}


def test_empty_thinking_block_does_not_suppress_the_fallback():
    """An explicitly EMPTY thinking block carries no text -> fallback slot."""
    out = _replay(_provider(), [{"type": "thinking", "thinking": "", "signature": ""},
                                {"type": "text", "text": "answer"}])
    th = [b for b in out[1]["content"] if b.get("type") == "thinking"]
    assert th == [{"type": "thinking", "thinking": "", "signature": ""}]
    assert {"type": "text", "text": "answer"} in out[1]["content"]


def test_thinking_absent_entirely_leaves_turn_unchanged():
    msg = _provider().format_assistant_message({"content": "c", "tool_calls": None})
    assert msg["content"] == [{"type": "text", "text": "c"}]


def test_official_endpoint_is_untouched():
    """Byte-identical for api.anthropic.com: the whole routine must no-op."""
    msgs = [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "t", "signature": ""}]}]
    out = _provider("https://api.anthropic.com")._passthrough_thinking(msgs)
    assert out is msgs, "official path must return the caller's object unchanged"


# ── D. the two properties together, over the live trial's shape ───────────

def test_live_trial_shape_survives_persist_and_replay():
    """End-to-end: a capped turn is persisted WITH its text, replayed WITH its
    text, and never claims a signature it does not have."""
    prov = _provider()
    capped = prov.format_assistant_message(
        {"thinking": "y" * 6154, "signature": None,
         "content": None, "tool_calls": [{"id": "t", "name": "n", "arguments": {}}]})
    out = prov._passthrough_thinking([capped])

    found = []
    for msg in out:
        for b in (msg["content"] if isinstance(msg["content"], list) else []):
            if isinstance(b, dict) and b.get("type") == "thinking":
                found.append(b)
    assert len(found) == 1, "the turn's reasoning must still be on the wire"
    assert found[0]["thinking"] == "y" * 6154, "text preserved through replay"
    assert found[0]["signature"] == "", "unsigned stays unsigned — no fabrication"


def _replay(provider, content):
    return provider._passthrough_thinking(
        [{"role": "user", "content": "q"},
         {"role": "assistant", "content": content}])

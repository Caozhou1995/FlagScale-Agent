"""read_image must stay visible through the anthropic provider.

Root cause this guards: some Anthropic-compatible gateways accept an image
block only when it sits directly in a *user* message's content list, and
silently drop image blocks nested inside ``tool_result.content`` -- the shape
the Anthropic API itself documents. ``format_tool_result`` emits the nested
shape, so on such a gateway every read_image result was stripped and the model
answered "the image content wasn't available".

The measured discriminator (same endpoint, same model, 64x64 quadrant image):
  * image in a user message          -> "TL: red, TR: blue, BL: green, BR: yellow"
  * image nested in tool_result      -> "I can't determine the colors"
  * image hoisted into a user message -> all four colours correct

These tests assert the *request shape* built by ``_build_kwargs`` -- no network
needed -- plus one opt-in live check that the pixels really reach the model.
"""

import base64
import io

import pytest

from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider
from flagscale_agent.react.tools.base import ImageResult


def _quadrant_png_b64():
    """64x64 PNG: TL red, TR blue, BL green, BR yellow."""
    Image = pytest.importorskip("PIL.Image")
    im = Image.new("RGB", (64, 64))
    px = im.load()
    for y in range(64):
        for x in range(64):
            if y < 32:
                px[x, y] = (255, 0, 0) if x < 32 else (0, 0, 255)
            else:
                px[x, y] = (0, 128, 0) if x < 32 else (255, 255, 0)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _provider():
    """A provider instance without constructing an SDK client (no network)."""
    p = AnthropicProvider.__new__(AnthropicProvider)
    p._model = "test-model"
    p._max_tokens = 1024
    p._temperature = None
    p._thinking_budget = 0
    p._auth_mode = "x-api-key"
    p._is_third_party = True
    return p


def _image_turn_messages(provider, image_result):
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "read the image at /tmp/q.png"},
        {"role": "assistant",
         "content": [{"type": "tool_use", "id": "t1", "name": "read_image",
                      "input": {"path": "/tmp/q.png"}}]},
        provider.format_tool_result("t1", image_result),
        {"role": "user", "content": [{"type": "text", "text": "describe it"}]},
    ]


def _count_image_blocks(messages):
    """(top-level image blocks, image blocks nested inside a tool_result)."""
    top = 0
    nested = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                top += 1
            elif block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                nested += sum(1 for i in block["content"]
                              if isinstance(i, dict) and i.get("type") == "image")
    return top, nested


def test_format_tool_result_still_emits_nested_shape():
    """The wire format the Anthropic API documents must not change."""
    prov = _provider()
    ir = ImageResult("text body", [{"media_type": "image/png", "data": "QUJD"}])
    msg = prov.format_tool_result("t1", ir)
    assert msg["role"] == "user"
    block = msg["content"][0]
    assert block["type"] == "tool_result"
    assert isinstance(block["content"], list)
    assert [b["type"] for b in block["content"]] == ["text", "image"]


def test_build_kwargs_hoists_image_out_of_tool_result():
    """The decisive assertion: no image may remain nested in a tool_result."""
    prov = _provider()
    ir = ImageResult("read_image -> delivered 64x64",
                     [{"media_type": "image/png", "data": _quadrant_png_b64()}])
    kwargs = prov._build_kwargs(_image_turn_messages(prov, ir), [], force_tool=False)
    top, nested = _count_image_blocks(kwargs["messages"])
    assert nested == 0, "image block left nested inside tool_result would be stripped"
    assert top == 1, f"expected exactly one hoisted image, got {top}"


def test_hoisted_message_immediately_follows_the_tool_result():
    """Order matters: the image must precede the user's next text turn."""
    prov = _provider()
    ir = ImageResult("body", [{"media_type": "image/png", "data": _quadrant_png_b64()}])
    kwargs = prov._build_kwargs(_image_turn_messages(prov, ir), [], force_tool=False)
    msgs = kwargs["messages"]
    idx = next(i for i, m in enumerate(msgs)
               if isinstance(m.get("content"), list)
               and any(isinstance(b, dict) and b.get("type") == "tool_result"
                       for b in m["content"]))
    nxt = msgs[idx + 1]
    assert any(b.get("type") == "image" for b in nxt["content"])
    # and the tool_result block itself now points at the hoisted pixels
    inner = msgs[idx]["content"][0]["content"]
    assert any("attached in the next user message" in i.get("text", "")
               for i in inner if isinstance(i, dict))


def test_no_image_history_is_byte_identical():
    """Batches that never return an image must be untouched by the hoist.

    ``_build_kwargs`` legitimately does two other things to any request (move the
    system message out, cache-control the last block), so this compares the
    hoisted path against ``_trim_old_images`` alone -- i.e. it isolates the new
    step rather than asserting against the raw input list.
    """
    prov = _provider()
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        prov.format_tool_result("t1", "plain text result"),
    ]
    _, chat = prov._split_system([dict(m) for m in msgs])
    baseline = prov._trim_old_images([dict(m) for m in chat])
    hoisted = prov._hoist_tool_result_images([dict(m) for m in chat])
    assert hoisted == baseline, "history without nested images must pass through unchanged"


def test_hoist_does_not_mutate_history():
    """History keeps the nested shape so a resumed session re-hoists correctly."""
    prov = _provider()
    ir = ImageResult("body", [{"media_type": "image/png", "data": _quadrant_png_b64()}])
    msgs = _image_turn_messages(prov, ir)
    _, nested_before = _count_image_blocks(msgs)
    prov._build_kwargs(msgs, [], force_tool=False)
    _, nested_after = _count_image_blocks(msgs)
    assert nested_before == nested_after == 1


@pytest.mark.skipif(
    not __import__("os").environ.get("FSAGENT_LIVE_VISION_PROBE"),
    reason="set FSAGENT_LIVE_VISION_PROBE=1 to hit the real endpoint",
)
def test_live_endpoint_actually_sees_the_pixels():
    """Opt-in: the request is not just well-shaped, the model reads the colours."""
    import json
    import os
    import urllib.request

    prov = _provider()
    ir = ImageResult("read_image -> delivered 64x64",
                     [{"media_type": "image/png", "data": _quadrant_png_b64()}])
    kwargs = prov._build_kwargs(
        _image_turn_messages(prov, ir) + [],
        [], force_tool=False)
    kwargs["messages"] = kwargs["messages"] + [
        {"role": "user", "content": [{"type": "text",
                                      "text": "Name the 4 quadrant colors TL,TR,BL,BR. One short line."}]}]
    body = {"model": os.environ.get("FSAGENT_MODEL", "DeepSeek-V4.1-Flash-1017"),
            "max_tokens": 1500, "messages": kwargs["messages"]}
    req = urllib.request.Request(
        os.environ.get("FSAGENT_BASE_URL", "https://api.flagos.net") + "/v1/messages",
        data=json.dumps(body).encode(),
        headers={"x-api-key": os.environ["FSAGENT_API_KEY"],
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    payload = json.loads(urllib.request.urlopen(req, timeout=180).read())
    seen = " ".join(b.get("text", "") + b.get("thinking", "")
                    for b in payload["content"]).lower()
    assert all(c in seen for c in ("red", "blue", "green", "yellow")), seen[:400]

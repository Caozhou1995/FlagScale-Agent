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

"""Regression tests: old images must not re-ship in every LLM request.

Measured failure (video-processing trial, 2026-10-24): the transcript carried
every historical read_image result, so a few images pushed each request body
over the gateway's 10MB limit -> HTTP 413 killed the session mid-run. The fix
replaces older image blocks with text placeholders once the base64 payload of
the kept images exceeds a byte ceiling.

The ceiling is counted in BYTES, not blocks. A count-of-blocks budget was the
original implementation and it was measurably wrong: it priced a 5 KB cropped
`region` view the same as a 3.5 MB photo. gcode-to-text (2026-11-27) then died
holding only one half of an image at a time — 49 read_image calls, each crop a
few KB, every earlier crop placeholder-dropped — and oscillated 68 times
between the two wrong readings of the same glyph before guessing wrong. These
tests pin the byte semantics in both directions: cheap images must SURVIVE, and
oversized ones must still be DROPPED (the 413 defence).
"""

import base64

from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider


def _provider():
    # Bypass __init__: it builds an API client; the method under test is pure.
    return AnthropicProvider.__new__(AnthropicProvider)


def _img(ch: str) -> str:
    return base64.standard_b64encode(bytes([ord(ch)] * 100)).decode()


def _img_b64(nbytes: int) -> str:
    """An image payload whose base64 length is exactly ``nbytes`` chars.

    Used by the byte-budget tests: the ceiling is metered in base64 characters,
    so a fixture must be able to state a payload size directly instead of
    indirectly via a raw byte count (base64 is 4/3 of raw, which makes the
    arithmetic in a test opaque).
    """
    return "A" * nbytes


def _tr(tid: str, text: str, img: str) -> dict:
    """A tool_result message carrying an image block (Anthropic nested shape)."""
    return {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": tid,
            "content": [
                {"type": "text", "text": text},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": img}},
            ],
        }],
    }


def _collect_images(messages):
    found = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                found.append(block["source"]["data"])
            inner = block.get("content")
            if block.get("type") == "tool_result" and isinstance(inner, list):
                for b in inner:
                    if isinstance(b, dict) and b.get("type") == "image":
                        found.append(b["source"]["data"])
    return found


def _count_placeholders(messages):
    n = 0
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            inner_blocks = [block]
            if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                inner_blocks = block["content"]
            for b in inner_blocks:
                if isinstance(b, dict) and b.get("type") == "text" \
                        and "dropped to" in str(b.get("text", "")):
                    n += 1
    return n


def test_keeps_only_within_byte_budget():
    """Oversized images are dropped once the byte budget is exhausted.

    This replaces the old `test_keeps_only_last_image`, which asserted a
    count-of-blocks invariant ("exactly 1 image survives"). That invariant was
    the bug: it forced the model to hold a single image no matter how small.
    The object under assertion is now the BYTE payload, not the block count.
    """
    # 3 images whose combined payload (~12M chars) exceeds the 6M ceiling.
    msgs = [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "read_image", "input": {}}]},
        _tr("t1", "Image: x.png", _img_b64(4_000_000)),
        _tr("t2", "Image: y.png", _img_b64(4_000_000)),
        _tr("t3", "Image: z.png", _img_b64(4_000_000)),
    ]
    p = _provider()
    _, chat = p._split_system(msgs)
    out = p._trim_old_images(chat)
    # Newest survives unconditionally; the other two exceed the ceiling.
    assert len(_collect_images(out)) == 1
    assert _count_placeholders(out) == 2


def test_small_crops_all_survive():
    """A few-KB crop must not cost a whole image slot (the gcode failure).

    Measured 2026-11-27: gcode-to-text made 49 read_image calls on ~5 KB crops
    and every earlier crop was dropped, so it could never hold "left half" and
    "right half" at once. Counting bytes lets all of them coexist — 32 x 5 KB
    is 0.16 MB, three orders of magnitude under the ceiling.
    """
    msgs = [{"role": "user", "content": "go"}]
    for i in range(32):
        msgs.append(_tr(f"t{i}", f"Image: crop{i}.png", _img_b64(5_000)))
    p = _provider()
    _, chat = p._split_system(msgs)
    out = p._trim_old_images(chat)
    assert len(_collect_images(out)) == 32, "cheap crops must not be dropped"
    assert _count_placeholders(out) == 0


def test_budget_is_bytes_not_blocks():
    """One big image plus several small ones: all fit, by arithmetic.

    4M chars (big) + 5 x 5 K chars (small) = ~4.03 M < 6 M ceiling, so nothing
    is dropped. Under the old count-of-blocks rule every small image after the
    big one was discarded regardless of its size.
    """
    msgs = [
        {"role": "user", "content": "go"},
        _tr("big", "Image: photo.png", _img_b64(3_000_000)),
    ]
    for i in range(5):
        msgs.append(_tr(f"s{i}", f"Image: crop{i}.png", _img_b64(5_000)))
    p = _provider()
    _, chat = p._split_system(msgs)
    out = p._trim_old_images(chat)
    assert len(_collect_images(out)) == 6
    assert _count_placeholders(out) == 0


def test_no_image_history_untouched():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        _tr("t1", "plain text result", None),  # text-only tool_result
    ]
    msgs[2]["content"][0]["content"] = [{"type": "text", "text": "plain"}]
    p = _provider()
    _, chat = p._split_system(msgs)
    out = p._trim_old_images(chat)
    assert out == chat
    assert _count_placeholders(out) == 0


def test_top_level_image_block_also_trimmed():
    # Non-nested image (e.g. a future tool attaching at message top level).
    # Both payloads are oversized so the byte ceiling (not the block count)
    # decides which one survives — the newest.
    big = 4_000_000
    msgs = [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text", "text": "Image: a.png"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": _img_b64(big)}},
            ]}]},
        {"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": _img_b64(big)}}]},
    ]
    p = _provider()
    _, chat = p._split_system(msgs)
    out = p._trim_old_images(chat)
    assert _collect_images(out) == [_img_b64(big)]
    assert _count_placeholders(out) == 1


def test_placeholder_tells_model_how_to_recover():
    big = 4_000_000   # each alone exceeds the ceiling -> the older one drops
    msgs = [_tr("t1", "Image: old.png", _img_b64(big)),
            _tr("t2", "Image: new.png", _img_b64(big))]
    p = _provider()
    _, chat = p._split_system(msgs)
    out = p._trim_old_images(chat)
    text = str(out[0]["content"][0]["content"][-1]["text"])
    assert "read_image" in text, "placeholder must name the recovery path"

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
keeps only the most recent image-bearing message and replaces older image
blocks with text placeholders.
"""

import base64

from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider


def _provider():
    # Bypass __init__: it builds an API client; the method under test is pure.
    return AnthropicProvider.__new__(AnthropicProvider)


def _img(ch: str) -> str:
    return base64.standard_b64encode(bytes([ord(ch)] * 100)).decode()


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


def test_keeps_only_last_image():
    msgs = [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "read_image", "input": {}}]},
        _tr("t1", "Image: x.png", _img("A")),
        _tr("t2", "Image: y.png", _img("B")),
        _tr("t3", "Image: z.png", _img("C")),
    ]
    p = _provider()
    _, chat = p._split_system(msgs)
    out = p._trim_old_images(chat)
    assert _collect_images(out) == [_img("C")]
    assert _count_placeholders(out) == 2


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
    msgs = [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text", "text": "Image: a.png"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": _img("A")}},
            ]}]},
        {"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": _img("B")}}]},
    ]
    p = _provider()
    _, chat = p._split_system(msgs)
    out = p._trim_old_images(chat)
    assert _collect_images(out) == [_img("B")]
    assert _count_placeholders(out) == 1


def test_placeholder_tells_model_how_to_recover():
    msgs = [_tr("t1", "Image: old.png", _img("A")),
            _tr("t2", "Image: new.png", _img("B"))]
    p = _provider()
    _, chat = p._split_system(msgs)
    out = p._trim_old_images(chat)
    text = str(out[0]["content"][0]["content"][-1]["text"])
    assert "read_image" in text, "placeholder must name the recovery path"

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

"""Regression tests for the vision path: read_image tool + image blocks.

The properties that matter, each of which would fail SILENTLY otherwise:
  * an image that reaches the tool must reach the PROVIDER (not be dropped by
    the registry's truncation, which slices plain strings);
  * a plain text result must be byte-identical to the pre-change behaviour, so
    batches that never read an image are unaffected;
  * the image is re-encoded/downscaled to fit a request, and the text SAYS so;
  * bad input produces an actionable error, not a traceback.
"""

import base64
import inspect
import io
import os

import pytest

from flagscale_agent.react.providers.anthropic_provider import AnthropicProvider
from flagscale_agent.react.tools.base import ImageResult, Tool
from flagscale_agent.react.tools.read_image import MAX_BYTES, MAX_EDGE, ReadImageTool

PIL = pytest.importorskip("PIL.Image", reason="Pillow needed to build test images")


def _write_png(path, w, h, colour_fn):
    img = PIL.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = colour_fn(x, y)
    img.save(path, format="PNG")
    return path


def _split(tmp_path, w=100, h=100):
    return _write_png(
        str(tmp_path / "split.png"), w, h,
        lambda x, y: (255, 0, 0) if x < w // 2 else (0, 0, 255),
    )


# ── ImageResult / provider ───────────────────────────────────────────────────

def test_image_result_is_a_string_so_existing_consumers_are_unaffected():
    r = ImageResult("text body", [{"media_type": "image/png", "data": "QQ=="}])
    assert isinstance(r, str)
    assert "text body" in r
    assert r.images[0]["media_type"] == "image/png"


def test_provider_emits_image_block_inside_tool_result():
    p = AnthropicProvider(model="m", api_key="k", max_tokens=10)
    r = ImageResult("look at this", [{"media_type": "image/png", "data": "QUJD"}])
    msg = p.format_tool_result("call_1", r)
    assert msg["role"] == "user"
    block = msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    inner = block["content"]
    assert isinstance(inner, list)
    assert inner[0] == {"type": "text", "text": "look at this"}
    assert inner[1]["type"] == "image"
    assert inner[1]["source"] == {
        "type": "base64", "media_type": "image/png", "data": "QUJD",
    }


def test_plain_string_tool_result_is_byte_identical_to_old_shape():
    """A batch that never returns an image must be unchanged."""
    p = AnthropicProvider(model="m", api_key="k", max_tokens=10)
    msg = p.format_tool_result("call_1", "just text")
    assert msg == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1",
                     "content": "just text"}],
    }


def test_empty_tool_result_still_says_empty():
    p = AnthropicProvider(model="m", api_key="k", max_tokens=10)
    msg = p.format_tool_result("c", "")
    assert msg["content"][0]["content"] == "(empty)"


# ── registry truncation must not drop the image ──────────────────────────────

def test_truncation_keeps_images():
    """The whole point of read_image is the image; clipping the text must not
    silently throw the pixels away."""
    from flagscale_agent.react.tools import ToolRegistry

    class BigTool(Tool):
        name = "big"
        description = "d"
        parameters = {"type": "object", "properties": {}}
        max_result_size = 10

        def execute(self, **kwargs):
            return ImageResult("x" * 100, [{"media_type": "image/png", "data": "QQ=="}])

    reg = ToolRegistry()
    reg.register(BigTool())
    out = reg.execute("big")
    assert len(out) > 10          # truncation happened
    assert "truncated" in out
    assert getattr(out, "images", None), "image was dropped by truncation"
    assert out.images[0]["data"] == "QQ=="


# ── read_image behaviour ────────────────────────────────────────────────────

def test_read_image_returns_text_plus_real_png(tmp_path):
    path = _split(tmp_path)
    out = ReadImageTool().execute(path=path)
    assert isinstance(out, ImageResult)
    assert path in out
    assert len(out.images) == 1
    img = out.images[0]
    assert img["media_type"] == "image/png"
    # payload must be valid base64 that decodes to a decodable PNG
    raw = base64.b64decode(img["data"])
    with io.BytesIO(raw) as fh:
        reopened = PIL.open(fh)
        assert reopened.size == (100, 100)


def test_read_image_downscales_and_says_so(tmp_path):
    path = _write_png(str(tmp_path / "huge.png"), MAX_EDGE * 2, MAX_EDGE,
                      lambda x, y: (x % 256, y % 256, 0))
    out = ReadImageTool().execute(path=path)
    assert "downscaled" in out
    raw = base64.b64decode(out.images[0]["data"])
    with io.BytesIO(raw) as fh:
        assert max(PIL.open(fh).size) <= MAX_EDGE


def test_byte_ceiling_shrink_is_reported_not_hidden(tmp_path):
    """A post-encode shrink must be stated; the text must never claim original pixels.

    Regression for an observed lie: a 1568x1568 noise PNG exceeded MAX_BYTES, the
    loop shrank it to ~1181px, and the reply still said
    "adjustments: none (you are seeing the original pixels)". The model then
    reasons about a reduced copy believing it saw the original.
    """
    import random
    random.seed(0)
    path = str(tmp_path / "noise.png")
    im = PIL.new("RGB", (MAX_EDGE, MAX_EDGE))
    px = im.load()
    for y in range(MAX_EDGE):
        for x in range(MAX_EDGE):
            px[x, y] = (random.randrange(256), random.randrange(256), random.randrange(256))
    im.save(path)

    out = ReadImageTool().execute(path=path)
    raw = base64.b64decode(out.images[0]["data"])
    delivered_max = None
    with io.BytesIO(raw) as fh:
        delivered_max = max(PIL.open(fh).size)

    assert delivered_max <= MAX_EDGE
    if delivered_max < MAX_EDGE:
        assert "adjustments: none" not in out, (
            "tool downscaled to %dpx but told the model it saw original pixels" % delivered_max
        )
        assert str(delivered_max) in out or "downscaled" in out


def test_read_image_region_crops(tmp_path):
    path = _split(tmp_path)
    out = ReadImageTool().execute(path=path, region="0,0,20,10")
    assert "cropped to region 0,0,20,10" in out
    raw = base64.b64decode(out.images[0]["data"])
    with io.BytesIO(raw) as fh:
        assert PIL.open(fh).size == (20, 10)


def test_read_image_region_out_of_bounds_is_actionable(tmp_path):
    path = _split(tmp_path)
    out = ReadImageTool().execute(path=path, region="90,90,50,50")
    assert out.startswith("ERROR:")
    assert "outside the image" in out
    assert not isinstance(out, ImageResult)


def test_missing_file_is_actionable(tmp_path):
    out = ReadImageTool().execute(path=str(tmp_path / "nope.png"))
    assert out.startswith("ERROR: no such file")


def test_unsupported_extension_is_actionable(tmp_path):
    p = tmp_path / "x.bmp"
    p.write_bytes(b"BM")
    out = ReadImageTool().execute(path=str(p))
    assert out.startswith("ERROR: unsupported image extension")


def test_corrupt_image_is_actionable_not_a_traceback(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not a png at all")
    out = ReadImageTool().execute(path=str(p))
    assert out.startswith("ERROR: could not decode image")


def test_tool_is_registered_on_the_agent():
    """far-end: the registry the agent builds must actually contain read_image."""
    from flagscale_agent.react import agent as agent_mod

    # Inspect the source-level registration instead of constructing a full agent
    # (which needs a live provider); the property that matters is that the tool
    # is registered in _register_tools.
    import inspect
    src = inspect.getsource(agent_mod.WorkerAgent._register_tools)
    assert "ReadImageTool()" in src
    # and the symbol it names must be importable from where the module imports it
    assert hasattr(agent_mod, "ReadImageTool")


def test_non_rgb_mode_conversion_is_reported(tmp_path):
    """RGBA->RGB discards alpha; the text must not claim original pixels."""
    p = str(tmp_path / "rgba.png")
    PIL.new("RGBA", (30, 30), (255, 0, 0, 128)).save(p)
    out = ReadImageTool().execute(path=p)
    raw = base64.b64decode(out.images[0]["data"])
    with io.BytesIO(raw) as fh:
        assert PIL.open(fh).mode != "RGBA"
    assert "none" not in [l for l in str(out).splitlines() if "adjustments" in l][0]
    assert "RGB" in out


def test_exif_orientation_is_applied_and_reported(tmp_path):
    """A stored 'rotate me' must be applied, not shown sideways with a 'none' claim."""
    p = str(tmp_path / "rot.jpg")
    im = PIL.new("RGB", (40, 20), (0, 0, 255))
    exif = im.getexif()
    exif[274] = 6  # rotate 90 CW
    im.save(p, exif=exif)
    out = ReadImageTool().execute(path=p)
    raw = base64.b64decode(out.images[0]["data"])
    with io.BytesIO(raw) as fh:
        got = PIL.open(fh).size
    assert got == (20, 40), f"orientation not applied, delivered {got}"
    assert "EXIF" in out


def test_encoded_size_stays_under_api_limit():
    """The raw ceiling must not encode past the 5 MB per-image API limit."""
    worst_b64 = len(base64.standard_b64encode(b"\0" * MAX_BYTES))
    assert worst_b64 <= 5 * 1024 * 1024, f"{worst_b64} b64 bytes exceeds 5 MB"


def test_cached_tool_result_keeps_images(tmp_path):
    """A cache hit must not downgrade ImageResult to str and drop the pixels.

    Behavioural, not source-text: build a real ImageResult, put it in the agent's
    turn cache, call execute_single twice, and require the SECOND (cached) result
    to still carry .images with the same payload.
    """
    from flagscale_agent.react.tool_executor import ToolExecutor

    p = _split(tmp_path)          # real image on disk
    img = ReadImageTool().execute(path=p)
    assert hasattr(img, "images") and img.images

    class _Reg:
        def execute(self, name, **kw):
            return img

    class _Agent:
        _tool_call_cache = {}
        tool_registry = _Reg()

    ex = ToolExecutor.__new__(ToolExecutor)
    ex._agent = _Agent()
    call = {"name": "read_image", "arguments": {"path": p}}
    first = ex.execute_single(call)
    second = ex.execute_single(call)          # cache hit
    assert "Cached result" in second
    assert hasattr(second, "images"), "cache hit dropped .images (model gets text only)"
    assert second.images[0]["data"] == first.images[0]["data"]

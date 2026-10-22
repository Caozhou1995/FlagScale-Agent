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

"""read_image tool: let the model actually LOOK at an image file.

Motivation (measured, not hypothetical): a blind agent that renders a figure and
then reasons about it only through numeric summaries picks the wrong convention
or misses a spatial fact the pixels state plainly. Reading the image is the
direct route; guessing from statistics is the slow, unreliable one.

The tool returns text (what the model reads) PLUS the encoded image, carried as
an ``ImageResult`` so the provider can attach it to the tool_result block.

Budget discipline: images are downscaled and re-encoded so one call cannot blow
the context or the request size. Everything that is *dropped* is reported in the
text, so the model knows it is looking at a reduced copy rather than the original.
"""

import base64
import io
import os
from typing import Any, Dict

from flagscale_agent.react.tools.base import ImageResult, Tool

# Formats accepted directly, mapped to the media type the API expects.
_SUPPORTED = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_PASSTHROUGH_MEDIA = {"image/png", "image/jpeg", "image/webp", "image/gif"}

# Hard ceilings. A 4 MB image is ~5.4 MB after base64, which is already a large
# single request; above that we downscale rather than fail, so a big screenshot
# is still usable.
MAX_BYTES = 4 * 1024 * 1024
MAX_EDGE = 1568          # Anthropic's recommended long-edge limit
DEFAULT_QUALITY = 85


class ReadImageTool(Tool):
    name = "read_image"
    description = (
        "Look at an image file and return it to you as a real image you can SEE "
        "(not a description): PNG/JPEG/WebP/GIF. Use this whenever the answer "
        "depends on what the pixels show — a rendered plot, a diagram, a "
        "screenshot, a frame montage you produced, a photo, a scanned figure — "
        "rather than guessing from the numbers. You may pass `region` to crop a "
        "sub-rectangle and `scale` to shrink it further before viewing. Large "
        "images are automatically downscaled; the reply states exactly what was "
        "done so you know whether you are seeing the original or a reduced copy."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the image file (relative to the working directory or absolute).",
            },
            "region": {
                "type": "string",
                "description": (
                    "Optional crop as 'x,y,width,height' in pixels. Use this to look "
                    "at one part of a large figure at full resolution."
                ),
            },
            "scale": {
                "type": "number",
                "description": "Optional extra downscale factor in (0, 1], applied after cropping.",
            },
        },
        "required": ["path"],
    }
    max_result_size = 20000

    def execute(self, **kwargs) -> str:
        path = kwargs.get("path") or ""
        region = kwargs.get("region")
        scale = kwargs.get("scale")

        if not path:
            return "ERROR: read_image needs a `path`."
        if not os.path.exists(path):
            return f"ERROR: no such file: {path}"
        if os.path.isdir(path):
            return f"ERROR: {path} is a directory, not an image file."

        ext = os.path.splitext(path)[1].lower()
        if ext not in _SUPPORTED:
            return (
                f"ERROR: unsupported image extension '{ext}'. Supported: "
                f"{', '.join(sorted(_SUPPORTED))}. If the file IS an image under "
                f"another name, convert it first (e.g. `python3 -c` with PIL)."
            )

        try:
            import PIL.Image as PILImage
        except ImportError:
            return (
                "ERROR: Pillow is not installed, so images cannot be inspected. "
                "Install it (`pip install pillow`) or fall back to numeric analysis."
            )

        notes = []
        raw_size = os.path.getsize(path)
        try:
            with PILImage.open(path) as im:
                orig_mode, orig_fmt = im.mode, im.format
                orig_w, orig_h = im.size
                work = im.convert("RGB") if im.mode not in ("RGB", "L") else im.copy()

            # 1) crop, if asked
            if region:
                try:
                    x, y, w, h = (int(v) for v in str(region).split(","))
                except Exception:
                    return (
                        "ERROR: `region` must be four integers 'x,y,width,height' "
                        f"(got {region!r})."
                    )
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > orig_w or y + h > orig_h:
                    return (
                        f"ERROR: region {x},{y},{w},{h} is outside the image "
                        f"({orig_w}x{orig_h})."
                    )
                work = work.crop((x, y, x + w, y + h))
                notes.append(f"cropped to region {x},{y},{w},{h}")

            # 2) extra scale, if asked
            if scale is not None:
                try:
                    s = float(scale)
                except Exception:
                    return f"ERROR: `scale` must be a number in (0,1], got {scale!r}."
                if not (0 < s <= 1):
                    return f"ERROR: `scale` must be in (0,1], got {s}."
                if s < 1:
                    nw, nh = max(1, int(work.width * s)), max(1, int(work.height * s))
                    work = work.resize((nw, nh), PILImage.LANCZOS)
                    notes.append(f"scaled by {s:g}")

            # 3) enforce the long-edge ceiling
            if max(work.width, work.height) > MAX_EDGE:
                ratio = MAX_EDGE / float(max(work.width, work.height))
                nw, nh = max(1, int(work.width * ratio)), max(1, int(work.height * ratio))
                work = work.resize((nw, nh), PILImage.LANCZOS)
                notes.append(
                    f"downscaled to fit the {MAX_EDGE}px long edge "
                    f"(original {orig_w}x{orig_h})"
                )

            data, media_type = self._encode(work, ext, notes)
        except Exception as exc:  # unreadable / corrupt image
            return f"ERROR: could not decode image {path}: {type(exc).__name__}: {exc}"

        # 4) statement after encoding: if even the compressed form is too big,
        #    shrink further rather than send a request the gateway may reject.
        guard = 0
        while len(data) > MAX_BYTES and guard < 6:
            guard += 1
            ratio = (MAX_BYTES / float(len(data))) ** 0.5
            nw = max(1, int(work.width * ratio))
            nh = max(1, int(work.height * ratio))
            if (nw, nh) == (work.width, work.height):
                break
            work = work.resize((nw, nh), PILImage.LANCZOS)
            data, media_type = self._encode(work, ext, notes)
        if len(data) > MAX_BYTES:
            return (
                f"ERROR: even after downscaling, {path} encodes to {len(data)} bytes "
                f"> {MAX_BYTES}. Use `region` to view one part at a time."
            )

        b64 = base64.standard_b64encode(data).decode("ascii")

        text = [
            f"Image: {path}",
            f"  original: {orig_w}x{orig_h} px, {orig_mode}, {orig_fmt or ext}, "
            f"{raw_size} bytes",
            f"  delivered: {work.width}x{work.height} px, {media_type}, "
            f"{len(data)} bytes",
        ]
        if notes:
            text.append("  adjustments: " + "; ".join(notes))
        else:
            text.append("  adjustments: none (you are seeing the original pixels)")
        text.append(
            "The image itself is attached to this result — answer from what you "
            "SEE, and say if a detail is too small to read at this resolution."
        )
        return ImageResult("\n".join(text), [{"media_type": media_type, "data": b64}])

    @staticmethod
    def _encode(work, ext, notes):
        """Encode ``work`` to bytes + media type, keeping the file format when we can."""
        import PIL.Image as PILImage

        buf = io.BytesIO()
        target = _SUPPORTED.get(ext, "image/png")
        if target not in _PASSTHROUGH_MEDIA:
            target = "image/png"
        if target == "image/jpeg":
            work.save(buf, format="JPEG", quality=DEFAULT_QUALITY, optimize=True)
        elif target == "image/webp":
            work.save(buf, format="WEBP", quality=DEFAULT_QUALITY)
        elif target == "image/gif":
            # A converted GIF loses animation; re-encoding as PNG is honest and
            # lossless for the still frame we are showing.
            work.save(buf, format="PNG", optimize=True)
            target = "image/png"
            notes.append("GIF re-encoded as PNG (still frame)")
        else:
            work.save(buf, format="PNG", optimize=True)
            target = "image/png"
        return buf.getvalue(), target

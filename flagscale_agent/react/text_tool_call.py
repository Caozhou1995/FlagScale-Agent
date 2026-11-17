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

"""Salvage a tool call that a model emitted as TEXT instead of via the native
tool-call channel.

Some models (observed with GLM / Anthropic-compatible endpoints) sometimes
write an Anthropic function-call block into the assistant's TEXT content
instead of streaming it through the structured tool_use channel:

    <invoke name="shell">
    <parameter name="command">echo hi</parameter>
    </invoke>

When that happens the kernel sees response["tool_calls"] == None and treats the
message as a plain-text turn, which drives a continuation / degenerate-repeat
loop -- the model wanted to act but the harness never ran the tool.

This module detects that case and re-routes it into tool_calls. The design is
deliberately conservative to avoid executing a block the model merely QUOTED
(e.g. inside a code fence while explaining itself):

  * TRAILING-ANCHORED: the invoke block(s) must sit at the very end of the
    message (only whitespace / function_calls wrappers may follow). A quote in
    the middle of prose is never salvaged.
  * KNOWN-TOOL-GATED: every tool name in the trailing run must be a real,
    registered tool. An unknown name aborts the whole salvage.
  * No XML parser and no regex: payloads carry unescaped <, > and & (shell
    commands, code), which break a real XML parser; a plain str.find scanner is
    both more robust here and keeps this off the regex path. Values are read as
    raw bytes between the parameter open and close tags.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

_INVOKE_OPEN = "<invoke name="
_INVOKE_CLOSE = "</invoke>"
_PARAM_OPEN = "<parameter name="
_PARAM_CLOSE = "</parameter>"


def _attr_value(segment: str, after: int) -> Tuple[Optional[str], int]:
    """Read a quoted attribute value starting at/after index `after`.

    Returns (value, index_just_past_closing_quote) or (None, after) if no
    quoted value is found. Tolerates single or double quotes and arbitrary
    whitespace/`=` between the attribute name token and the opening quote.
    """
    i = after
    n = len(segment)
    # Skip to the first quote char.
    while i < n and segment[i] not in ("'", '"'):
        # Stop early if we hit the end of this tag -- a malformed open tag.
        if segment[i] == ">":
            return None, after
        i += 1
    if i >= n:
        return None, after
    quote = segment[i]
    start = i + 1
    end = segment.find(quote, start)
    if end < 0:
        return None, after
    return segment[start:end], end + 1


def _parse_params(block: str) -> Dict[str, str]:
    """Extract {param_name: raw_value} from one invoke block body.

    The value is the raw bytes between <parameter name="X"> ...> and the
    matching </parameter>, taken verbatim (no unescaping) so unescaped <, >, &
    in shell/code survive intact.
    """
    params: Dict[str, str] = {}
    pos = 0
    while True:
        p = block.find(_PARAM_OPEN, pos)
        if p < 0:
            break
        name, after_name = _attr_value(block, p + len(_PARAM_OPEN))
        if name is None:
            break
        # Value starts after the '>' that closes the opening <parameter ...> tag.
        gt = block.find(">", after_name)
        if gt < 0:
            break
        val_start = gt + 1
        close = block.find(_PARAM_CLOSE, val_start)
        if close < 0:
            break
        params[name] = block[val_start:close]
        pos = close + len(_PARAM_CLOSE)
    return params


def _coerce(value: str, spec: Optional[dict]):
    """Coerce a raw string param value per its JSON-schema type.

    Unknown/absent spec -> keep the string. A failed coercion keeps the string
    too (the tool's own required-arg / type handling then applies), so salvage
    never raises on a surprising value.
    """
    if not isinstance(spec, dict):
        return value
    t = spec.get("type")
    try:
        if t == "integer":
            return int(value.strip())
        if t == "number":
            return float(value.strip())
        if t == "boolean":
            s = value.strip().lower()
            if s in ("true", "1", "yes"):
                return True
            if s in ("false", "0", "no"):
                return False
            return value
        if t in ("array", "object"):
            return json.loads(value)
    except (ValueError, TypeError, json.JSONDecodeError):
        return value
    return value


def _schema_props(schema: dict) -> Dict[str, dict]:
    """Return the properties dict from an openai- or anthropic-shaped schema."""
    if not isinstance(schema, dict):
        return {}
    # anthropic: {"name":..., "input_schema": {"properties": {...}}}
    inp = schema.get("input_schema")
    if isinstance(inp, dict) and isinstance(inp.get("properties"), dict):
        return inp["properties"]
    # openai: {"function": {"name":..., "parameters": {"properties": {...}}}}
    fn = schema.get("function")
    if isinstance(fn, dict):
        params = fn.get("parameters")
        if isinstance(params, dict) and isinstance(params.get("properties"), dict):
            return params["properties"]
    return {}


def _schema_name(schema: dict) -> Optional[str]:
    if not isinstance(schema, dict):
        return None
    if isinstance(schema.get("name"), str):
        return schema["name"]
    fn = schema.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    return None


def _strip_trailing_wrappers(text: str) -> str:
    """Remove trailing whitespace and any trailing function_calls wrapper
    close-tags, so the trailing-anchor test looks past cosmetic wrappers."""
    s = text.rstrip()
    while s.endswith("</function_calls>"):
        s = s[: -len("</function_calls>")].rstrip()
    return s


def salvage_tool_calls(
    content: Optional[str],
    schemas: Optional[List[dict]],
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Detect a tool call emitted as TEXT and convert it to tool_calls.

    Args:
        content: the assistant message text (response["content"]).
        schemas: the tool schemas list (openai- or anthropic-shaped), used both
            to gate on known tool names and to coerce parameter value types.

    Returns:
        (tool_calls, cleaned_text) when a trailing, known-tool invoke run is
        found:
          * tool_calls: list of {"id", "name", "arguments"} ready for the kernel.
          * cleaned_text: the message with the trailing invoke run removed
            (leading prose kept), or None when nothing but the block remained.
        (None, None) when there is nothing to salvage -- the caller keeps the
        original response untouched.
    """
    if not content or not isinstance(content, str):
        return None, None
    if _INVOKE_OPEN not in content:
        return None, None

    # Build the known-tool name set and a per-tool properties map.
    names: set = set()
    props_by_name: Dict[str, Dict[str, dict]] = {}
    for sch in (schemas or []):
        nm = _schema_name(sch)
        if nm:
            names.add(nm)
            props_by_name[nm] = _schema_props(sch)
    if not names:
        return None, None

    # The salvage is trailing-anchored: after the LAST </invoke> only whitespace
    # and function_calls wrapper close-tags may remain. A block quoted mid-prose
    # (e.g. inside a ``` fence while the model explains itself) is never trailing
    # and so is never salvaged.
    anchor = _strip_trailing_wrappers(content)
    if not anchor.endswith(_INVOKE_CLOSE):
        return None, None

    # Walk backwards collecting consecutive trailing invoke blocks. Each block is
    # [open .. close]; between two blocks only whitespace / wrapper tokens may
    # appear. Stop as soon as the gap carries real prose -> that prose is the
    # message body and the run of calls sits after it.
    blocks: List[Tuple[int, int]] = []  # (open_idx, close_end_idx) in `content`
    scan_end = len(anchor)
    while True:
        close_idx = anchor.rfind(_INVOKE_CLOSE, 0, scan_end)
        if close_idx < 0:
            break
        open_idx = anchor.rfind(_INVOKE_OPEN, 0, close_idx)
        if open_idx < 0:
            break
        # Whatever sits between this block's close and the previously collected
        # block's open must be wrapper/whitespace only.
        gap = anchor[close_idx + len(_INVOKE_CLOSE): scan_end]
        gap_clean = gap.replace("<function_calls>", "").replace("</function_calls>", "")
        if gap_clean.strip():
            break  # real text after this block -> not part of the trailing run
        blocks.append((open_idx, close_idx + len(_INVOKE_CLOSE)))
        scan_end = open_idx

    if not blocks:
        return None, None
    blocks.reverse()  # restore document order

    # Parse each block; abort the WHOLE salvage if any tool name is unknown.
    tool_calls: List[Dict[str, Any]] = []
    for k, (o, c) in enumerate(blocks):
        block = anchor[o:c]
        name, after_name = _attr_value(block, len(_INVOKE_OPEN))
        if not name or name not in names:
            return None, None
        raw_params = _parse_params(block)
        props = props_by_name.get(name, {})
        arguments = {pn: _coerce(pv, props.get(pn)) for pn, pv in raw_params.items()}
        tool_calls.append({
            "id": f"salvaged_text_{k}",
            "name": name,
            "arguments": arguments,
        })

    # Cleaned text = everything before the first salvaged block.
    first_open = blocks[0][0]
    cleaned = content[:first_open].rstrip()
    return tool_calls, (cleaned or None)

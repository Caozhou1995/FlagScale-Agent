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

"""Guard: validate tool call argument types against JSON schema before execution.

If the LLM produces a malformed argument (e.g. start_line: "[130]" instead of 130),
this guard blocks the call and returns a clear error so the LLM can retry with
correct types. This prevents TypeError crashes in both tool execution and post-guards.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# JSON schema type -> acceptable Python types
_TYPE_MAP = {
    "integer": (int,),
    "number": (int, float),
    "string": (str,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _try_decode_container(value: str, expected_type: str):
    """Return the decoded container when a str JSON-encodes the expected type.

    Returns None when `value` is not JSON, or decodes to a different type — the
    caller then reports the plain type mismatch. Used only to EXPLAIN a failure,
    never to accept the argument.
    """
    import json

    text = value.strip()
    if not text or text[0] not in "{[":
        return None
    try:
        decoded = json.loads(text)
    except (ValueError, TypeError):
        return None
    want = {"object": dict, "array": list}.get(expected_type)
    if want is not None and isinstance(decoded, want):
        return decoded
    return None


class ArgTypeGuard(Guard):
    """Block tool calls with arguments that don't match the declared schema types."""

    name = "arg_type"

    def __init__(self, tool_registry=None):
        self._tool_registry = tool_registry

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if self._tool_registry is None:
            return None

        try:
            tool = self._tool_registry.get(ctx.tool_name)
        except KeyError:
            return None

        props = tool.parameters.get("properties", {})
        if not props:
            return None

        errors = []
        for key, value in ctx.tool_args.items():
            if value is None:
                continue
            spec = props.get(key, {})
            expected_type = spec.get("type")
            if not expected_type:
                continue
            py_types = _TYPE_MAP.get(expected_type)
            if py_types and not isinstance(value, py_types):
                detail = (
                    f"  - '{key}': expected {expected_type}, "
                    f"got {type(value).__name__} ({repr(value)!s:.80})"
                )
                # A very common LLM failure mode: the caller JSON-ENCODES an
                # object/array and passes the resulting STRING instead of the
                # structured value. The type error is correct but uselessly
                # opaque — the value looks right in a log, so the caller keeps
                # re-sending it. Detect the specific case (a str whose content
                # parses to the expected container type) and say so outright,
                # rather than making the caller infer it from `got str`.
                if (isinstance(value, str)
                        and expected_type in ("object", "array")):
                    decoded = _try_decode_container(value, expected_type)
                    if decoded is not None:
                        detail += (
                            f"\n    ↑ you passed a JSON-encoded {expected_type} as "
                            f"a STRING. It parses correctly, so the content is "
                            f"fine — the problem is the ENCODING. Pass the "
                            f"structured {expected_type} itself (the tool-call "
                            f"argument must be a real {expected_type}, not its "
                            f"JSON serialization)."
                        )
                errors.append(detail)

        if errors:
            msg = (
                f"Invalid argument types for '{ctx.tool_name}'. "
                f"Fix the types and retry:\n" + "\n".join(errors)
            )
            return GuardVerdict.block(msg, reason="arg_type_mismatch", category="arg_type")

        return None

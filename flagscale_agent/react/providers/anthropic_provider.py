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

"""Anthropic provider implementation."""

import json

from typing import Any, Dict, Iterator, List

import anthropic

from flagscale_agent.react.providers.base import LLMProvider



class AnthropicProvider(LLMProvider):
    schema_format = "anthropic"

    def __init__(self, model: str, api_key: str, base_url: str = None, max_tokens: int = 8192, thinking_budget: int = 0, temperature: float = None):
        self._model = model
        self._max_tokens = max_tokens
        self._thinking_budget = thinking_budget
        self._temperature = temperature
        self._api_key = api_key
        self._base_url = base_url
        self._is_third_party = base_url and "anthropic.com" not in base_url
        self._auth_mode = None  # Will be auto-detected on first call
        self._timeout = 120.0  # 2-minute timeout for API calls + summarizer
        self._client = self._build_client()

    def _build_client(self):
        """Build Anthropic client with current auth mode."""
        kwargs = {"api_key": self._api_key, "timeout": self._timeout}
        if self._base_url:
            kwargs["base_url"] = self._base_url
            if self._is_third_party and self._auth_mode == "bearer":
                kwargs["api_key"] = "placeholder"
                kwargs["default_headers"] = {"Authorization": f"Bearer {self._api_key}"}
        return anthropic.Anthropic(**kwargs)

    def _switch_auth_and_retry(self):
        """Switch from x-api-key to Bearer auth after a 401."""
        if self._auth_mode == "bearer":
            return False  # Already tried Bearer, nothing more to do
        self._auth_mode = "bearer"
        self._client = self._build_client()
        return True

    def _split_system(self, messages):
        """Separate system message from chat messages (Anthropic requires this)."""
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)
        return system, chat_messages

    # Transcript-level image budget: EVERY request re-sends the whole history,
    # so N historical read_image results re-ship N images each call. Two 3MB
    # base64 images (~4MB encoded each) alone approach the gateway's 10MB
    # request limit and kill the session mid-run (measured: video-processing
    # trial died with HTTP 413 "request body too large" at turn 1). The model
    # needs the most recent image; older ones are replaceable — the tool text
    # still carries the measured facts, and the file remains on disk to re-read.
    MAX_REQUEST_IMAGE_BLOCKS = 1

    # Some Anthropic-compatible gateways accept an image block only when it sits
    # directly in a *user* message's content list, and silently drop image blocks
    # nested inside tool_result.content (the shape the Anthropic API documents).
    # Measured on this deployment's gateway: the same ImageResult answered the
    # quadrant colours correctly from a user message and answered "I can't
    # determine the colours — the image content wasn't available to read" when
    # nested in tool_result. Hoisting costs nothing where nesting works and is
    # the only thing that works where it does not, so it is unconditional.
    def _hoist_tool_result_images(self, chat_messages):
        """Move image blocks out of tool_result.content into a trailing user msg.

        Runs on the per-request copy only (same contract as ``_trim_old_images``),
        so history keeps the nested shape and a resumed session is unaffected.
        Returns a NEW list; messages without nested images are reused unchanged
        so batches that never return images stay byte-identical.
        """
        def _nested_images(content):
            for block in content:
                if not isinstance(block, dict):
                    continue
                inner = block.get("content")
                if block.get("type") == "tool_result" and isinstance(inner, list):
                    for item in inner:
                        if isinstance(item, dict) and item.get("type") == "image":
                            return True
            return False

        if not any(isinstance(m.get("content"), list) and _nested_images(m["content"])
                   for m in chat_messages):
            return chat_messages

        out = []
        for msg in chat_messages:
            content = msg.get("content")
            if not isinstance(content, list) or not _nested_images(content):
                out.append(msg)
                continue
            hoisted = []
            new_blocks = []
            for block in content:
                inner = block.get("content") if isinstance(block, dict) else None
                if isinstance(block, dict) and block.get("type") == "tool_result" \
                        and isinstance(inner, list):
                    new_inner = []
                    for item in inner:
                        if isinstance(item, dict) and item.get("type") == "image":
                            hoisted.append(item)
                        else:
                            new_inner.append(item)
                    new_inner.append({
                        "type": "text",
                        "text": ("[image pixels from this tool result are attached in "
                                 "the next user message]"),
                    })
                    block = dict(block)
                    block["content"] = new_inner
                new_blocks.append(block)
            new_msg = dict(msg)
            new_msg["content"] = new_blocks
            out.append(new_msg)
            if hoisted:
                out.append({"role": "user", "content": list(hoisted)})
        return out

    def _trim_old_images(self, chat_messages):
        """Keep images only in the LAST image-bearing tool_result of the history.

        Image blocks live INSIDE ``tool_result.content`` lists (Anthropic shape),
        so scan one level deep. Earlier image blocks are replaced by a text
        placeholder naming the tool that produced them, so a later turn can
        re-read the file if needed. Runs on the per-request copy only — history
        keeps the full images.
        """
        def _find_image_blocks(content):
            """Yield image blocks in a message's content, incl. inside tool_result."""
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "image":
                    yield block
                elif block.get("type") == "tool_result" and isinstance(block.get("content"), list):
                    yield from _find_image_blocks(block["content"])

        last_img_msg = None
        for msg in reversed(chat_messages):
            content = msg.get("content")
            if isinstance(content, list) and next(_find_image_blocks(content), None) is not None:
                last_img_msg = msg
                break
        if last_img_msg is None:
            return chat_messages
        kept = 0
        budget = self.MAX_REQUEST_IMAGE_BLOCKS
        for msg in reversed(chat_messages):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_blocks = []
            changed = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result" \
                        and isinstance(block.get("content"), list):
                    new_inner = []
                    for inner in block["content"]:
                        is_img = isinstance(inner, dict) and inner.get("type") == "image"
                        if is_img and (msg is not last_img_msg or kept >= budget):
                            new_inner.append({
                                "type": "text",
                                "text": ("[image from an earlier read_image dropped to "
                                         "stay under the request size limit; the tool "
                                         "text above still carries what it showed — "
                                         "re-run read_image on the same path to view "
                                         "it again]"),
                            })
                            changed = True
                        else:
                            if is_img:
                                kept += 1
                            new_inner.append(inner)
                    if changed:
                        block = dict(block)
                        block["content"] = new_inner
                    new_blocks.append(block)
                elif isinstance(block, dict) and block.get("type") == "image" \
                        and (msg is not last_img_msg or kept >= budget):
                    new_blocks.append({
                        "type": "text",
                        "text": ("[image from an earlier read_image dropped to "
                                 "stay under the request size limit; the tool "
                                 "text above still carries what it showed — "
                                 "re-run read_image on the same path to view "
                                 "it again]"),
                    })
                    changed = True
                else:
                    if isinstance(block, dict) and block.get("type") == "image":
                        kept += 1
                    new_blocks.append(block)
            if changed:
                msg["content"] = new_blocks
        return chat_messages

    def _passthrough_thinking(self, chat_messages):
        """Guarantee every assistant turn can carry `reasoning_content`.

        MEASURED FAILURE (2026-11-26, tokenhub = anthropic-compatible gateway):
        every LLM call died with HTTP 400
            invalid_request_error: "The `reasoning_content` in the thinking mode
            must be passed back to the API"
        even with thinking_budget=0 — the endpoint thinks unconditionally, and
        its anthropic->OpenAI translation requires each replayed assistant turn
        to carry the reasoning field.

        Two shapes in our history cannot carry it:
          1. an EVICTED assistant message, whose content history.py replaces
             with a plain placeholder STRING (`evict_message`);
          2. an assistant turn whose blocks were rebuilt as text-only (no
             thinking block ever appended).
        Both are legal on the wire for official Anthropic, which is why this
        went unnoticed until the endpoint changed.

        Fix is a superset of every reading: normalise assistant content to a
        block list and ensure a thinking block exists. An empty
        {"type":"thinking","thinking":"","signature":""} is accepted by the
        gateway (measured: probes with signature='' all returned 200), so the
        replay always satisfies the requirement while changing nothing about
        the turn's real text.

        Gated on third-party base_url: official api.anthropic.com neither needs
        nor is tolerant of synthesized thinking blocks, so its requests stay
        byte-identical to before.
        """
        if not self._is_third_party:
            return chat_messages

        out = []
        changed = False
        for msg in chat_messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                out.append(msg)
                continue
            content = msg.get("content")
            if isinstance(content, str):
                blocks = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                blocks = [dict(b) if isinstance(b, dict) else b for b in content]
            else:
                out.append(msg)
                continue
            # Drop ANY pre-existing reasoning block, then insert one uniform
            # empty block. MEASURED (N=4 per arm, real 120-message replay of
            # this session's own conversation.json):
            #   strip-all, insert nothing  -> 3 ok / 1 fail
            #   keep-existing, add if none -> 1 ok / 1 fail  (unstable)
            #   strip-all, insert empty    -> 4 ok / 0 fail
            # The middle arm is the one that made an earlier version of this
            # fix look WORSE than no fix: history carries real thinking blocks
            # from turns issued under a different model/session, and their
            # signatures are not trusted by the gateway, so passing them back
            # is itself a rejection source. Normalising to ONE synthesized empty
            # block removes that variable instead of adding one.
            blocks = [b for b in blocks
                      if not (isinstance(b, dict)
                              and b.get("type") in ("thinking", "redacted_thinking"))]
            # Place the reasoning slot first: the gateway reads it as the
            # turn's reasoning, and an empty payload keeps the text intact.
            blocks.insert(0, {"type": "thinking", "thinking": "",
                              "signature": ""})
            if blocks is not content:
                copy = dict(msg)
                copy["content"] = blocks
                out.append(copy)
                changed = True
            else:
                out.append(msg)
        return out if changed else chat_messages

    # ── Third-party gateway: reasoning-replay rejection ───────────────────
    # MEASURED 2026-11-26 on tokenhub (anthropic-compatible, base_url not
    # anthropic.com): replayed histories of ~120 messages intermittently die
    # with HTTP 400 "The `reasoning_content` in the thinking mode must be
    # passed back to the API". The SAME payload, re-sent unchanged, sometimes
    # returns 200 — 16 probes over three arms produced 3 rejections spread
    # across every shape (no thinking / real thinking / synthesized empty
    # thinking), and one arm passed 6/6 immediately after failing 2/2. That
    # pattern (identical bytes, different verdict) means the gateway balances
    # across upstreams that do not all enforce the rule, so NO single message
    # shape is a fix. What removes it is retrying the call, and degrading the
    # payload on the retry so a stricter upstream has nothing to object to.
    _REASONING_REPLAY_MARKER = "reasoning_content"

    def _is_reasoning_replay_error(self, exc) -> bool:
        """True for the gateway's reasoning_content replay rejection only."""
        try:
            text = str(exc)
        except Exception:
            return False
        return (self._REASONING_REPLAY_MARKER in text
                and "must be passed back" in text)

    def _strip_all_thinking(self, chat_messages):
        """Degraded payload: every assistant turn carries text, no reasoning.

        This is the shape the gateway can never reject for a missing reasoning
        field — there is no reasoning field to require. Assistant turns left
        with no blocks at all get an empty text block so the turn stays legal.
        """
        out = []
        for msg in chat_messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                out.append(msg)
                continue
            content = msg.get("content")
            if isinstance(content, str):
                blocks = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                blocks = [dict(b) if isinstance(b, dict) else b for b in content]
            else:
                out.append(msg)
                continue
            blocks = [
                b for b in blocks
                if not (isinstance(b, dict)
                        and b.get("type") in ("thinking", "redacted_thinking"))
            ]
            if not blocks:
                blocks = [{"type": "text", "text": ""}]
            copy = dict(msg)
            copy["content"] = blocks
            out.append(copy)
        return out

    def _degraded_kwargs(self, kwargs):
        degraded = dict(kwargs)
        degraded["messages"] = self._strip_all_thinking(
            kwargs.get("messages") or [])
        return degraded

    def _build_kwargs(self, messages, tools, force_tool=False):
        system, chat_messages = self._split_system(messages)
        chat_messages = self._hoist_tool_result_images(chat_messages)
        chat_messages = self._trim_old_images(chat_messages)
        # Must run AFTER the image passes (they rebuild content lists) and
        # BEFORE the cache_control rewrite below (which assumes a list).
        chat_messages = self._passthrough_thinking(chat_messages)
        kwargs = {"model": self._model, "max_tokens": self._max_tokens, "messages": chat_messages}
        if self._temperature is not None:
            # Route temperature via extra_body, NOT as a typed kwarg: anthropic
            # SDK 1.x removed temperature/top_p/top_k from the create()/stream()
            # signatures entirely, so passing it typed raises TypeError. extra_body
            # is merged verbatim into the JSON request body and is present on both
            # the 0.x and 1.x client lines, so this works across the unpinned
            # `anthropic>=0.39.0` range the container resolves.
            kwargs.setdefault("extra_body", {})["temperature"] = self._temperature
        if self._thinking_budget and self._thinking_budget > 0:
            # NOTE (user flag, 2026-10-30): the serving endpoint FORCES
            # thinking on regardless of this kwarg, so omitting it under
            # force_tool would not stop the model from thinking. The real
            # lever is client-side: when force_tool is armed, agent.py
            # stands down the token cap and the in-stream wall-clock cap so
            # the mandatory tool call can actually be reached (the thinking
            # is then bounded by the server-side max_tokens). This kwarg is
            # still omitted under force_tool to avoid requesting a budget
            # the enforced call will not honor.
            if force_tool:
                kwargs["max_tokens"] = self._max_tokens
            else:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}
        if system:
            # Split static body from dynamic dashboard for prompt caching.
            # The dashboard (Turn counter, memory keys) changes every turn and
            # must be AFTER the cache_control breakpoint to avoid invalidating
            # the cached static body. Do not write the static body's size here:
            # a measurement in a comment drifts silently as the prompt is edited.
            # Measure it with prompt.prompt_len() if you need the number.
            sep = "\n---\n["
            idx = system.find(sep)
            if idx >= 0:
                kwargs["system"] = [
                    {"type": "text", "text": system[:idx], "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": system[idx:]},
                ]
            else:
                kwargs["system"] = [
                    {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
                ]
        if tools:
            kwargs["tools"] = tools
            if force_tool:
                # Structural channel enforcement: the model must emit a
                # tool_use block; a prose-only answer is not a legal reply.
                kwargs["tool_choice"] = {"type": "any"}
        # Add cache_control to the last message to cache conversation prefix.
        # Copy to avoid mutating shared message references from history.
        if chat_messages:
            last = chat_messages[-1]
            last_copy = dict(last)
            content = last.get("content")
            if isinstance(content, str):
                last_copy["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}},
                ]
            elif isinstance(content, list):
                content_copy = [dict(b) if isinstance(b, dict) else b for b in content]
                if content_copy and isinstance(content_copy[-1], dict):
                    content_copy[-1]["cache_control"] = {"type": "ephemeral"}
                last_copy["content"] = content_copy
            chat_messages[-1] = last_copy
        return kwargs

    def _create_with_reasoning_retry(self, kwargs, attempts: int = 4):
        """Re-send a reasoning-replay rejection, degrading on later attempts.

        Deterministic part: the FIRST retry uses degraded kwargs (no reasoning
        blocks anywhere), because if the rejection is genuine that payload is
        the one a strict upstream must accept. Retries exist because the same
        bytes measured both 200 and 400 on this gateway, so a single resend is
        a real lever, not a ritual.
        """
        last = None
        for i in range(attempts):
            try:
                if i == 0:
                    return self._client.messages.create(**kwargs)
                return self._client.messages.create(
                    **self._degraded_kwargs(kwargs))
            except anthropic.BadRequestError as exc:
                if not self._is_reasoning_replay_error(exc):
                    raise
                last = exc
            except Exception:
                raise
        raise last

    def chat(self, messages: List[Dict[str, Any]], tools: List[dict],
             force_tool: bool = False) -> Dict[str, Any]:
        kwargs = self._build_kwargs(messages, tools, force_tool=force_tool)
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            if self._is_third_party and self._switch_auth_and_retry():
                response = self._client.messages.create(**kwargs)
            else:
                raise
        except anthropic.BadRequestError as exc:
            if not (self._is_third_party
                    and self._is_reasoning_replay_error(exc)):
                raise
            response = self._create_with_reasoning_retry(kwargs)

        content = None
        tool_calls = None
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input})

        return {"content": content, "tool_calls": tool_calls}

    def chat_stream(self, messages: List[Dict[str, Any]], tools: List[dict],
                    force_tool: bool = False) -> Iterator[Dict[str, Any]]:
        kwargs = self._build_kwargs(messages, tools, force_tool=force_tool)
        stream_ctx = None

        try:
            stream_ctx = self._client.messages.stream(**kwargs)
            stream = stream_ctx.__enter__()
        except anthropic.AuthenticationError:
            if self._is_third_party and self._switch_auth_and_retry():
                # Close old context before creating new one
                if stream_ctx is not None:
                    try:
                        stream_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
                stream_ctx = self._client.messages.stream(**kwargs)
                stream = stream_ctx.__enter__()
            else:
                raise
        except anthropic.BadRequestError as exc:
            # Same reasoning-replay rejection as chat(), same recovery: the
            # stream is opened lazily, so a rejected open has consumed nothing
            # and the call can be re-opened with a degraded payload.
            if not (self._is_third_party
                    and self._is_reasoning_replay_error(exc)):
                raise
            last = exc
            for attempt in range(4):
                try:
                    degraded = self._degraded_kwargs(kwargs)
                    stream_ctx = self._client.messages.stream(**degraded)
                    stream = stream_ctx.__enter__()
                    last = None
                    break
                except anthropic.BadRequestError as retry_exc:
                    if not self._is_reasoning_replay_error(retry_exc):
                        raise
                    last = retry_exc
            if last is not None:
                raise last

        stream_error = None
        final_message = None
        try:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        yield {"type": "tool_start", "id": block.id, "name": block.name}
                    elif block.type == "thinking":
                        yield {"type": "thinking_start"}
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield {"type": "text", "content": delta.text}
                    elif delta.type == "input_json_delta":
                        yield {"type": "tool_delta", "id": "", "arguments_delta": delta.partial_json}
                    elif delta.type == "thinking_delta":
                        yield {"type": "thinking", "content": delta.thinking}
                    elif delta.type == "signature_delta":
                        yield {"type": "signature", "content": delta.signature}
                elif event.type == "content_block_stop":
                    yield {"type": "block_stop"}
            
            # Get usage BEFORE closing the stream
            if stream_ctx is not None:
                try:
                    final_message = stream.get_final_message()
                except Exception:
                    pass
        except Exception as e:
            stream_error = e
        finally:
            if stream_ctx is not None:
                try:
                    stream_ctx.__exit__(None, None, None)
                except Exception:
                    pass

        if stream_error:
            raise stream_error

        # Yield usage data if we got it
        if final_message and final_message.usage:
            usage_data = {
                "type": "usage",
                "input_tokens": final_message.usage.input_tokens,
                "output_tokens": final_message.usage.output_tokens,
            }
            # Include cache info when prompt caching is active
            cache_read = getattr(final_message.usage, "cache_read_input_tokens", None)
            cache_create = getattr(final_message.usage, "cache_creation_input_tokens", None)
            if cache_read:
                usage_data["cache_read_input_tokens"] = cache_read
            if cache_create:
                usage_data["cache_creation_input_tokens"] = cache_create
            yield usage_data

        yield {"type": "done"}

    def format_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        content_blocks = []
        if response.get("thinking"):
            content_blocks.append({
                "type": "thinking",
                "thinking": response["thinking"],
                "signature": response.get("signature", ""),
            })
        if response["content"]:
            content_blocks.append({"type": "text", "text": response["content"]})
        if response["tool_calls"]:
            for tc in response["tool_calls"]:
                content_blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})
        return {"role": "assistant", "content": content_blocks}

    def format_tool_result(self, tool_call_id: str, content: str) -> Dict[str, Any]:
        # A tool may attach images (ImageResult: a str subclass carrying
        # .images). Anthropic takes them as sibling content blocks inside the
        # SAME tool_result block, which is the documented shape for returning
        # an image from a tool. A plain str takes the original path unchanged,
        # so batches that never return images are byte-identical to before.
        images = getattr(content, "images", None) or []
        text = str(content) if content else "(empty)"
        if images:
            blocks = [{"type": "text", "text": text}]
            for img in images:
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": img["data"],
                    },
                })
            inner = blocks
        else:
            inner = text
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": inner}],
        }

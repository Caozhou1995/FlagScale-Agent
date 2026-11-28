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


def _reply_has_tool_use(reply) -> bool:
    """True when a Messages API reply carries at least one `tool_use` block.

    The degraded shape (stage 1) drops `tool_choice`, so the model is no longer
    forced to call a tool. Measured on api.deepseek.com/anthropic it still does
    when a tool schema is present — but the caller's channel constraint must
    not depend on that holding, so the recovery path checks instead of trusting
    it. Tolerant by design: a test double or an unexpected reply shape reads as
    "no tool_use", which sends the path to the shape that restores forcing.
    """
    try:
        blocks = getattr(reply, "content", None) or []
        return any(getattr(b, "type", None) == "tool_use" for b in blocks)
    except Exception:
        return False


def _image_b64_len(block) -> int:
    """Base64 payload length of an Anthropic image block; 0 when unreadable.

    The transcript budget is metered in these characters because that is what
    the request actually carries — a cropped `region` view is a few KB while a
    full-size photo is a few MB, so a count-of-blocks budget prices them
    identically and drops the cheap ones for no size benefit. Tolerant by
    design: an unexpected shape reads as 0 bytes, which can only ever KEEP an
    image (never drop one we could have afforded).
    """
    try:
        if not isinstance(block, dict):
            return 0
        src = block.get("source")
        if isinstance(src, dict):
            data = src.get("data")
            if isinstance(data, str):
                return len(data)
        return 0
    except Exception:
        return 0


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
    # trial died with HTTP 413 "request body too large" at turn 1).
    #
    # MEASURED BUG — the budget must count BYTES, not blocks (2026-11-27):
    # this was `MAX_REQUEST_IMAGE_BLOCKS = 1`, i.e. it silently priced a 5 KB
    # crop the same as a 3.5 MB photo. read_image caps raw at 3.5 MB
    # (~4.67M base64 chars), but a cropped `region` view is typically a few KB.
    # gcode-to-text died on exactly this: 49 read_image calls, each crop ~5 KB,
    # and every earlier crop was replaced by a placeholder — so the model could
    # never hold "left half" and "right half" at once, and it oscillated 68
    # times between the two wrong readings of the same glyph ('0'/'o') before
    # guessing wrong. Its own trace names the mechanism: "the earlier images get
    # dropped … I claimed readings without seeing". Counting bytes keeps the
    # 413 defence intact (one maxed-out image still exhausts the budget: 4.67M
    # chars > 6M/2) while letting any number of small crops coexist (32 × 5 KB
    # = 0.16 MB, three orders of magnitude under the ceiling).
    MAX_REQUEST_IMAGE_BYTES = 6_000_000   # base64 chars of image data per request

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
        # Byte-metered budget, newest image first. `remaining` starts at the
        # ceiling and is charged the base64 length of every image we KEEP, so a
        # few-KB crop costs a few KB rather than a whole slot.
        remaining = self.MAX_REQUEST_IMAGE_BYTES
        seen_image = False
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
                        if is_img:
                            size = _image_b64_len(inner)
                            # The newest image is kept unconditionally: a
                            # maxed-out photo must never leave the model with
                            # zero pixels to reason about.
                            fits = (not seen_image) or size <= remaining
                            seen_image = True
                            if not fits:
                                new_inner.append({
                                    "type": "text",
                                    "text": ("[image from an earlier read_image dropped to "
                                             "stay under the request size limit; the tool "
                                             "text above still carries what it showed — "
                                             "re-run read_image on the same path to view "
                                             "it again]"),
                                })
                                changed = True
                                continue
                            remaining -= size
                        new_inner.append(inner)
                    if changed:
                        block = dict(block)
                        block["content"] = new_inner
                    new_blocks.append(block)
                elif isinstance(block, dict) and block.get("type") == "image":
                    size = _image_b64_len(block)
                    fits = (not seen_image) or size <= remaining
                    seen_image = True
                    if not fits:
                        new_blocks.append({
                            "type": "text",
                            "text": ("[image from an earlier read_image dropped to "
                                     "stay under the request size limit; the tool "
                                     "text above still carries what it showed — "
                                     "re-run read_image on the same path to view "
                                     "it again]"),
                        })
                        changed = True
                        continue
                    remaining -= size
                    new_blocks.append(block)
                else:
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
        block list and ensure a reasoning block exists. An empty
        {"type":"thinking","thinking":"","signature":""} is accepted by the
        gateway (measured: probes with signature='' all returned 200), so the
        replay always satisfies the requirement while changing nothing about
        the turn's real text.

        UPDATED 2026-11-28: that empty block is now the FALLBACK, not the rule.
        A real thinking block whose signature is non-empty is kept verbatim —
        the vendor doc for thinking-mode tool continuation asks for the full
        assistant content array to be replayed as-is, and a signed block is the
        endpoint's own so replaying it is what the doc requires. Blocks whose
        signature is empty are NOT trustworthy (they cannot be verified against
        the endpoint) and are dropped, which is what keeps the tokenhub
        reasoning-field requirement satisfied without shipping a provenance the
        block does not have.

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
            # Keep a REAL reasoning block when it is verifiably the endpoint's
            # own: a non-empty signature is the proof (see
            # format_assistant_message for the measured event ordering — the
            # signature arrives as one trailing signature_delta). MEASURED
            # 2026-11-28 on flagos, the backend the write-compressor trial ran
            # on: turn-1 blocks carry thinking_len 11723/14621 with a 32-hex-char
            # signature, and replaying that full block returns 200.
            #
            # This REVERSES the earlier strip-all rule below, and the reversal
            # is evidence-driven, not a re-reading of the old note: the old
            # N=4 measurement mixed two different populations — blocks whose
            # signatures came from a DIFFERENT model/session (not trusted, and
            # a genuine rejection source) and blocks from this same session
            # (valid). Dropping only the untrustworthy ones separates them.
            # The vendor doc says to replay the full content array verbatim;
            # keeping valid blocks is what satisfies it, and small stray-block
            # risk is no reason to delete what the authority requires.
            kept = [b for b in blocks
                    if not (isinstance(b, dict)
                            and b.get("type") in ("thinking", "redacted_thinking"))]
            valid = [b for b in blocks
                     if isinstance(b, dict)
                     and b.get("type") == "thinking"
                     and (b.get("signature") or "").strip()]
            blocks = valid + kept
            # Only when NOTHING valid survives — an evicted turn (placeholder
            # string content) or a turn built text-only — insert one empty
            # block so a gateway that demands the reasoning field still gets it.
            # A real-text/empty-signature block is never emitted: that shape
            # asserts a provenance it does not have.
            if not valid:
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

    # ── "Thinking mode does not support this tool_choice" ──
    # Measured 2026-11-26 (write-compressor__F6LwKJb): the kernel's endgame
    # window armed force_tool, so the request carried tool_choice={"type":"any"};
    # the gateway answered HTTP 400 "'Thinking mode does not support this
    # tool_choice'" and the turn DIED — 660.7s of work, ~4min of budget left,
    # reward 0. The rejection is not about our bytes being malformed: 44 probes
    # of the identical shape (thinking on, tool_choice any, stream, temp 0,
    # thinking-bearing history) all returned 200 against the same gateway. Only
    # SOME upstreams behind the balancer are reasoner deployments that forbid
    # the combination — the same per-request coin flip already documented for
    # the reasoning_replay rejection above. So no "probe first, then decide"
    # scheme can help (a probe lands on a healthy upstream and proves nothing);
    # the only routing-independent recovery is to DEGRADE THIS CALL and retry.
    #
    # What the degradation must be: drop tool_choice, keep everything else.
    # Dropping thinking is not available — the endpoint forces thinking on
    # server-side and the provider already omits that kwarg under force_tool,
    # which is exactly why this fired with no `thinking` field in the request.
    _TOOLCHOICE_THINKING_MARKER = "tool_choice"

    def _is_toolchoice_thinking_error(self, exc) -> bool:
        """True for the gateway's thinking-vs-tool_choice rejection only."""
        try:
            text = str(exc).lower()
        except Exception:
            return False
        if self._TOOLCHOICE_THINKING_MARKER not in text:
            return False
        mentions_thinking = "thinking" in text or "reasoning" in text
        rejects = ("does not support" in text or "not support" in text
                   or "unsupported" in text or "not allowed" in text)
        return mentions_thinking and rejects

    def _is_generic_retryable_400(self, exc) -> bool:
        """A 400 we cannot attribute from its text — still worth one reshaping.

        Measured 2026-11-26, two DIFFERENT opaque bodies from the same gateway
        on otherwise-accepted payloads, each of which killed a turn outright:

            {'error': {'type': '400001', 'message': 'The request is invalid: the
             request was rejected by an internal MaaS component. Please check
             the request body, required fields, and request format. ...'}}

        That message names no field. 22 probes over the payload features our
        provider actually adds (max_tokens 1k..200k, cache_control on system /
        messages, synthesized empty thinking blocks, text-only assistant turns,
        inline temperature) all returned 200, so the rejection is a property of
        the UPSTREAM the balancer picked, not of a specific field — the same
        conclusion already measured for the reasoning_content replay rule.

        A 400 that arrives from an unattributable upstream is therefore NOT a
        verdict on our bytes, and treating it as permanent (which is what
        raised "LLM call failed" and ended the turn with minutes left on the
        clock) is the defect this predicate exists to close. Context-limit
        errors are excluded: those need eviction, not a resend.
        """
        from flagscale_agent.react.retry import _extract_status
        if _extract_status(exc) != 400:
            # Some SDK/proxy paths only put the code in the message text.
            try:
                text = str(exc)
            except Exception:
                return False
            if "error code: 400" not in text.lower():
                return False
        try:
            text_l = str(exc).lower()
        except Exception:
            return False
        from flagscale_agent.react.retry import _is_context_limit_error
        if _is_context_limit_error(exc):
            return False
        # A malformed-tool-pairing 400 is deterministic and reshapeless: the
        # history itself is wrong, and resending cannot repair it.
        if "tool_use" in text_l and "tool_result" in text_l:
            return False
        # NOT every 400 is unattributable. A 400 that names a cause we cannot
        # fix by reshaping the body is a verdict, and resending it just burns
        # the turn's remaining wall clock (4 identical retries = 4 round trips
        # of latency-for-nothing). Measured 2026-11-26 with a held-out vector
        # set: "invalid api key", "insufficient balance" and "jinja template
        # error" were all classified retryable by the version of this predicate
        # that returned True for every non-context-limit 400. These markers are
        # the credential / account / template / hard-limit classes: reshapeless
        # by construction, because none of them is a property of the payload
        # features the degrade stages strip (cache_control, temperature,
        # thinking, tool_choice).
        for fatal in ("invalid api key", "invalid x-api-key", "api key",
                      "authentication", "unauthorized", "permission",
                      "insufficient", "balance", "quota", "billing",
                      "not found", "does not exist", "unknown model",
                      "template", "jinja"):
            if fatal in text_l:
                return False
        return True

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
            # The endpoint thinks by DEFAULT when the request carries no
            # `thinking` field, so omitting the kwarg does not stop it. It DOES
            # honour an explicit {"type":"disabled"} (measured 2026-11-26:
            # 3/3 text-only replies, and 3/3 tool_use when combined with
            # tool_choice=any) — that shape is the stage-1 degrade in
            # _degrade_for_toolchoice. Under force_tool the kwarg is still
            # simply omitted rather than sent as disabled: the mandatory tool
            # call is reached with thinking allowed, and the client-side caps
            # are stood down so it can complete.
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

    # A rejection is a per-request coin flip, not a verdict on our bytes: the
    # gateway fronts several upstreams and only some enforce the replay rule.
    # Four back-to-back resends therefore sample too few upstreams -- measured
    # 2026-11-26, five consecutive user-visible failures each exhausted the old
    # four attempts. So retries are (a) numerous, (b) spread in TIME so the
    # balancer lands on another upstream, and (c) alternate the two shapes we
    # have (all-thinking-stripped / all-thinking-present), so no single
    # upstream's preference can reject every attempt.
    _MAX_REASONING_RETRIES = 50

    @staticmethod
    def _retry_backoff(i: int) -> None:
        """Short jittered wait before retry `i` (1-based); total stays bounded.

        Sleep defeats per-node stickiness: a burst of immediate resends can be
        routed to the same upstream, while a spaced retry is re-balanced.
        Worst case over 50 attempts is ~50s of waiting, which is far cheaper
        than a dead turn.
        """
        import random
        import time as _time
        _time.sleep(min(0.15 * i, 1.0) + random.uniform(0.0, 0.10))

    def _retry_kwargs(self, kwargs, i: int):
        """Payload shape for retry `i` (1-based): alternate the two shapes."""
        if i % 2 == 1:
            return self._degraded_kwargs(kwargs)
        return kwargs

    def _note_retry(self, i: int, attempts: int) -> None:
        """Leave a trace: an invisible retry cannot be diagnosed afterwards."""
        import sys as _sys
        shape = "stripped" if i % 2 == 1 else "normalised"
        _sys.stderr.write(
            "[llm-retry] reasoning-replay 400; attempt "
            f"{i}/{attempts - 1} shape={shape}\n")
        _sys.stderr.flush()

    # ── Recovery for the thinking-vs-tool_choice rejection ──
    # Three attempts is enough: unlike the reasoning-replay coin flip (which
    # needs ~50 samplings because most upstreams enforce that rule), here the
    # rejection is a property of one deployment class, and the FIRST retry
    # already carries the shape no deployment can reject — the same request
    # with tool_choice removed. Further attempts exist only to survive a
    # transient upstream error, not to re-roll the shape.
    _MAX_TOOLCHOICE_RETRIES = 3

    # Budget for an UNATTRIBUTABLE 400 (see _is_generic_retryable_400). Kept
    # small on purpose: each stage strips one more decorative feature, and the
    # point is to survive a bad upstream, not to brute-force a genuinely
    # malformed request.
    _MAX_GENERIC_400_RETRIES = 3

    def _drop_tool_choice(self, kwargs):
        """Stage 2 degrade: the identical request minus `tool_choice`."""
        degraded = dict(kwargs)
        degraded.pop("tool_choice", None)
        return degraded

    def _degrade_for_toolchoice(self, kwargs, stage):
        """Degrade a thinking-vs-tool_choice rejection, reasoning first.

        ORDERING RULE: this rejection means the gateway refuses `tool_choice`
        *while thinking is on*. The cheap, reversible lever is therefore the
        FORCING (`tool_choice`), not the REASONING. Two ladders were measured
        on api.deepseek.com/anthropic + deepseek-flash (2026-11-27):

            no thinking field + tool_choice:any -> 400 (the endpoint turns
              thinking ON by default, so this is really ON + FORCED)
            thinking:disabled + tool_choice:any -> 200, blocks=['tool_use']
            thinking:disabled + no tool_choice  -> 200, blocks=['tool_use']
            thinking:enabled  + no tool_choice  -> 200, blocks=['thinking',
                                                            'tool_use']

        The last row is the one that matters: with an explicit tool schema in
        the request the model emits `tool_use` WITHOUT being forced. So
        dropping `tool_choice` costs far less than dropping thinking — the
        channel constraint is a safety rail, while thinking is the reasoning
        that decides what to write. The first version of this method had it
        backwards (disable thinking, keep tool_choice); measured on the trial
        logs that ordering wrote artifacts on a reasoning-disabled call, and
        the single historical PASS for that task ran with ZERO such degrades.

        STAGE 1 therefore drops `tool_choice` and KEEPS thinking. STAGE 2 —
        only for an upstream that still refuses, or for a stage-1 reply that
        came back without the `tool_use` the caller asked for — RESTORES the
        caller's forcing and disables thinking. One of the two levers has to
        go on an endpoint that refuses the pair; when the reasoning-preserving
        shape fails to produce a tool call, the caller's channel constraint is
        the one that must survive.
        """
        degraded = dict(kwargs)
        if stage >= 2:
            degraded["thinking"] = {"type": "disabled"}
        else:
            degraded.pop("tool_choice", None)
        return degraded

    def _note_toolchoice_degrade(self, attempt: int) -> None:
        """Trace the degradation: the transcript must explain what was lost.

        stderr is captured by the harness into the agent transcript, so this
        line is what makes "the endgame force_tool stopped forcing" diagnosable
        after the fact instead of looking like the mechanism never fired. It
        states WHICH lever went, because stage 1 keeps the REASONING and
        stage 2 does not — a single wording would misreport one of them.
        """
        import sys as _sys
        if attempt < 2:
            what = ("tool_choice DROPPED, thinking KEPT "
                    "(reasoning intact; forcing off for this call)")
        else:
            what = ("tool_choice DROPPED, thinking DISABLED "
                    "(no forcing AND no reasoning for this call)")
        _sys.stderr.write(
            f"[llm-degrade] gateway rejected thinking+tool_choice "
            f"({self._TOOLCHOICE_THINKING_MARKER}); retry {attempt} with {what}\n")
        _sys.stderr.flush()

    def _create_with_reasoning_retry(self, kwargs, attempts: int = None):
        """Re-send a reasoning-replay rejection until an upstream accepts it.

        Attempt 0 sends the caller's bytes untouched, so a healthy gateway sees
        byte-identical traffic to before this patch. Every later attempt is
        time-spaced and alternates shape (see _MAX_REASONING_RETRIES).
        """
        attempts = attempts or self._MAX_REASONING_RETRIES
        last = None
        for i in range(attempts):
            if i:
                self._note_retry(i, attempts)
                self._retry_backoff(i)
            try:
                if i == 0:
                    return self._client.messages.create(**kwargs)
                return self._client.messages.create(**self._retry_kwargs(kwargs, i))
            except anthropic.BadRequestError as exc:
                if not self._is_reasoning_replay_error(exc):
                    raise
                last = exc
            except Exception:
                raise
        raise last

    def _create_with_toolchoice_recovery(self, kwargs):
        """Re-issue after the thinking-vs-tool_choice rejection, escalating
        from the LEAST destructive degrade (see _degrade_for_toolchoice).

        Stage 1 keeps the REASONING and drops the forcing. That is only a win
        if the caller still gets its tool call, so this method checks the
        response instead of assuming: when the caller sent `tool_choice` and
        the degraded reply came back without `tool_use`, the forcing is
        restored and the reasoning is what gives way (stage 2). Without that
        check, trading forcing for reasoning could silently turn a forced tool
        call into prose.

        It re-issues until the gateway accepts, and raises the last rejection
        only if even the fully degraded shape is refused — which would mean
        tool_choice was not the objection after all.
        """
        require_tool = bool(kwargs.get("tool_choice"))
        unforced_without_tool = False
        last = None
        for attempt in range(1, self._MAX_TOOLCHOICE_RETRIES + 1):
            # A stage-1 reply that arrived without `tool_use` means the
            # unforced model chose prose; go straight to the shape that keeps
            # the forcing instead of repeating stage 1.
            stage = attempt + 1 if unforced_without_tool else attempt
            self._note_toolchoice_degrade(stage)
            self._retry_backoff(attempt)
            degraded = self._degrade_for_toolchoice(kwargs, stage)
            try:
                reply = self._client.messages.create(**degraded)
            except anthropic.BadRequestError as exc:
                if self._is_toolchoice_thinking_error(exc):
                    last = exc
                    continue
                if self._is_third_party and self._is_reasoning_replay_error(exc):
                    # Two rejections on one call: hand the already-degraded
                    # payload to the replay retry rather than losing the turn.
                    return self._create_with_reasoning_retry(degraded)
                raise
            except Exception:
                raise
            if require_tool and not _reply_has_tool_use(reply):
                # The unforced reply lost the tool call: reasoning must give
                # way instead of the caller's channel constraint.
                unforced_without_tool = True
                last = last or RuntimeError(
                    "stage-1 degrade returned no tool_use; restoring forcing")
                continue
            return reply
        raise last

    def _degrade_for_generic_400(self, kwargs, stage):
        """Reshape a request the gateway refused for an unattributable reason.

        Each stage removes one decorative feature that the endpoint does not
        need but a strict upstream may object to. Everything that carries the
        CONVERSATION and the CHANNEL CONSTRAINT survives to the last stage, so a
        degraded call is still a usable call.
        """
        degraded = dict(kwargs)
        if stage >= 1:
            # Prompt-cache breakpoints are pure decoration on our side.
            system = degraded.get("system")
            if isinstance(system, list):
                degraded["system"] = [
                    {"type": "text", "text": b.get("text", "")}
                    if isinstance(b, dict) else b for b in system]
            msgs = degraded.get("messages") or []
            stripped = []
            for m in msgs:
                c = m.get("content") if isinstance(m, dict) else None
                if isinstance(c, list):
                    m = dict(m)
                    m["content"] = [
                        {k: v for k, v in b.items() if k != "cache_control"}
                        if isinstance(b, dict) else b for b in c]
                stripped.append(m)
            degraded["messages"] = stripped
        if stage >= 2:
            # temperature rides in extra_body; some upstreams reject the body.
            eb = dict(degraded.get("extra_body") or {})
            eb.pop("temperature", None)
            if eb:
                degraded["extra_body"] = eb
            else:
                degraded.pop("extra_body", None)
        if stage >= 3:
            degraded = self._degrade_for_toolchoice(degraded, 2)
        return degraded

    def _create_with_generic_recovery(self, kwargs):
        """Resend an unattributable 400 with progressively plainer bodies.

        Bounded and honest: if the plainest shape is refused too, the rejection
        is about the conversation itself and it is raised, so the caller still
        sees the real error rather than a fabricated empty reply.
        """
        last = None
        for attempt in range(1, self._MAX_GENERIC_400_RETRIES + 1):
            self._note_generic_degrade(attempt)
            self._retry_backoff(attempt)
            degraded = self._degrade_for_generic_400(kwargs, attempt)
            try:
                return self._client.messages.create(**degraded)
            except anthropic.BadRequestError as exc:
                if self._is_generic_retryable_400(exc):
                    last = exc
                    continue
                raise
            except Exception:
                raise
        raise last

    def _note_generic_degrade(self, attempt: int) -> None:
        import sys as _sys
        shapes = {
            1: "cache_control breakpoints removed",
            2: "extra_body.temperature removed",
            3: "thinking disabled, tool_choice dropped",
        }
        _sys.stderr.write(
            f"[llm-degrade] gateway rejected the request with an unattributable "
            f"400; retry {attempt} with {shapes.get(attempt, 'plainer body')}\n")
        _sys.stderr.flush()

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
            if self._is_toolchoice_thinking_error(exc):
                # This rejection is a per-upstream policy, not a malformed
                # request: degrade rather than ending the turn (measured
                # 2026-11-26, write-compressor F6LwKJb — one 400 cost the
                # whole turn and the reward).
                response = self._create_with_toolchoice_recovery(kwargs)
            elif self._is_third_party and self._is_reasoning_replay_error(exc):
                response = self._create_with_reasoning_retry(kwargs)
            elif self._is_generic_retryable_400(exc):
                # Opaque 400 the gateway will not explain (e.g. the 400001
                # "rejected by an internal MaaS component" body). Measured
                # 2026-11-26: such a body can arrive for a payload the same
                # gateway accepts moments later, and treating it as permanent
                # killed a turn with minutes still on the clock.
                response = self._create_with_generic_recovery(kwargs)
            else:
                raise

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
            if self._is_toolchoice_thinking_error(exc):
                # The rejection this trial died on (2026-11-26, write-compressor
                # F6LwKJb): `messages.stream()` opens lazily, so a rejected open
                # has consumed nothing and NOTHING was written to the caller —
                # re-opening with the degraded shape is safe here.
                last = exc
                for attempt in range(1, self._MAX_TOOLCHOICE_RETRIES + 1):
                    self._note_toolchoice_degrade(attempt)
                    self._retry_backoff(attempt)
                    try:
                        stream_ctx = self._client.messages.stream(
                            **self._degrade_for_toolchoice(kwargs, attempt))
                        stream = stream_ctx.__enter__()
                        last = None
                        break
                    except anthropic.BadRequestError as retry_exc:
                        if self._is_toolchoice_thinking_error(retry_exc):
                            last = retry_exc
                            continue
                        raise
                    except Exception:
                        raise
                if last is not None:
                    raise last
            # Same reasoning-replay rejection as chat(), same recovery: the
            # stream is opened lazily, so a rejected open has consumed nothing
            # and the call can be re-opened with a degraded payload.
            elif self._is_generic_retryable_400(exc):
                # Opaque 400 (see _is_generic_retryable_400). The stream is
                # opened lazily, so nothing was consumed and nothing was
                # written to the caller — re-opening a plainer body is safe and
                # keeps the turn alive instead of ending it with
                # "LLM call failed".
                last = exc
                for attempt in range(1, self._MAX_GENERIC_400_RETRIES + 1):
                    self._note_generic_degrade(attempt)
                    self._retry_backoff(attempt)
                    try:
                        stream_ctx = self._client.messages.stream(
                            **self._degrade_for_generic_400(kwargs, attempt))
                        stream = stream_ctx.__enter__()
                        last = None
                        break
                    except anthropic.BadRequestError as retry_exc:
                        if self._is_generic_retryable_400(retry_exc):
                            last = retry_exc
                            continue
                        raise
                    except Exception:
                        raise
                if last is not None:
                    raise last
            elif not (self._is_third_party
                      and self._is_reasoning_replay_error(exc)):
                raise
            else:
                last = exc
                attempts = self._MAX_REASONING_RETRIES
                for attempt in range(1, attempts):
                    self._note_retry(attempt, attempts)
                    self._retry_backoff(attempt)
                    try:
                        retry_kwargs = self._retry_kwargs(kwargs, attempt)
                        stream_ctx = self._client.messages.stream(**retry_kwargs)
                        stream = stream_ctx.__enter__()
                        last = None
                        break
                    except anthropic.BadRequestError as retry_exc:
                        if not self._is_reasoning_replay_error(retry_exc):
                            raise
                        last = retry_exc
                    except Exception:
                        raise
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
        # A thinking block is only VALID when it carries the signature the
        # endpoint issued for it. MEASURED 2026-11-28 (flagos, the backend the
        # write-compressor trial actually ran on): the signature arrives as a
        # SINGLE `signature_delta` at the END of the thinking block, after all
        # the `thinking_delta`s. Anything that ends the stream early therefore
        # keeps the text and loses the signature.
        #
        # That is exactly what the runtime thinking cap does: agent.py breaks
        # out of the event loop on a cap hit, so the trailing signature_delta is
        # never consumed and `response["signature"]` is None while
        # `response["thinking"]` holds the full real text. Persisting that pair
        # produced the shape observed in that trial's conversation.json — five
        # thinking blocks of 47791/63619/307/178/6154 chars, EVERY ONE with
        # signature == "". A real-text/empty-signature block is the worst of the
        # available shapes: the vendor documents that the full content array
        # should be replayed verbatim, so an unverifiable block asserts a
        # provenance it does not have.
        #
        # Dropping it loses nothing: when the cap fires, the kernel re-injects
        # the accumulated reasoning as a <previous_reasoning> user message
        # (kernel.py, was_thinking_capped / was_reasoning_only branches), so the
        # text still reaches the model through a channel that does not pretend
        # to be a signed thinking block.
        sig = response.get("signature")
        if response.get("thinking") and sig:
            content_blocks.append({
                "type": "thinking",
                "thinking": response["thinking"],
                "signature": sig,
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

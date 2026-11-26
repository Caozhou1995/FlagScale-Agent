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

"""KnowledgeSkillGuard — MIDDLE-phase research discipline.

Part of the three-phase guard framework:
  • START  — StartupGuard: setup pipeline (backup, network probe, first research).
  • MIDDLE — this guard: CONTINUOUS research nudges throughout the task.
  • END    — VerificationGuard: completion-time verification.

Logic (MIDDLE phase only, ADVISORY ONLY — this guard never blocks):
- Track tool calls since last knowledge/skill/web_fetch load.
- Every 30 calls without a research load → inject a reminder.
- On the call that crosses 80 calls → inject an ESCALATED reminder once
  (the 30-call rhythm resumes afterwards).
- After any research load (web_fetch/load_knowledge/load_skill), inject an
  information-gain self-check (three-state: ACQUIRED / MISSING / SELF-FILLABLE).
- Meta tools (evict, plan, memory) don't count toward the threshold.

The 80-call threshold used to return GuardVerdict.block(). It is an inject now:
the gate still fires at the same call count, but "you have not consulted an
external reference for a while" is a reminder about a missing habit, not an
irreversibly-wrong action — blocking execution bought nothing the text did not,
and cost a full round trip each time. The escalation is latched (a flag cleared
only by a research load / override), because the block decision here reads a
PROJECTED counter that is never written back for a not-yet-executed call; a bare
`calls_since >= 80` inject would re-fire on every call past the mark.

The START-phase duties this guard USED to carry — the single-shot early
research gate and the network probe gate — now live in StartupGuard
(NetworkProbePhase + ResearchPhase). This guard no longer has any single-shot
mode; it is always-on and advisory/periodic.

Design parallel to MemoryDisciplineGuard but with looser thresholds, because
not every task requires domain knowledge.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


# web_fetch reports its outcome with a `[WEB_FETCH_*]` marker family (see
# react/tools/web_fetch.py). Exactly three of those markers mean the page body
# WAS retrieved and only its delivery was degraded — truncated, short, or
# non-extractable. Every other `[WEB_FETCH_*]` marker means the fetch failed
# before any content came back, so it must NOT take the "you fetched
# knowledge" branch. A single-marker test (`... == NETWORK_ERROR`) silently
# classified proxy refusals — the commonest failure on a filtered egress — as
# success. Match on the marker family's string prefix; never on a regex.
_SUCCESS_FETCH_MARKERS = (
    "[WEB_FETCH_SIZE_EXCEEDED",
    "[WEB_FETCH_LOW_CONTENT",
    "[WEB_FETCH_UNSUPPORTED",
)


def _is_fetch_failure(result_str: str) -> bool:
    """True iff a tool result is a web_fetch that returned NO page content.

    Anchored at the START of the result: every one of web_fetch's own returns
    is either the bare page text or a result that BEGINS with its marker, so a
    fetched page whose body merely mentions a marker mid-text must not be read
    as a failure.
    """
    upper = str(result_str).lstrip().upper()
    if any(marker in upper for marker in _SUCCESS_FETCH_MARKERS):
        return False
    return upper.startswith("[WEB_FETCH_")


class KnowledgeSkillGuard(Guard):
    """Remind agent to load knowledge/skills if it hasn't done so recently."""

    name = "knowledge_skill"
    priority = 85  # Low priority — advisory

    INJECT_THRESHOLD = 30
    BLOCK_THRESHOLD = 80

    _KNOWLEDGE_TOOLS = frozenset((
        "load_knowledge", "load_skill", "web_fetch",
    ))

    # Tools that don't count toward threshold (meta-operations)
    _META_TOOLS = frozenset((
        "evict", "recall",
        "plan_status", "plan_create", "plan_update",
        "memory_read", "memory_list", "memory_write",
    ))

    def __init__(self):
        self._calls_since_knowledge = 0
        # Set when a write is blocked with the "[PostEditBlocked]" mandate: the
        # guard whose block demands a cold re-read must not be the guard that
        # then blocks that very re-read. While set, the mandated read_file /
        # the re-issued edit_file to `self._mandated_path` are exempt from the
        # counter and the block (see _re_read_exempt).
        self._mandated_reverify = False
        self._mandated_path = ""
        # One-shot escalation latch for the 80-call mark: the decision below
        # reads a PROJECTED count (counter+1) that is never written back for a
        # call that has not executed yet, so a bare `>= BLOCK_THRESHOLD` inject
        # would re-fire on every call past the mark. The latch makes the
        # escalated text land exactly once per cycle.
        self._escalated = False
        # See memory_discipline: one-shot latch consumed but possibly suppressed.
        self._escalation_unreturned = False

    def _re_read_exempt(self, ctx: GuardContext) -> bool:
        """True when ctx is a guard-mandated cold re-read / re-issued edit.

        Exempting EXACTLY the mandated path (not a blanket read/edit pass)
        keeps the gate's discipline: an unrelated call still counts and still
        blocks at the threshold. Any real non-exempt call clears a stale
        mandate (see _persist_call_count); a turn boundary clears it too.
        """
        if not self._mandated_reverify:
            return False
        if ctx.tool_name not in ("read_file", "edit_file"):
            return False
        path = str((ctx.tool_args or {}).get("path") or "").strip()
        if not path or not self._mandated_path:
            return True
        return path == self._mandated_path

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        # Meta tools don't count (and never trigger a reminder). Checked FIRST so
        # plan/memory bookkeeping is never blocked.
        if ctx.tool_name in self._META_TOOLS:
            return None

        # Knowledge/skill loaded — reset counter.
        if ctx.tool_name in self._KNOWLEDGE_TOOLS:
            self._calls_since_knowledge = 0
            self._escalated = False
            return None

        # A guard-mandated cold re-read / re-issued edit is exempt: the guard
        # that demanded it must not block it. An edit consumes the mandate
        # (the re-issue is the delivery); a read keeps it (the re-issue edit
        # still follows).
        if self._re_read_exempt(ctx):
            if ctx.tool_name == "edit_file":
                self._mandated_reverify = False
            return None

        # Read-only projection for the block/inject DECISION. The persistent
        # increment happens in check_post AFTER the tool executed, so a blocked or
        # retried call does not inflate the count on every check_pre pass.
        calls_since = self._calls_since_knowledge + 1

        if calls_since >= self.BLOCK_THRESHOLD and not self._escalated:
            self._escalated = True
            self._escalation_unreturned = True
            return GuardVerdict.inject(
                f"[KnowledgeSkill] {calls_since} tool calls without loading "
                "domain knowledge, skills, or external references — this is the "
                "heavy mark. Before your next irreversible step, consider "
                "load_knowledge()/load_skill() (INTERNAL domains), web_fetch() "
                "(EXTERNAL domains, for any field where your prior knowledge may not "
                "reflect standard methods), or networked shell operations (git clone, "
                "pip/apt install, wget, curl downloads). Concluding "
                "'no better/other method exists' is an unverified knowledge gap — "
                "web_fetch the standard technique for the problem class first; you cannot "
                "prove 'no other method exists' with 'I do not know another method'. "
                "If you do decide to proceed without research, do NOT merely assert "
                "'I know this domain' or 'I have prior experience' — that bare claim is "
                "the self-exemption this gate targets. "
                "Instead make the exemption FALSIFIABLE: name the PROBLEM CLASS in one "
                "phrase AND the STANDARD METHOD you apply (the concrete algorithm / API / "
                "config name, not 'my experience'). If you cannot name BOTH concretely — "
                "if it comes out as 'some heuristic' or 'I'll tune it' — that vagueness IS "
                "the knowledge gap; research before proceeding.",
                reason=f"no_knowledge_load_{self.BLOCK_THRESHOLD}_calls",
                category="knowledge_skill",
            )

        if calls_since % self.INJECT_THRESHOLD == 0:
            return GuardVerdict.inject(
                f"[KnowledgeSkill] {calls_since} tool calls without "
                "loading domain knowledge, skills, or external references. Two channels: "
                "(1) INTERNAL domain → load_knowledge()/load_skill(). "
                "(2) EXTERNAL domain → web_fetch() for any field where your prior knowledge "
                "may not reflect standard methods. The trigger is a "
                "KNOWLEDGE GAP, not a syntax error — and you often will NOT feel the "
                "gap. Reaching for a hand-tuned threshold, a static assumption, or "
                "concluding 'no other method is available within this library/tool' is "
                "itself an unverified knowledge gap: web_fetch the standard technique for "
                "the problem class BEFORE committing. Before proceeding without research, "
                "run one falsifiable self-check: can you name this task's PROBLEM CLASS "
                "and the STANDARD METHOD (concrete algorithm / API / config) you apply? "
                "If yes, proceed. If the answer is a vague 'I'll figure it out' or 'some "
                "heuristic', that is the gap — look it up first.",
                reason="no_knowledge_load_recently",
                category="knowledge_skill",
            )

        return None


    def _persist_call_count(self, ctx: GuardContext) -> None:
        """Advance the persistent counter for an ACTUALLY-EXECUTED tool call.

        Mirrors the skip/reset rules of check_pre, but runs post-execution so a
        blocked or not-yet-run call never advances the count. Rules:
          • No tool name → nothing happened, skip.
          • Result carries the [BLOCKED BY GUARD] marker → the call was prevented,
            do NOT count it (this is the exact over-count source).
          • Knowledge tool executed → reset counter.
          • Meta tool → does not count.
          • Anything else that ran → advance the counter by one.
        """
        name = ctx.tool_name
        if not name:
            return
        result = ctx.tool_result
        if isinstance(result, str) and "[BLOCKED BY GUARD]" in result:
            return
        if name in self._KNOWLEDGE_TOOLS:
            self._calls_since_knowledge = 0
            self._escalated = False
            return
        if name in self._META_TOOLS:
            return
        # Exempt calls (the mandated re-read and the re-issued edit) do not
        # count — mirror check_pre, or the re-read would still advance the very
        # counter whose block it is escaping.
        if self._re_read_exempt(ctx):
            return
        # A real, executed tool call. Any non-exempt call makes a pending
        # mandate stale: the re-read/cleanup window is the immediately-following
        # turns, not an indefinite pass.
        self._mandated_reverify = False
        self._calls_since_knowledge += 1

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Persist the call-count increment HERE — after the tool has actually
        # executed. check_pre only READS a projected count for its block/inject
        # decision; doing the increment here means a tool call that was ultimately
        # blocked (its result carries the [BLOCKED BY GUARD] marker) or retried
        # does NOT inflate the counter on every check_pre pass.
        self._persist_call_count(ctx)

        # PostEditBlocked mandate: a write was blocked pending a cold re-read of
        # the far end. Record THAT path so the mandated read_file (and the
        # re-issued edit_file) are exempt from this guard's own counter/block —
        # otherwise the guard deadlocks with the guard that issued the mandate.
        if (ctx.tool_name in ("write_file", "edit_file")
                and isinstance(ctx.tool_result, str)
                and ctx.tool_result.startswith("[BLOCKED BY GUARD]")
                and "[PostEditBlocked]" in ctx.tool_result):
            self._mandated_reverify = True
            self._mandated_path = str(
                (ctx.tool_args or {}).get("path") or "").strip()

        # Information-gain inject for knowledge tools (inject, not block).
        # Aligns with the system prompt's Information Gain section: after any
        # knowledge acquisition (web_fetch / load_knowledge / load_skill), inject a
        # gentle nudge to self-check whether the retrieval produced NEW information
        # the agent did not already know. Advisory (inject), not a block.
        if ctx.tool_name in self._KNOWLEDGE_TOOLS and ctx.tool_result:
            result_str = str(ctx.tool_result)
            # A fetch that returned NO page must not take the success branch —
            # otherwise the agent is told to "classify what you now have" when
            # it has nothing, and answers from memory instead of switching
            # source. Match the marker FAMILY (see _is_fetch_failure), not one
            # marker: PROXY_REFUSED / UPSTREAM_5xx / HTTP_4xx / BLOCKED are all
            # failures, while SIZE_EXCEEDED / LOW_CONTENT / UNSUPPORTED are
            # degraded-but-acquired.
            if ctx.tool_name == "web_fetch" and _is_fetch_failure(result_str):
                return GuardVerdict.inject(
                    "[KnowledgeSkill] Your last web_fetch did NOT return the page — "
                    "the fetch failed before any content was retrieved, so you gained "
                    "NOTHING from it. The source is still the authority: a failed fetch "
                    "is NOT a licence to answer from memory. Do these IN ORDER:\n"
                    "  (1) SWITCH SOURCE — fetch the same fact from a MIRROR, an "
                    "alternate host/page for the same vendor or standard, or an archive "
                    "snapshot (e.g. web.archive.org). A different source that answers "
                    "the gap fully substitutes for the blocked one.\n"
                    "  (2) BYPASS THE HOP — if the failure names a proxy or upstream, "
                    "retry with the proxy unset (requests.get(url, proxies={}), or "
                    "urllib with an empty ProxyHandler + build_opener), and probe a "
                    "known-good domain to separate a per-host block from a network-wide "
                    "one.\n"
                    "  (3) OFFLINE SOURCE — installed package docs, load_knowledge, or a "
                    "local copy of the spec.\n"
                    "Re-anchoring the axis on your own remembered convention and marking "
                    "it NOT VERIFIED does NOT satisfy this: if the task text NAMED the "
                    "standard, what that standard requires is part of the task, and "
                    "'I could not fetch it' is a gap to close, not an excuse. "
                    "Information gain is the exit, not a retry of the same path.",
                    reason="web_fetch_failure_switch_source",
                    category="knowledge_skill",
                )
            # Success — classify the info-gain into one of three states so the
            # agent knows its NEXT move, not just whether it "learned something".
            return GuardVerdict.inject(
                "[KnowledgeSkill] You just fetched external knowledge. Classify what "
                "you now have into ONE of three states, then take that state's exit:\n"
                "  • ACQUIRED — the fetch answered the gap with something you did NOT "
                "already know. State the concrete gain (the fact/API/method) and "
                "PROCEED to apply it.\n"
                "  • MISSING — the gap is still open and it is NOT self-fillable "
                "(needs an external fact you don't have). Do NOT reason from memory — "
                "fetch AGAIN with a narrower query, a different source, or a mirror.\n"
                "  • SELF-FILLABLE — the fetch gave you enough that the remaining gap "
                "can be closed by REASONING from what you already have (inference, "
                "connecting known facts). Don't over-fetch — reason it out, then proceed.\n"
                "SPEC-LOYALTY CHECK (apply to every imported fact): imported information "
                "supplies HOW (methods, API shapes, algorithms) — it normally does not "
                "carry a decision about WHAT the deliverable must be. If the external "
                "standard answers a convention axis the task text left SILENT, that is "
                "smuggling, not gain: re-anchor that axis on the task text and the "
                "provided artifacts (minimal-uncertainty default), and keep the fetched "
                "method only for the parts the spec actually names. If the spec genuinely "
                "constrains nothing on the axis, say so explicitly and say why the axis "
                "is silent — do not substitute an external preference for the text's.\n"
                "  EXEMPTION — NOT SILENT IS NOT SMUGGLING. If the task text NAMES the "
                "standard (a kit, vendor, protocol, file format, or spec by name), then "
                "whatever that standard REQUIRES of the deliverable is part of what the "
                "task asked for. Apply it. Fetching it and then declining because 'the "
                "task does not require it' is wrong: naming the standard IS the "
                "requirement. The minimal-uncertainty default applies only to axes the "
                "text and the named standard BOTH leave open.\n"
                "If the fetch merely confirmed prior knowledge and added nothing (not "
                "any of the three), you are stalled — narrow the query or switch source.",
                reason="knowledge_info_gain_three_state",
                category="knowledge_skill",
            )
        return None

    def block_not_surfaced(self):
        """Hand back an escalation latch consumed by a verdict that never surfaced."""
        if self._escalation_unreturned:
            self._escalated = False
            self._escalation_unreturned = False

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Allow override of the call-count block on a substantive reason (>5 chars).

        The START-phase gates (single-shot early research, network probe/recovery)
        that used stricter, evidence-scoped overrides now live in StartupGuard.
        Only the periodic call-count block remains here, released by any
        substantive reason.
        """
        if not (reason and len(reason.strip()) > 5):
            return False
        self._calls_since_knowledge = 0
        self._escalated = False
        return True

    def reset_turn(self):
        """Don't reset per-turn — knowledge need persists across turns."""
        # ...but a PostEditBlocked mandate is a SINGLE-turn cleanup window: the
        # mandated re-read happens in the same turn as the block. Dropping it at
        # the boundary prevents an indefinite read/edit exemption.
        self._mandated_reverify = False
        self._mandated_path = ""
        # The escalation latch is per-turn: a fresh turn is a fresh chance to
        # research, so the heavy mark may land again in the next turn.
        self._escalated = False

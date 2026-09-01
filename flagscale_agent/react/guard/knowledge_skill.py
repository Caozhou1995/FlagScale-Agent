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

"""KnowledgeSkillGuard — reminds agent to load domain knowledge/skills proactively.

Logic:
- Track tool calls since last knowledge/skill load
- Every 15 calls without knowledge/skill load → inject a reminder
- Every 40 calls without knowledge/skill load → block (overridable)
- If LLM loads knowledge/skill, reset counter
- Meta tools (evict, plan, memory) don't count toward threshold

Design parallel to MemoryDisciplineGuard but with looser thresholds,
because not every task requires domain knowledge.
"""

from __future__ import annotations

from flagscale_agent.react.guard import Guard, GuardContext, GuardVerdict


class KnowledgeSkillGuard(Guard):
    """Remind agent to load knowledge/skills if it hasn't done so recently."""

    name = "knowledge_skill"
    priority = 85  # Low priority — advisory

    INJECT_THRESHOLD = 15
    BLOCK_THRESHOLD = 40

    _KNOWLEDGE_TOOLS = frozenset((
        "load_knowledge", "load_skill", "web_fetch",
    ))

    # Tools that don't count toward threshold (meta-operations)
    _META_TOOLS = frozenset((
        "evict", "recall",
        "plan_status", "plan_create", "plan_update",
        "memory_read", "memory_list", "memory_write",
    ))

    # Single-shot early gate: block after this many non-meta tool calls if
    # no research call has been made. Set to 3 to avoid collision with
    # BackupGuard (which blocks the 1st shell and may require a 2nd for backup).
    SINGLE_SHOT_EARLY_THRESHOLD = 3

    # Network-recovery gate: after web_fetch hits a network error, the agent
    # must make this many DISTINCT genuine recovery attempts (different URL/case,
    # proxy toggle, alternative source, offline cache) before it is allowed to
    # fall back to prior-knowledge work. Info gain from the network is the ONLY
    # way past a model's own capability ceiling on knowledge-gap tasks, so a mere
    # "network is restricted" assertion must NOT be an exit — only real attempts.
    REQUIRED_RECOVERY_ATTEMPTS = 5

    # Tokens that mark a shell command as a genuine network-access attempt.
    _NETWORK_CMD_TOKENS = (
        "curl", "wget", "urllib", "requests.get", "requests.post", "requests.head",
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
        "pip install", "pip download", "apt-get", "apt install",
        "git clone", "git fetch", "git pull",
        "nslookup", "dig ", "host ", "ping ", "nc ", "ncat", "telnet",
        "openssl s_client", "http.client", "socket.", "aiohttp", "httpx", "urlopen",
    )

    # Tokens that mark a shell command as a network PROBE (connectivity/speed test).
    # These are allowed through the single-shot early gate BEFORE _early_fired
    # is set, so the agent can test which sources are reachable and fast.
    _NETWORK_PROBE_TOKENS = (
        "curl -sI", "curl -si", "curl --head", "curl -I",
        "curl -sL", "curl -s ", "curl -o /dev/null",
        "wget --spider", "wget -q --spider",
        "ping -c", "ping -w",
        "nc -z", "nc -vz",
        "nslookup ", "dig ", "host ",
        "time curl", "time wget",
    )

    def __init__(self, single_shot: bool = False):
        self._calls_since_knowledge = 0
        # Single-shot mode: no human supervisor to nudge toward research, so
        # block after SINGLE_SHOT_EARLY_THRESHOLD non-meta tool calls if no
        # research call has been made. Set to 3 to avoid collision with
        # BackupGuard (which blocks the 1st shell and may require a 2nd).
        self._single_shot = single_shot
        # Count of non-meta tool calls in single-shot mode (excluding knowledge tools)
        self._single_shot_call_count = 0
        # Cleared ONLY when a knowledge tool (web_fetch/load_knowledge/load_skill)
        # is actually called — the single-shot early gate re-blocks every real
        # tool call until then.
        self._early_fired = False
        # Network probe gate: set True when the agent runs a network probe
        # command (curl/wget/ping etc.) in single-shot mode. The early gate
        # requires BOTH _network_probed AND _early_fired to clear.
        self._network_probed = False
        # Network-recovery gate state. Set True in check_post when web_fetch
        # reports a network error; while True, substantive non-network work is
        # blocked until REQUIRED_RECOVERY_ATTEMPTS distinct attempts are made.
        self._network_error_seen = False
        # Signatures (url / shell command) of recovery attempts already made this
        # episode — de-duplicated so re-running the SAME failing command does not
        # count. Forces genuinely DIFFERENT techniques, not repetition.
        self._recovery_signatures: set[str] = set()

    def set_single_shot(self, enabled: bool = True):
        """Enable single-shot early-advisory at runtime (set once run mode known)."""
        self._single_shot = enabled

    def _recovery_signature(self, ctx: GuardContext) -> str | None:
        """Return a de-dup signature if this call is a genuine network-recovery
        attempt, else None.

        A recovery attempt is either:
          • web_fetch (any URL — retrying with a different/case-flipped URL or a
            mirror is exactly what we want), signature = "web_fetch:<url>", or
          • a shell command that actually touches the network (curl/wget/urllib/
            requests/pip/apt/git-clone/dns tools/proxy toggles), signature =
            "shell:<normalized-command>".
        Signatures are normalized so re-running the identical command does not
        advance the quota — the gate wants DISTINCT techniques.
        """
        name = ctx.tool_name
        args = ctx.tool_args or {}
        if name == "web_fetch":
            url = str(args.get("url", "")).strip().lower()
            return f"web_fetch:{url}"
        if name == "shell":
            cmd = str(args.get("command", ""))
            if any(tok in cmd for tok in self._NETWORK_CMD_TOKENS):
                # Normalize whitespace so trivial reformatting is not a new sig.
                norm = " ".join(cmd.split())
                return f"shell:{norm}"
        return None

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        # Meta tools don't count (and don't trigger the early advisory).
        # Checked FIRST so plan/memory bookkeeping is never blocked by any gate.
        if ctx.tool_name in self._META_TOOLS:
            return None

        # Network-recovery gate is checked BEFORE the knowledge-tool branch on
        # purpose. load_knowledge/load_skill read LOCAL/INTERNAL knowledge and do
        # NOT touch the network, so they are NOT recovery attempts — yet they used
        # to slip through the knowledge branch's unconditional `return None`,
        # letting the agent "escape" a failed web_fetch by falling back to prior
        # knowledge (the exact wrong reflex on a knowledge-gap task). With the gate
        # first, only a genuine network attempt (web_fetch retry, or a network
        # shell cmd) is allowed through while the gate is armed; local knowledge
        # fallbacks are blocked until the recovery quota is met.

        # Network-recovery gate — DISABLED (overlaps with new network probe gate).
        # The new single-shot early gate already requires a network probe (curl -sI
        # --connect-timeout 3 --max-time 5) before any real work, making the
        # post-failure recovery gate redundant in most cases. To re-enable, uncomment
        # the block below and the check_post web_fetch error detection block.
        # if self._network_error_seen:
        #     sig = self._recovery_signature(ctx)
        #     if sig is not None:
        #         self._recovery_signatures.add(sig)
        #         if len(self._recovery_signatures) >= self.REQUIRED_RECOVERY_ATTEMPTS:
        #             self._network_error_seen = False
        #             self._recovery_signatures = set()
        #         if ctx.tool_name in self._KNOWLEDGE_TOOLS:
        #             self._calls_since_knowledge = 0
        #             self._early_fired = True
        #         return None
        #     remaining = self.REQUIRED_RECOVERY_ATTEMPTS - len(self._recovery_signatures)
        #     return GuardVerdict.block(
        #         "[NetworkResilience] web_fetch hit a NETWORK error and you are about "
        #         f"to move on with other work having made only {len(self._recovery_signatures)} "
        #         f"genuine recovery attempt(s) — {remaining} more required before you may "
        #         "fall back to prior knowledge. On a knowledge-gap task the model's own "
        #         "capability has a ceiling; real information from the network is the ONLY "
        #         "way past it, so 'the network is restricted, I'll use what I know' is the "
        #         "wrong reflex, not a valid exit. Make a DISTINCT recovery attempt now "
        #         "(each must differ from the last):\n"
        #         "  • Retry web_fetch with a different URL, a mirror, or flipped case.\n"
        #         "  • Toggle the proxy in a shell cmd: `env -u HTTP_PROXY -u HTTPS_PROXY "
        #         "curl -sSL <url>` (proxy may be the blocker) — or the reverse, ADD a "
        #         "proxy if none is set.\n"
        #         "  • Use urllib/requests directly in python3 against the URL.\n"
        #         "  • Try an alternative source: package mirror, archive, CDN, raw GitHub.\n"
        #         "This block releases automatically once you have made "
        #         f"{self.REQUIRED_RECOVERY_ATTEMPTS} distinct attempts — the exit is the "
        #         "ACTION, not an argument that it is unreachable. If a single attempt "
        #         "proved a hard, host-level block (e.g. proxy returns 500 for that exact "
        #         "host and a direct connection is refused), state that concrete evidence "
        #         "in the override reason.",
        #         reason="network_recovery_attempts_incomplete",
        #         category="network_resilience",
        #     )

        # Knowledge/skill loaded — reset counter. In single-shot, mark the early
        # advisory as satisfied: the agent already reached for external knowledge.
        # Reached only when the network gate is NOT armed (gate above handles the
        # armed case and only lets genuine network attempts through).
        if ctx.tool_name in self._KNOWLEDGE_TOOLS:
            self._calls_since_knowledge = 0
            self._early_fired = True
            return None

        # Single-shot early gate: NON-OVERRIDABLE block after
        # SINGLE_SHOT_EARLY_THRESHOLD non-meta tool calls if no research call
        # AND no network probe have been made. Set to 3 to avoid collision with
        # BackupGuard (which blocks the 1st shell and may require a 2nd for
        # backup). This gives the agent room to satisfy BackupGuard first.
        #
        # Two requirements to clear the gate:
        #   1. _network_probed: agent ran a network probe (curl/ping/dig etc.)
        #   2. _early_fired: agent made a research call (web_fetch/load_knowledge/
        #      load_skill)
        # Both must be satisfied. A network probe command is allowed THROUGH the
        # gate (returns None) so the agent can execute it, but does NOT clear
        # the research requirement.
        #
        # Why non-overridable: an overridable block was released by the agent's
        # own "I have solid prior experience" reason (observed: it named the
        # problem class, overrode, then hand-tuned to the one visible sample and
        # crashed on the hidden input). A self-issued exemption defeats the gate.
        # Making it non-overridable forces the corrective ACTION instead of an
        # argument — the agent must run a real research call, not merely assert it
        # does not need one.
        if self._single_shot and not (self._early_fired and self._network_probed):
            # Allow network probe commands through the gate so the agent can
            # test connectivity. Mark _network_probed so the gate knows this
            # requirement is met after execution (check_post will persist).
            if ctx.tool_name == "shell":
                cmd = str((ctx.tool_args or {}).get("command", ""))
                if any(tok in cmd for tok in self._NETWORK_PROBE_TOKENS):
                    # Network probe — let it through, persist flag in check_post
                    return None
            # Count this call as the (n+1)th for the BLOCK DECISION, but do NOT
            # persist the increment here — persistence happens in check_post,
            # AFTER the tool actually executed. This prevents a tool call that is
            # ultimately blocked (by BackupGuard etc.) or retried from inflating
            # the count on every check_pre pass. See _persist_pre_effects.
            projected = self._single_shot_call_count + 1
            if projected >= self.SINGLE_SHOT_EARLY_THRESHOLD:
                # Build the step-2 hint dynamically based on what's missing
                steps_done = []
                steps_needed = []
                if self._network_probed:
                    steps_done.append("network probe ✓")
                else:
                    steps_needed.append(
                        "STEP 1 — Network probe (FAST, each command MUST have --connect-timeout 3 --max-time 5):\n"
                        "  # Test official sources WITH and WITHOUT proxy:\n"
                        "  curl -sI --connect-timeout 3 --max-time 5 https://github.com\n"
                        "  curl -sI --connect-timeout 3 --max-time 5 https://pypi.org/simple\n"
                        "  env -u HTTP_PROXY -u HTTPS_PROXY curl -sI --connect-timeout 3 --max-time 5 https://github.com\n"
                        "  # Test mirrors (often faster in restricted environments):\n"
                        "  curl -sI --connect-timeout 3 --max-time 5 https://mirrors.tuna.tsinghua.edu.cn\n"
                        "  # Test task-relevant sources (e.g. astral.sh for uv, npm for node, etc.):\n"
                        "  curl -sI --connect-timeout 3 --max-time 5 https://astral.sh\n"
                        "  # Write results to memory so you don't re-test later.\n"
                    )
                if self._early_fired:
                    steps_done.append("research pass ✓")
                else:
                    steps_needed.append(
                        "STEP 2 — Research pass (choose one or more):\n"
                        "  web_fetch() — EXTERNAL domains (any field where your prior knowledge\n"
                        "    may not reflect the current standard method).\n"
                        "  load_knowledge() / load_skill() — INTERNAL FlagScale domains.\n"
                    )
                done_str = f" Done: {', '.join(steps_done)}." if steps_done else ""
                needed_str = "\n".join(steps_needed)
                return GuardVerdict.block(
                    f"[KnowledgeSkill] {projected} tool calls without network probe "
                    f"AND research pass. {done_str}\n\n"
                    f"{needed_str}\n"
                    "Both steps are required to clear this gate. This block is "
                    "NON-OVERRIDABLE: a text/tool-arg override will not release it, "
                    "and neither will meta tools (plan/memory/evict) — only running "
                    "the actual actions clears it. First name the PROBLEM CLASS, "
                    "then probe the network, then look up the standard technique. "
                    "Adopting the vocabulary of a 'structural/principled method' "
                    "without looking it up is exactly the failure this gate targets: "
                    "the dangerous case is when the example looks simple and you "
                    "feel no gap — that feeling is not evidence you hold the general "
                    "rule, it means you are about to hand-tune to the one visible "
                    "sample. Concluding 'no better method exists' from your own "
                    "memory is an unverified knowledge gap, not a fact. Run the "
                    "probe and lookup now.",
                    reason="single_shot_early_research_gate",
                    category="knowledge_skill",
                    overridable=False,
                )

        # Read-only projection for the block/inject DECISION. The persistent
        # increment happens in check_post AFTER the tool executed, so a blocked or
        # retried call does not inflate the count on every check_pre pass.
        calls_since = self._calls_since_knowledge + 1

        if calls_since >= self.BLOCK_THRESHOLD:
            # Do NOT reset counter here — only reset in accept_override if override succeeds
            return GuardVerdict.block(
                f"[KnowledgeSkill] {self.BLOCK_THRESHOLD} tool calls without loading "
                "domain knowledge, skills, or external references. Consider whether "
                "load_knowledge()/load_skill() (INTERNAL FlagScale domains) or web_fetch() "
                "(EXTERNAL domains, for any field where your prior knowledge may not "
                "reflect standard methods) would help. If you are about to conclude "
                "'no better/other method exists within this library or tool budget', that "
                "is an unverified knowledge gap — web_fetch the standard technique for the "
                "problem class before you commit to it; you cannot prove 'no other method "
                "exists' with 'I do not know another method'. "
                "If the task genuinely does not need domain knowledge and you have solid "
                "prior experience in its exact domain, override with a reason explaining why.",
                reason=f"no_knowledge_load_{self.BLOCK_THRESHOLD}_calls",
                category="knowledge_skill",
            )

        if calls_since % self.INJECT_THRESHOLD == 0:
            return GuardVerdict.inject(
                f"[KnowledgeSkill] {calls_since} tool calls without "
                "loading domain knowledge, skills, or external references. Two channels: "
                "(1) INTERNAL domain → load_knowledge()/load_skill() for FlagScale areas "
                "(parallelism, training config, NCCL, data pipeline, model porting). "
                "(2) EXTERNAL domain → web_fetch() for any field where your prior knowledge "
                "may not reflect standard methods. The trigger is a "
                "KNOWLEDGE GAP, not a syntax error — and note you often will NOT feel the "
                "gap. If you are about to reach for a hand-tuned threshold, a static "
                "assumption, or conclude 'no other method is available within this "
                "library/tool', that conclusion is itself an unverified knowledge gap: "
                "web_fetch the standard technique for the problem class BEFORE committing. "
                "If the task is genuinely straightforward and you have solid prior "
                "experience in its exact domain, proceed as normal.",
                reason="no_knowledge_load_recently",
                category="knowledge_skill",
            )

        return None

    def _persist_call_count(self, ctx: GuardContext) -> None:
        """Advance the persistent counters for an ACTUALLY-EXECUTED tool call.

        Mirrors the skip/reset rules of check_pre, but runs post-execution so a
        blocked or not-yet-run call never advances the counts. Rules:
          • No tool name → nothing happened, skip.
          • Result carries the [BLOCKED BY GUARD] marker → the call was prevented,
            do NOT count it (this is the exact over-count source).
          • Knowledge tool executed → reset counters + satisfy the early gate.
          • Meta tool → does not count.
          • Anything else that ran → advance both counters by one.
        """
        name = ctx.tool_name
        if not name:
            return
        result = ctx.tool_result
        if isinstance(result, str) and "[BLOCKED BY GUARD]" in result:
            return
        if name in self._KNOWLEDGE_TOOLS:
            self._calls_since_knowledge = 0
            self._early_fired = True
            return
        if name in self._META_TOOLS:
            return
        # Check if this was a network probe command — mark _network_probed.
        if name == "shell":
            cmd = str((ctx.tool_args or {}).get("command", ""))
            if any(tok in cmd for tok in self._NETWORK_PROBE_TOKENS):
                self._network_probed = True
        # A real, executed tool call.
        self._calls_since_knowledge += 1
        if self._single_shot and not (self._early_fired and self._network_probed):
            self._single_shot_call_count += 1

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # Persist the call-count increments HERE — after the tool has actually
        # executed. check_pre only READS a projected count for its block/inject
        # decision; doing the increment here means a tool call that was ultimately
        # blocked (its result carries the [BLOCKED BY GUARD] marker) or retried
        # does NOT inflate the counters on every check_pre pass. This fixes the
        # "count lags / over-counts by one" bug where a blocked first shell still
        # advanced the research/knowledge counters.
        self._persist_call_count(ctx)

        # Network-recovery inject — DISABLED (overlaps with new network probe gate).
        # The new single-shot early gate already requires a network probe before
        # any real work, making the post-failure recovery inject redundant.
        # To re-enable, uncomment the block below.
        # if ctx.tool_name == "web_fetch" and ctx.tool_result:
        #     result_upper = ctx.tool_result.upper()
        #     if "[WEB_FETCH_NETWORK_ERROR]" in result_upper:
        #         self._network_error_seen = True
        #         self._recovery_signatures = set()
        #         return GuardVerdict.inject(
        #             "[NetworkResilience] web_fetch reported a network error. ...",
        #             reason="web_fetch_network_failure",
        #             category="network_resilience",
        #         )
        return None

    # Evidence keywords that make an override reason RELEVANT to the network gate.
    # The gate only releases on a reason that actually argues network futility —
    # not on any reason that happens to satisfy some OTHER guard's block in the
    # same tool call (e.g. a BackupGuard "backup already made" reason). This is
    # the fix for the override-crosstalk bug: the registry calls EVERY blocking
    # guard's accept_override with the SAME ctx.override_reason, so a guard must
    # confirm the reason is aimed at ITS OWN gate before honoring it.
    _NETWORK_EVIDENCE_TOKENS = (
        "proxy", "host", "refused", "connection", "unreachable", "dns",
        "resolve", "timeout", "timed out", "500", "502", "503", "504",
        "403", "404", "network", "offline", "no route", "firewall",
        "cert", "ssl", "tls", "url", "mirror", "endpoint",
    )

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Allow override of a block if the LLM gives a substantive reason.

        Two DISTINCT gates live in this guard and they honor DIFFERENT reasons:

        • Call-count / early-research block: any substantive reason (>5 chars)
          releases it — the agent asserting it has solid prior experience.

        • Network-recovery block: releases ONLY on a reason that carries concrete
          NETWORK evidence (proxy/host/refused/500/dns/…). This prevents the
          override-crosstalk bug where the registry calls this method with an
          override reason the agent actually wrote for a DIFFERENT guard's block
          in the same tool call (e.g. a BackupGuard 'backup already made' reason).
          A reason with no network-futility content must NOT discharge the network
          gate — the agent has to articulate why the network is genuinely a dead
          end, matching what the block message explicitly asks for.
        """
        if not (reason and len(reason.strip()) > 5):
            return False

        # Network-recovery gate — DISABLED (overlaps with new network probe gate).
        # The _network_error_seen flag is never set now, so this branch is dead
        # code. To re-enable, uncomment the block below along with the check_pre
        # and check_post network-recovery blocks.
        # if self._network_error_seen:
        #     low = reason.lower()
        #     if not any(tok in low for tok in self._NETWORK_EVIDENCE_TOKENS):
        #         return False
        #     self._network_error_seen = False
        #     self._recovery_signatures = set()
        #     self._calls_since_knowledge = 0
        #     return True

        # No network gate armed — this is a call-count / early-research override.
        self._calls_since_knowledge = 0
        return True

    def reset_turn(self):
        """Don't reset per-turn — knowledge need persists across turns."""
        pass

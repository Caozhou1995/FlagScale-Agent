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

    def __init__(self, single_shot: bool = False):
        self._calls_since_knowledge = 0
        # Single-shot mode: no human supervisor to nudge toward research, so
        # fire one early advisory before the agent's first real (non-meta,
        # non-knowledge) tool call — the moment it is about to commit to an
        # implementation guessed from the sample instead of the standard method.
        self._single_shot = single_shot
        # Cleared ONLY when a knowledge tool (web_fetch/load_knowledge/load_skill)
        # is actually called — the single-shot early gate re-blocks every real
        # tool call until then.
        self._early_fired = False

    def set_single_shot(self, enabled: bool = True):
        """Enable single-shot early-advisory at runtime (set once run mode known)."""
        self._single_shot = enabled

    def check_pre(self, ctx: GuardContext) -> GuardVerdict | None:
        if not ctx.tool_name:
            return None

        # Knowledge/skill loaded — reset counter. In single-shot, mark the early
        # advisory as satisfied: the agent already reached for external knowledge.
        if ctx.tool_name in self._KNOWLEDGE_TOOLS:
            self._calls_since_knowledge = 0
            self._early_fired = True
            return None

        # Meta tools don't count (and don't trigger the early advisory)
        if ctx.tool_name in self._META_TOOLS:
            return None

        # Single-shot early gate: NON-OVERRIDABLE block that PERSISTS until the
        # agent actually calls a knowledge tool. Note we do NOT set _early_fired
        # here — only the _KNOWLEDGE_TOOLS branch (above) clears the gate. Every
        # real (non-meta, non-knowledge) tool call is re-blocked until then, so
        # the ONLY exit is a genuine research call: web_fetch / load_knowledge /
        # load_skill. Meta tools (plan/memory/evict) still pass through, so there
        # is no deadlock — the agent can always plan/record while forced to first
        # look up the standard technique.
        #
        # Why non-overridable: an overridable block was released by the agent's
        # own "I have solid prior experience" reason (observed: it named the
        # problem class, overrode, then hand-tuned to the one visible sample and
        # crashed on the hidden input). A self-issued exemption defeats the gate.
        # Making it non-overridable forces the corrective ACTION instead of an
        # argument — the agent must run a real research call, not merely assert it
        # does not need one.
        if self._single_shot and not self._early_fired:
            return GuardVerdict.block(
                "[KnowledgeSkill] First real action of this task is BLOCKED until "
                "you run a research pass. The ONLY way past this gate is to actually "
                "CALL one of: web_fetch() (EXTERNAL domain — any field where your "
                "prior knowledge may not reflect the current standard method), "
                "load_knowledge() or load_skill() (INTERNAL FlagScale domains). "
                "This block is NON-OVERRIDABLE: a text/tool-arg override will not "
                "release it, and neither will meta tools (plan/memory/evict) — only "
                "a real knowledge call clears it. First name the PROBLEM CLASS, then "
                "look up its standard technique. Adopting the vocabulary of a "
                "'structural/principled method' without looking it up is exactly the "
                "failure this gate targets: the dangerous case is when the example "
                "looks simple and you feel no gap — that feeling is not evidence you "
                "hold the general rule, it means you are about to hand-tune to the "
                "one visible sample. Concluding 'no better method exists' from your "
                "own memory is an unverified knowledge gap, not a fact. Run the "
                "lookup now.",
                reason="single_shot_early_research_gate",
                category="knowledge_skill",
                overridable=False,
            )

        self._calls_since_knowledge += 1

        if self._calls_since_knowledge >= self.BLOCK_THRESHOLD:
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

        if self._calls_since_knowledge % self.INJECT_THRESHOLD == 0:
            return GuardVerdict.inject(
                f"[KnowledgeSkill] {self._calls_since_knowledge} tool calls without "
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

    def check_post(self, ctx: GuardContext) -> GuardVerdict | None:
        # If web_fetch just failed due to network, inject the Environment Resilience
        # reminder from system prompt to surface network troubleshooting steps.
        if ctx.tool_name == "web_fetch" and ctx.tool_result:
            result_upper = ctx.tool_result.upper()
            # Match web_fetch.py's [WEB_FETCH_NETWORK_ERROR] marker
            if "[WEB_FETCH_NETWORK_ERROR]" in result_upper:
                return GuardVerdict.inject(
                    "[NetworkResilience] web_fetch reported a network error. Before giving up, "
                    "try systematic troubleshooting (these are standard techniques in restricted "
                    "container/CI environments):\n"
                    "  • **Proxy interference**: HTTP_PROXY/HTTPS_PROXY may block the target. "
                    "Try with proxy unset — use `env -u HTTP_PROXY -u HTTPS_PROXY curl ...` or "
                    "`python3 -c 'import os; [os.environ.pop(k,None) for k in [\"HTTP_PROXY\",\"HTTPS_PROXY\",\"http_proxy\",\"https_proxy\"]]; import urllib.request; ...'`.\n"
                    "  • **URL case sensitivity**: Many servers (FTP mirrors, CDNs) are case-sensitive. "
                    "If a URL returns 404, try UPPER and lower case variants.\n"
                    "  • **Alternative sources**: Search for mirrors, package archives, or alternative "
                    "download endpoints. A 403/404 on one host does not mean the resource does not exist.\n"
                    "  • **Offline fallback**: Check local caches (apt, pip, pre-installed packages, mounted volumes).\n"
                    "\n"
                    "Only after exhausting these should you conclude the network is truly unavailable. "
                    "Trying one fetch and accepting failure is premature — the system prompt's "
                    "Environment Resilience section describes this standard protocol.",
                    reason="web_fetch_network_failure",
                    category="network_resilience",
                )
        return None

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        """Allow override of block if LLM explains why knowledge isn't needed."""
        if reason and len(reason.strip()) > 5:
            self._calls_since_knowledge = 0
            return True
        return False

    def reset_turn(self):
        """Don't reset per-turn — knowledge need persists across turns."""
        pass

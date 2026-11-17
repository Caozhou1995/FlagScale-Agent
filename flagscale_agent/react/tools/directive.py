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

"""Directive tool — record what the USER ordered into the Do / Do NOT block.

WHY THIS TOOL EXISTS AND WHY IT IS NOT memory_write
---------------------------------------------------
The user's orders kept being violated while memory entries stating them
existed. Measured examples: "edit the source tree, not the deploy tree" (the
agent had the rule in memory and still edited the deploy copy, repeatedly);
"run only the neighbouring regression tests" (memory said adjacent-suite,
the agent kept running the full suite). The mechanism is not forgetting — it
is COMPETITION: a memory entry is the agent's own finding, so at decision
time the agent's judgement can overrule it. A user order must not be
overrulable, so it (a) carries the user's verbatim words and (b) is rendered
in the system prompt every turn instead of being retrieved on demand.

The extraction rule, which the schema states explicitly: only record what the
USER said. A rule the agent invented, or inferred from a failure, belongs in
memory — putting it here would launder the agent's own judgement into an
order from the user, which is exactly the confusion this block must not have.
"""

from flagscale_agent.react.tools.base import Tool


class DirectiveTool(Tool):
    name = "directive"
    description = (
        "Record, update, or remove a DIRECTIVE — a rule the USER stated about HOW "
        "to work (not a finding about the world).\n\n"
        "Directives render in the system prompt on EVERY turn (the 'Directives "
        "from the USER' block), so they stay in front of you at decision time "
        "instead of needing to be looked up. They outlive plans and sessions.\n\n"
        "USE THIS, NOT memory_write, when:\n"
        "- the user told you what to do / not do, and corrective in tone "
        "('don't do X', 'always Y', 'this is wrong, do Z instead');\n"
        "- the rule would still be true for a completely different task "
        "(it is about your working habits, not about this task);\n"
        "- you broke it more than once — repetition is the signal that the "
        "instruction needs to be resident, not merely retrievable.\n\n"
        "DO NOT use it for: your own conclusions, task state, one-off "
        "instructions for the current task only, or anything the user did not "
        "say. Training yourself through this channel corrupts its authority.\n\n"
        "`quote` is REQUIRED in spirit: put the user's own words there verbatim. "
        "An entry without the user's words cannot be audited by the user at "
        "review time, and unauditable orders get deleted."
    )
    def __init__(self, session_dir=""):
        # Session-scoped, like conversation.json: the dashboard is rebuilt per
        # session, so a directive recorded here must never reach another one.
        #
        # A CALLABLE is accepted because the agent REBINDS its session dir when
        # a session is resumed (agent.py: `self._session_dir = session_dir`).
        # A frozen string would keep writing to the pre-resume session, which
        # would look like it worked while the dashboard read a different file.
        self._session_dir = session_dir

    def _dir(self) -> str:
        d = self._session_dir
        return d() if callable(d) else (d or "")

    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "remove"],
                "description": (
                    "'add' (default) records or updates a directive; 'list' shows "
                    "the current block; 'remove' deletes one by rule text."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["do", "dont"],
                "description": (
                    "'do' = the user told you something to DO. 'dont' = the user "
                    "told you something NOT to do."
                ),
            },
            "rule": {
                "type": "string",
                "description": (
                    "The rule, imperative and short (<=240 chars), in the language "
                    "the user used. Say the action, not a virtue: 'edit source "
                    "tree, never the deploy copy' — not 'be careful about trees'."
                ),
            },
            "quote": {
                "type": "string",
                "description": (
                    "The user's own words, verbatim (<=300 chars). This is the "
                    "provenance proof that the order came from the user."
                ),
            },
            "why": {
                "type": "string",
                "description": (
                    "One short consequence clause (<=200 chars) — what goes wrong "
                    "when the rule is broken. Optional but strongly preferred: "
                    "a rule with its consequence survives review."
                ),
            },
        },
        "required": [],
    }

    def execute(self, **kwargs) -> str:
        from flagscale_agent.react import directives as store

        session_dir = self._dir()
        if not session_dir:
            return "ERROR: no session directory — directives are session-scoped."
        action = str(kwargs.get("action") or "add").strip().lower()
        rule = str(kwargs.get("rule") or "").strip()

        if action == "list":
            block = store.render_directives_block(session_dir)
            return block or "(no directives recorded yet)"

        if action == "remove":
            if not rule:
                return "ERROR: 'remove' needs the `rule` text to match."
            removed = store.remove_directive(session_dir, rule)
            return (
                f"Removed directive: {rule!r}"
                if removed
                else f"No directive matched {rule!r}. Use action='list' to see them."
            )

        if not rule:
            return (
                "ERROR: 'add' needs `rule` (the imperative rule) and `quote` "
                "(the user's verbatim words)."
            )
        quote = str(kwargs.get("quote") or "").strip()
        if not quote:
            return (
                "ERROR: `quote` is required — put the user's own words here "
                "verbatim. A directive without the user's words cannot be "
                "audited, and unauditable orders are rejected at review."
            )
        kind = store.normalize_kind(str(kwargs.get("kind") or "do"))
        rec = store.upsert_directive(
            session_dir, kind=kind, rule=rule, quote=quote,
            why=str(kwargs.get("why") or ""),
        )
        label = "DO" if rec["kind"] == "do" else "DO NOT"
        return (
            f"Recorded ({label}): {rec['rule']}\n"
            f"  user's words: {rec['quote']}\n"
            f"  total directives: {len(store.list_directives(session_dir))}\n"
            "This now renders in the system prompt every turn."
        )

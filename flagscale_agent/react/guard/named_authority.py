"""NamedAuthorityPhase: a task that NAMES an external standard owes that
standard a read, and an unreadable standard owes it a systematic route sweep.

THE FAILURE THIS EXISTS FOR (measured, 2 trials, same task, reward=0 both):
  The task statement named a vendor standard and said nothing else about what it
  required ("check that the X you design satisfy <vendor>'s requirements"). The
  agent tried the vendor's own site, was refused, tried the same host again in a
  few more shapes, and then treated the REQUIREMENT itself as unreadable — it
  enumerated the standard's sub-requirements from memory, missed the one the
  grader checked, and wrote a sentence dismissing that sub-requirement as "not
  required by task".

  The same fact was reachable from the same machine by a different route, and I
  confirmed that afterwards: the refusal was a statement about ONE ROUTE, not
  about the requirement. See the guard's own docstring in net_awareness.py for
  the route taxonomy; this phase is the START-of-run half of the same lesson.

DESIGN CONSTRAINTS (each one is a mistake this could have made):
  * NO TASK LITERALS. This text is rendered into every model context that trips
    the phase, so it may not name any task, vendor, enzyme, file format, field
    name or value. It names the SHAPE ("the statement names an external
    standard"), never an instance. tests/test_guard_named_authority.py asserts
    that mechanically.
  * NARROW TRIGGER. Most tasks name no external standard. A phase that fires on
    every task pays on all of them to serve a few, so the trigger requires an
    explicit "satisfy / comply with / according to <a named external body>"
    construction in the statement — not merely the word "standard".
  * STRUCTURE, NOT SEMANTICS. Nothing here grades whether the sweep was good. It
    asks that the sweep EXIST and that each route tried was recorded with its
    outcome, so a later reader (or the agent itself) can see which axes remain.
  * NON-OVERRIDABLE. A self-issued exemption is exactly what happened in the
    measured trial: the agent decided the requirement was "not required by task".
"""

from __future__ import annotations

import re

from flagscale_agent.react.guard import GuardContext, GuardVerdict
from flagscale_agent.react.guard import startup as _startup
from flagscale_agent.react.guard.startup import StartupPhase

# ── Trigger: does the statement name an external standard it says to satisfy? ─
# The construction being matched is "the deliverable must satisfy <someone
# else's>" — a vendor, a body, a protocol, a kit, a spec. Deliberately NOT a
# bare "standard"/"spec" noun match: a task can say "use a standard file format"
# and mean nothing external to read.
_AUTHORITY_CUE = re.compile(
    r"(?:satisf\w*|comply|complies|compliance|conform\w*|accord(?:ing)?\s+to|"
    r"as\s+(?:specified|defined|required)\s+(?:by|in)|"
    r"\bper\s+(?:the\s+)?|"
    r"follow\w*\s+the\s+(?:requirements|specification|protocol|guidelines?))"
    r"[^.\n]{0,120}?"
    r"(?:requirements?|specification|spec|protocol|guidelines?|standards?|"
    r"manual|instructions|datasheet|convention)",
    re.I,
)

#: Words that are USUALLY a named party but are too generic to be a target on
#: their own. Kept short and abstract — never a real vendor name (that would be
#: the literal this module forbids).
_GENERIC_PARTIES = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "their", "its",
    "any", "all", "each", "your", "our", "standard", "normal", "usual",
})


#: "as required by <Party>" / "as specified by <Party>" / "per <Party> spec" —
#: the party FOLLOWS the cue and no governing noun is needed after it.
_CUE_BY_PARTY = re.compile(
    r"(?:as\s+(?:required|specified|defined|mandated|prescribed)\s+by|"
    r"in\s+accordance\s+with|"
    r"pursuant\s+to)\s+"
    r"(?P<party>[A-Z][A-Za-z0-9&.\-]{1,}|\b[A-Z]{2,}\b)",
)


def names_external_authority(text: str) -> bool:
    """True when the statement tells the deliverable to satisfy a NAMED outside party.

    Three conditions, any of which is enough:
      (a) the satisfiability construction is present (satisfy/comply/conform/
          according-to/follow-the-requirements-of) AND a possessive or a
          capitalised token sits in the same SENTENCE, so the reader has to go
          somewhere for the answer;
      (b) an "as required by <Party>" / "in accordance with <Party>" form names
          the party directly after the cue;
      (c) an acronym (>=2 caps) appears in either construction.

    Conditions (a)-(c) are what keep this from firing on "satisfy the length
    requirement", where everything needed is already in the statement.
    """
    if not text:
        return False
    for m in _AUTHORITY_CUE.finditer(text):
        # Widen the check to the SENTENCE the cue sits in, not the matched span:
        # the named party can follow the matched nouns ("per the requirements OF
        # <Party>") and a span-bounded check misses exactly that ordering.
        lo = max(0, m.start() - 200)
        hi = min(len(text), m.end() + 200)
        window = text[lo:hi]
        # A possessive ("vendor's requirements") or a capitalised token inside
        # the sentence means an outside party was named.
        if re.search(r"[A-Za-z0-9]-?(?:'s|s')\s+", window):
            return True
        for tok in re.findall(r"\b[A-Z][A-Za-z0-9-]{1,}\b", window):
            if tok.lower() not in _GENERIC_PARTIES:
                return True
        # An acronym (all caps, >=2 chars) is a named party form too.
        if re.search(r"\b[A-Z]{2,}\b", window):
            return True
    # (b) the party directly follows the cue — no governing noun needed.
    for m in _CUE_BY_PARTY.finditer(text):
        party = m.group("party")
        if party.lower() not in _GENERIC_PARTIES:
            return True
    return False


# ── Message (no task literals; abstract shape only) ──────────────────────────
_SWEEP_MESSAGE = """[StartupGuard/NamedAuthority] This statement hands a requirement to an OUTSIDE party — it says the deliverable must satisfy something named, and does not itself write out what that something requires. So the statement is not the whole specification: part of it lives where that party published it, and you will be judged on the parts of it you did not enumerate.

Do this BEFORE the design is fixed, because the sub-requirement you skip is not the one you will notice:

1. ENUMERATE the sub-requirements from a POSITION OF IGNORANCE.
   Write the list of independent things that party's requirement constrains — one
   line each — and treat it as incomplete by construction. A standard is a LIST,
   not one requirement: a fixed element AND an ordering constraint AND a spacing
   rule AND a length bound AND an allowed-character set are five, and a
   deliverable meeting one is graded exactly as if it met none.
   Your memory is biased toward the element most often quoted. The element you
   did not think of is the one most likely to be the failing check.

2. THEN TRY TO READ IT — SWITCHING AXES, NOT RE-ASKING ONE ADDRESS.
   A refusal is a statement about a ROUTE. Before concluding the requirement is
   unreadable, exercise at least three DIFFERENT axes and record each attempt:
     · ADDRESS — mirror / regional variant / different path / raw or plain form
       / API or feed instead of the rendered page.
     · PROVENANCE — search for the FACT, not for the one site you were pointed
       at. The same requirement is usually ALSO published by a party that is not
       the original publisher: an open-access full-text index, a preprint or
       archive, a standards-body mirror, a package that vendors the data, a
       reference implementation that encodes the rule in CODE, the tool's own
       bundled documentation. Search in the wording a PRACTITIONER of the field
       would use — not the wording the statement used, and not a term whose
       dominant meaning in that field is something else.
     · LOCAL — an installed package's source, a mounted cache, the image, or the
       material you were given may already hold it.

3. WRITE THE TABLE, ONE ROW PER SUB-REQUIREMENT:
     <sub-requirement> -> <how you know it: read | inferred> -> <the delivered
     bytes that satisfy it>
   Where a row is "inferred", say so and satisfy the MOST CONSERVATIVE reading.
   A row you cannot fill is not a row you may delete.

4. IF THE SWEEP GENUINELY EXHAUSTS: mark it UNCONSULTED, enumerate both readings
   of every element you could not settle, and satisfy every one you can. An
   unreadable source is a gap to fill conservatively — it is never a licence to
   call a requirement satisfied, and never a reason to downgrade it."""


class NamedAuthorityPhase(StartupPhase):
    """Order 2.4: a named external standard owes a sub-requirement table.

    Fires once, between ResearchPhase (2) and FirstActionPhase (3). Satisfied by
    the FIRST recording act — the phase is about whether the enumeration
    happened at all, not about how many rows it had, because a row count is a
    threshold this has no evidence to set.
    """

    name = "named_authority"
    order = 2.4
    single_shot_only = True

    def __init__(self):
        self._armed = False
        self._satisfied = False
        self._checked_statement = False

    # -- arming -------------------------------------------------------------
    def _arm(self, ctx: GuardContext) -> None:
        """Arm only if the task statement names an outside authority.

        The statement is read from the conversation's first user message, which
        is where the harness puts it. If it cannot be found, the phase stays
        DISARMED: a phase that cannot see the statement must not guess.
        """
        if self._checked_statement:
            return
        self._checked_statement = True
        for msg in (ctx.messages or []):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                self._armed = names_external_authority(content)
                return

    def is_satisfied(self) -> bool:
        return self._satisfied or not self._armed

    # -- gating -------------------------------------------------------------
    def check(self, ctx: GuardContext) -> GuardVerdict | None:
        self._arm(ctx)
        if self.is_satisfied():
            return None
        name = ctx.tool_name
        if not name or name in _startup._META_TOOLS:
            return None
        # A research/sweep act is the way to clear it — never block those.
        if name in _startup._RESEARCH_TOOLS:
            return None
        if name == "shell":
            cmd = str((ctx.tool_args or {}).get("command", ""))
            if _startup._is_network_probe(cmd) or _startup._is_heavy_network_cmd(cmd):
                return None
        return GuardVerdict.block(
            message=_SWEEP_MESSAGE,
            reason="startup_named_authority",
            category="startup_named_authority",
            overridable=False,
        )

    def observe_post(self, ctx: GuardContext) -> None:
        self._arm(ctx)
        if not self._armed or self._satisfied:
            return
        result = ctx.tool_result
        if not isinstance(result, str) or result.startswith("[BLOCKED BY GUARD]"):
            return
        name = ctx.tool_name
        # A plan/notes write IS the recording act this phase asks for, and so is
        # any research/sweep call: either makes the enumeration inspectable.
        if name in _startup._META_TOOLS or name in _startup._RESEARCH_TOOLS:
            self._satisfied = True
            return
        if name == "shell":
            cmd = str((ctx.tool_args or {}).get("command", ""))
            if _startup._is_network_probe(cmd) or _startup._is_heavy_network_cmd(cmd):
                self._satisfied = True

    def accept_override(self, reason: str, ctx: GuardContext) -> bool:
        # NON-OVERRIDABLE: the measured failure was the agent deciding on its own
        # that the requirement did not apply.
        return False

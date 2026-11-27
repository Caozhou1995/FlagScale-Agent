"""Bounds on rule 7 (narrower-reading tie-break) and its named-standard exemption.

Measured failure this pins (dna-assembly, 2026-11-15 batch, reward=0.0):
the task text's last bullet names the vendor explicitly —
  "If you aren't familiar with BsaI-HF v2 make sure to check that the enzyme
   cut-sites you design satisfy NEB's requirements."
The agent web_fetch'd the NEB page, read "add 4-6 bases 5' of the recognition
site", and then wrote verbatim "But the task doesn't require it" and shipped
primers with no 5' clamp. The only failing assertion was
`Primer must have clamp of at least 1 nucleotide before the recognition site`.

Two harness texts pushed it there and both are bounded here:
  • `_READERS_PREAMBLE` rule 7 ("prefer the reading whose CHANGES ARE SMALLER")
    read as "edit fewer bytes", which the missing clamp satisfies.
  • KnowledgeSkillGuard's SPEC-LOYALTY CHECK, which labelled any imported
    convention the text left silent "smuggling" — the agent generalised it to
    "external standards may never decide the deliverable".

The trap on the OTHER side is a rule that says "whatever the kit says wins",
which would license overriding what the statement itself pins. Both bounds are
asserted, plus the ORDER, because order carries the precedence and a substring
test cannot see it.
"""

import pytest


# ── rule 7 bound, in the readings preamble ──────────────────────────────


def _preamble() -> str:
    from flagscale_agent.react.agent import WorkerAgent

    return WorkerAgent._READERS_PREAMBLE


def test_rule7_named_standard_elements_are_not_scope_creep():
    """7(a): what the NAMED standard requires is part of the ask, not creep."""
    p = _preamble()
    assert "WHAT THE NAMED STANDARD REQUIRES IS NOT 'WIDER SCOPE'" in p
    # The reason must be stated as containment, not as a preference — a
    # preference would lose the next tie against a genuinely smaller edit.
    assert "are part of what the statement asked for" in p
    assert "Judge scope by " in p and "never by counting how many bytes" in p


def test_rule7_omission_is_not_the_smaller_change():
    """An answer missing a required element must not win by being smaller."""
    p = _preamble()
    assert "is not 'the smaller change'; it is an incomplete answer" in p


def test_rule7_exemption_is_bounded_to_named_standards():
    """7(b): the narrow reading still wins where the standard is silent.

    Without this bound the exemption becomes "external docs always win", which
    re-opens the smuggling hole SPEC-LOYALTY exists to close.
    """
    p = _preamble()
    assert "only wins where the standard is genuinely " in p
    assert "silent" in p
    assert "following it IS the literal statement, not " in p


def test_rule7_bound_comes_after_the_tiebreak_it_qualifies():
    """ORDER: the bound must qualify rule 7, not precede it as a new rule.

    A substring test accepts either placement; only the index comparison sees
    that the exemption rides on the tie-break it is bounding.
    """
    p = _preamble()
    i_tiebreak = p.index("CHANGES ARE SMALLER AND REVERSIBLE")
    i_bound_a = p.index("WHAT THE NAMED STANDARD REQUIRES IS NOT")
    i_bound_b = p.index("The narrower reading only wins where the standard is genuinely")
    assert i_tiebreak < i_bound_a < i_bound_b
    # And it must stay inside rule 7 — before the next numbered rule would begin.
    assert "8. " not in p[i_tiebreak:i_bound_b]


# ── SPEC-LOYALTY exemption, in the guard ────────────────────────────────


def _loyalty_text() -> str:
    """Extract the SPEC-LOYALTY block from the guard's source.

    Read from the module source rather than by calling the guard: the text is
    assembled at call time and a stub ctx would only re-assert the stub.
    """
    import inspect

    from flagscale_agent.react.guard import knowledge_skill as ks

    src = inspect.getsource(ks)
    i = src.index("SPEC-LOYALTY CHECK")
    j = src.index("If the fetch merely confirmed prior knowledge", i)
    return src[i:j]


def test_spec_loyalty_exempts_standards_the_task_names():
    t = _loyalty_text()
    assert "EXEMPTION — NOT SILENT IS NOT SMUGGLING" in t
    assert "If the task text NAMES the " in t
    # The decisive sentence: naming the standard IS the requirement. This is the
    # exact phrasing the dna-assembly run needed — it reasoned the opposite
    # ("the task doesn't require it") and lost the only failing assertion.
    assert "naming the standard IS the " in t


def test_spec_loyalty_still_blocks_smuggling_on_true_silence():
    """The original prohibition must survive — this is a bound, not a repeal."""
    t = _loyalty_text()
    assert "the task text left SILENT, that is " in t
    assert "smuggling" in t
    assert "minimal-uncertainty default" in t


def test_spec_loyalty_default_is_bounded_to_double_silence():
    """minimal-uncertainty default applies only where text AND standard are open."""
    t = _loyalty_text()
    assert "axes the " in t and "BOTH leave open" in t


# ── convergence squeeze budget ─────────────────────────────────────────


def test_squeeze_budget_is_2048():
    """User directive 2026-11-15: squeeze budget 1024 -> 2048.

    128 was measured broken (0/4 write_file, cap-retry 38-50: a stream cut below
    the "think + emit a tool call" unit can never reach the escape branch).
    1024 vs 2048 is Fisher p=0.486 on n=4 — NOT distinguishable, so this is a
    directive, not a measured win. The test pins the value so a later edit
    cannot silently drift it back.
    """
    from flagscale_agent.react.kernel import AgentKernel as Kernel

    assert Kernel.THINKING_SQUEEZE_TOKENS == 2048


def test_squeeze_budget_stays_above_the_deadlock_floor():
    """2048 must remain well above the measured-broken 128, not merely != 128."""
    from flagscale_agent.react.kernel import AgentKernel as Kernel

    assert Kernel.THINKING_SQUEEZE_TOKENS >= 1024


def test_every_squeeze_site_uses_the_constant_not_a_literal():
    """No site may hard-code its own floor — that is how 1024 got scattered.

    A site is compliant if the value it assigns is the constant itself OR a
    helper whose returns are all floored at that constant. The rule is about the
    FLOOR, not about the syntax of the assignment, so the ladder helper
    (_next_squeeze_budget) counts as compliant: it returns
    `max(THINKING_SQUEEZE_TOKENS, ...)` / the constant directly, and its own
    test asserts the floor holds for every arm.
    """
    import inspect

    from flagscale_agent.react import kernel as k

    src = inspect.getsource(k)
    # Assignments to the override must come from the constant or a floored helper.
    assigns = [
        ln.strip()
        for ln in src.splitlines()
        if "_thinking_cap_override = " in ln and "None" not in ln
    ]
    assert assigns, "expected at least one squeeze assignment"

    helper = "_next_squeeze_budget"
    # The helper must itself be floored at the constant, or the exemption is a
    # hole: a helper that returned an arbitrary int would let a literal in.
    # Behavioural, not syntactic — a source grep for the constant name still
    # passes when only ONE of several returns is floored, which is exactly the
    # hole this guard exists to close.
    from types import SimpleNamespace

    floor = k.AgentKernel.THINKING_SQUEEZE_TOKENS
    fn = k.AgentKernel._next_squeeze_budget
    for base in (0, 1024, 4096, 65536):
        stub = SimpleNamespace(
            THINKING_SQUEEZE_TOKENS=floor,
            THINKING_SQUEEZE_START_FROM_CONFIG=True,
            _squeeze_steps=0,
            deps=SimpleNamespace(config=SimpleNamespace(thinking_budget=base)),
        )
        for arm in range(8):
            v = fn(stub)
            assert v >= floor, (
                f"arm {arm} with base={base} returned {v}, below the "
                f"{helper} floor {floor} — the helper is not floored at the "
                f"constant")

    for ln in assigns:
        assert ("THINKING_SQUEEZE_TOKENS" in ln) or (helper in ln), \
            f"literal squeeze floor: {ln}"

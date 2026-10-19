"""M14 — every failing requirement is reported in ONE message.

Design principle (user-ratified): guards must not cascade — surfacing one block,
waiting for the fix, then surfacing the next. A cascade costs one full round trip
PER violation and hides the remaining violations until the agent has already spent
a corrective edit on the first. All guards that fire at a decision point must be
reported together.

Two structural layers are covered:
  (a) registry — several guards each returning one block;
  (b) guard-internal — ONE guard whose check_pre short-circuits at its first
      failing sub-rule. `Guard.check_all_pre` is the collector API; the registry
      merges every verdict it returns.

Falsifiers below fail if the merge is reverted to top-1 presentation.
"""
from flagscale_agent.react.guard import (
    Guard,
    GuardContext,
    GuardRegistry,
    GuardVerdict,
)


def _ctx(**kw):
    return GuardContext(tool_name="shell", tool_args={"command": "ls"}, **kw)


class _Always(Guard):
    """A guard that always returns one block with a distinct, inspectable text."""

    def __init__(self, name, reason, tag, priority=50, overridable=True):
        self.name = name
        self.priority = priority
        self._reason = reason
        self._tag = tag
        self._overridable = overridable

    def check_pre(self, ctx):
        return GuardVerdict.block(
            f"<<{self._tag}>>", reason=self._reason, category=self._tag,
            overridable=self._overridable,
        )


class _Multi(Guard):
    """A guard with THREE independent sub-rules that all fail.

    check_pre short-circuits (the defect class); check_all_pre collects all three.
    """

    name = "multi_rule"
    priority = 40

    def check_pre(self, ctx):
        return GuardVerdict.block("<<first_rule>>", reason="rule_one",
                                  category="multi")

    def check_all_pre(self, ctx):
        return [
            GuardVerdict.block("<<first_rule>>", reason="rule_one", category="multi"),
            GuardVerdict.block("<<second_rule>>", reason="rule_two", category="multi"),
            GuardVerdict.block("<<third_rule>>", reason="rule_three", category="multi"),
        ]


# ── (a) registry layer: cross-guard cascade ──────────────────────────────

def test_three_guards_produce_one_message_naming_all_three():
    reg = GuardRegistry()
    reg.register(_Always("g_one", "reason_one", "block_one", priority=10))
    reg.register(_Always("g_two", "reason_two", "block_two", priority=20))
    reg.register(_Always("g_three", "reason_three", "block_three", priority=30))

    v = reg.check_pre(_ctx())
    assert v is not None and v.action == "block"
    # ONE verdict reaches the agent, and it carries every guard's own message
    # and slug — not just the top-ranked one.
    for tag in ("block_one", "block_two", "block_three"):
        assert f"<<{tag}>>" in v.message, f"{tag} missing from the merged message"
    for reason in ("reason_one", "reason_two", "reason_three"):
        assert reason in v.message, f"[failing check: {reason}] missing"


def test_merged_message_states_how_many_failed():
    reg = GuardRegistry()
    reg.register(_Always("g_one", "reason_one", "b1", priority=10))
    reg.register(_Always("g_two", "reason_two", "b2", priority=20))
    v = reg.check_pre(_ctx())
    assert "2 independent requirements failed" in v.message


def test_single_violation_is_not_wrapped_in_a_multi_header():
    """The merge must not add noise when there is only one failing requirement."""
    reg = GuardRegistry()
    reg.register(_Always("g_only", "reason_only", "solo", priority=10))
    v = reg.check_pre(_ctx())
    assert "<<solo>>" in v.message
    assert "independent requirements failed" not in v.message


def test_lead_block_is_still_the_hardest():
    """Merging must not disturb severity ranking: the hardest block leads."""
    reg = GuardRegistry()
    reg.register(_Always("soft", "soft_reason", "soft_tag", priority=10))
    reg.register(_Always("hard", "hard_reason", "hard_tag", priority=20,
                         overridable=False))
    v = reg.check_pre(_ctx())
    assert v.guard_name == "hard", "non-overridable block must still lead"
    assert v.overridable is False
    assert "<<soft_tag>>" in v.message, "the softer block must still be reported"


def test_escalate_leads_and_others_are_merged():
    class _Esc(Guard):
        name = "stop"
        priority = 60

        def check_pre(self, ctx):
            return GuardVerdict.escalate("<<escalate_tag>>", reason="stop_reason",
                                         category="stop")

    reg = GuardRegistry()
    reg.register(_Always("soft", "soft_reason", "soft_tag", priority=10))
    reg.register(_Esc())
    v = reg.check_pre(_ctx())
    assert v.action == "escalate"
    assert "<<soft_tag>>" in v.message


# ── (b) guard-internal layer: one guard, many sub-rules ──────────────────

def test_one_guard_reports_all_its_sub_rules():
    reg = GuardRegistry()
    reg.register(_Multi())
    v = reg.check_pre(_ctx())
    for tag in ("first_rule", "second_rule", "third_rule"):
        assert f"<<{tag}>>" in v.message, f"{tag} missing — guard still short-circuits"
    assert "independent requirements failed" in v.message


def test_default_collector_wraps_check_pre_for_single_rule_guards():
    """A guard that only implements check_pre needs no change (default wrapper)."""
    g = _Always("single", "single_reason", "single_tag")
    out = g.check_all_pre(_ctx())
    assert len(out) == 1 and out[0].reason == "single_reason"
    # And a silent guard yields nothing.
    class _Silent(Guard):
        name = "silent"
        def check_pre(self, ctx):
            return None
    assert _Silent().check_all_pre(_ctx()) == []


# ── falsifiers: reverting either layer must turn a test red ──────────────

def test_falsifier_registry_top1_reverts_to_cascade():
    """Simulate the OLD registry (present only the first block) and show the
    merged-message assertions above would fail."""
    reg = GuardRegistry()
    reg.register(_Always("g_one", "reason_one", "block_one", priority=10))
    reg.register(_Always("g_two", "reason_two", "block_two", priority=20))
    v = reg.check_pre(_ctx())
    # What top-1 presentation would have produced:
    top1_only = v.message.split("=" * 62)[0]
    assert "<<block_two>>" not in top1_only
    assert "<<block_two>>" in v.message, "merge is what makes block_two visible"


def test_falsifier_internal_shortcircuit_hides_sub_rules():
    """The collector, not check_pre, is what surfaces sub-rules 2 and 3."""
    first = _Multi().check_pre(_ctx())
    assert "<<second_rule>>" not in first.message, \
        "check_pre alone only knows the first sub-rule — collection is load-bearing"
    allv = _Multi().check_all_pre(_ctx())
    assert len(allv) == 3

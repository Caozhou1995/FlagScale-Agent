"""Pin the E10 clause (a self-authored EXPLANATION does not close a lead).

Why this clause exists — two measured losses, both in memory as
insight/tbench/self_authored_forbidden_overrides_prompt:

  1. 2026-11-28__16-25-04 / gpt2-codegolf__DGLPFXH. The agent passed the
     5000-byte gate, then its program printed a constant "croftcroft...". It
     wrote "If this checkpoint is in fact meant to hold real GPT-2 weights, the
     parameter ordering ... is the thing to re-examine first" — and never
     examined it, because it had already concluded "the checkpoint is synthetic,
     so this is expected". The story closed the lead its own author opened.

  2. 2026-11-26 dna-assembly. The correct 5'-clamp recipe was in its own notes,
     judged harmless, then shipped WITHOUT the clamp, justified as "NEB's own
     tool does exactly this".

E9 already covers the prohibition half ("your own unverified guess forbids
nothing — it asks for a LABEL"); E10 covers the explanation half.

Two ablation directions are pinned here:
  (A) delete E10            -> the presence tests go red
  (B) delete E9's sentence  -> the E9-preservation tests go red
Without (B), a future edit could satisfy E10 by overwriting E9 and still pass.
"""

import pytest

from flagscale_agent.react import prompt as p


@pytest.fixture()
def static_text() -> str:
    return p.SYSTEM_PROMPT_STATIC


class TestClausePresent:
    def test_header_is_present_exactly_once(self, static_text):
        assert static_text.count(
            "A SELF-AUTHORED EXPLANATION DOES NOT CLOSE A LEAD") == 1

    def test_carries_the_verdict_word(self, static_text):
        body = static_text[static_text.find("E10 "):]
        # The clause must say what the story IS, not merely that one exists.
        assert "HYPOTHESIS" in body[:400], "E10 lost its 'this is a hypothesis' half"

    def test_carries_the_required_action(self, static_text):
        body = static_text[static_text.find("E10 "):][:400]
        assert "verify it" in body, "E10 lost the required action"
        assert "keep the lead open" in body, "E10 lost the alternative disposition"

    def test_scoped_against_e9_so_it_is_not_a_duplicate(self, static_text):
        """E10 must say WHICH half it covers, or a reader cannot tell the two
        apart and the clause reads as a restatement (the failure the proposal
        existed to avoid)."""
        body = static_text[static_text.find("E10 "):][:400]
        assert "E9 covers prohibitions" in body
        assert "this covers explanations" in body


class TestE9Survives:
    """E10 must be an ADDITION. These are the exact sentences E9's own test file
    pins; if E10 were built by rewriting E9 they would disappear."""

    def test_prohibition_sentence_intact(self, static_text):
        assert ("A self-authored prohibition is subordinate to the task text "
                "and its authorities") in static_text

    def test_forbids_nothing_half_intact(self, static_text):
        assert "your own unverified guess forbids nothing" in static_text

    def test_label_half_intact(self, static_text):
        assert "it asks for a LABEL" in static_text

    def test_e9_header_intact(self, static_text):
        assert "E9 AN UNREAD STANDARD IS NOT A PROBABILITY" in static_text


class TestOrdering:
    def test_e10_comes_after_e9(self, static_text):
        i9 = static_text.find("E9 AN UNREAD STANDARD")
        i10 = static_text.find("E10 A SELF-AUTHORED EXPLANATION")
        assert i9 != -1 and i10 != -1
        assert i9 < i10, "E10 must follow E9 (it explicitly extends E9)"

    def test_e10_precedes_the_ranks_coda(self, static_text):
        """'Each rank exists...' is the closing line of the rank table; a clause
        appended after it would read as outside the ladder."""
        i10 = static_text.find("E10 A SELF-AUTHORED EXPLANATION")
        i_coda = static_text.find("Each rank exists because ignoring it lost a run.")
        assert i10 < i_coda


class TestBudget:
    def test_prompt_fits_its_budget(self):
        n, b = p.prompt_len(), p.PROMPT_BUDGET
        assert n <= b, f"prompt over budget: {n} > {b}"

    def test_budget_is_the_authorized_value(self):
        # The user authorized exactly +200 over the previous 51_500. Pinning the
        # number keeps a later edit from quietly raising it again.
        assert p.PROMPT_BUDGET == 51_700, (
            f"budget changed to {p.PROMPT_BUDGET}; the authorized value is 51_700")

    def test_e10_cost_is_bounded(self):
        """E10 itself must stay inside the authorized slack (~214 chars). A
        clause that needs more must be argued for, not slipped in."""
        line = [ln for ln in p.SYSTEM_PROMPT_STATIC.split("\n")
                if ln.startswith("E10 ")][0]
        assert len(line) <= 214, f"E10 grew to {len(line)} chars (slack was 214)"


# ─────────────────────────────────────────────────────────────────────────────
# PROVENANCE — the two measured cases that motivated E10.
#
# These lived briefly in prompt.py and were rejected by TWO independent gates,
# both measured red:
#   1. budget — inside the triple-quoted literal, "#" lines are TEXT, not
#      comments: prompt_len jumped 51_699 -> 53_172.
#   2. task-agnostic — test_prompt_scientific_protocol reads prompt.py SOURCE,
#      not just the string, and black-lists domain nouns; even a real python
#      comment there leaks the instance name.
# The evidence therefore belongs with the test that pins the rule.
#
# Case 1 — gpt2-codegolf, 2026-11-28. With the artifact-size gate already
#   passed (5425 -> 2736 bytes after the harness printed the size), the program
#   still emitted a constant token. The agent wrote, verbatim: "If this
#   checkpoint is in fact meant to hold real GPT-2 weights, the parameter
#   ordering I derived from the exact 124,439,808-float count is the thing to
#   re-examine first" — and then never examined it, because it had already
#   concluded "the checkpoint is synthetic, so this is expected: the argmax
#   token is a fixed property of the weights." The story closed a lead its own
#   author had just opened. UNVERIFIED: whether that checkpoint really is
#   synthetic was never checked (container reclaimed). Under BOTH readings the
#   error is the same: the lead was skipped, not answered.
#
# Case 2 — dna-assembly, 2026-11-26. The correct 5'-end recipe was already in
#   the agent's own notes; it judged the extra base harmless and shipped
#   without it, justified as "the vendor's own tool does exactly this". Same
#   shape — a story, not a finding.
#
# E9 covers the PROHIBITION half ("your own unverified guess forbids nothing —
# it asks for a LABEL"). E10 adds the EXPLANATION half, which E9's closing
# sentence does not reach: an explanation that makes misbehaviour acceptable.

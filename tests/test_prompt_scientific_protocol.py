"""The loop protocol (S1-S5) must stay task-agnostic AND stay executable.

The prompt moved from the old S1-S4 doctrine to the 10-section THE LOOP
architecture; this suite was re-pointed at the CURRENT wording (owner ruling:
production code is authoritative, tests adapt). The two pinned failure modes
are unchanged.

1. LEAKAGE. A black-list only fires on words someone remembered to list, and a
   story told in specific measured values describes ONE instance, not a
   principle. So leakage is checked in BOTH directions:
     (a) no domain noun appears (black-list), and
     (b) no bare numeric constant appears (white-list of *numbers allowed*).
   Numbers the rule legitimately needs are written in words rather than digits,
   so the digit scan stays clean and any future digit is a deliberate act.

2. SOFTENING. A protocol that is only "be careful" satisfies any text check
   while teaching nothing. Each loop step must retain its OBLIGATION in the
   prompt bytes -- every assert below targets wording that IS present, and goes
   RED if that clause is deleted.
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"

# Read the SOURCE, not an imported symbol: the clause lives in the prompt file and
# must be checkable without importing the package (no runtime deps, no config load).
RAW = re.sub(r"\s+", " ", SRC.read_text(encoding="utf-8"))
FLAT = RAW.lower()


def has_word(term: str) -> bool:
    """Word-boundary containment.

    Substring matching is the wrong test for short domain nouns: 'ligation' is a
    substring of 'obligation', 'gene' of 'degenerate'. Using a bare `in` here
    produces false leaks and then pressures a future editor to REMOVE a correct
    English word instead of keeping the clause honest.
    """
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", FLAT) is not None

# Domain nouns from the instances that motivated the clause. Kept, not removed:
# these are the words a future edit is most likely to let slip back in.
BANNED_DOMAIN_NOUNS = (
    "oligotm", "primer3", "primer", "bsai", "ggtctc", "overhang", "anneal",
    "dna-assembly", "plasmid", "circular", "fasta", "ligation", "ligase",
    "melting temperature", "egfp", "nucleotide",
)

# A story told in specific measured values describes ONE instance. The principle
# must survive without the numbers; the numbers are what make it identifiable.
BANNED_NUMERIC_CONSTANTS = (
    "0.21", "0.41", "1.39", "5.41", "73.39", "63.48", "67.72",
)

STEP_RE = re.compile(r"(?m)^(S\d+)\s+[A-Z]")


def test_protocol_section_present():
    """The loop exists as a named block: one method, five steps."""
    assert "the loop" in FLAT
    for tag in ("s1 observe", "s2 hypothesize", "s3 test", "s4 read", "s5 revise"):
        assert tag in FLAT, f"missing loop step: {tag}"


def test_each_step_carries_an_obligation_and_an_output():
    """Each step is numbered and actionable, not scattered advice.

    The numbered 'S<n>' tags are the contract that makes the steps executable: a
    stranger can refer to S3 and ask what it demands. Five steps, five tags.
    """
    steps = STEP_RE.findall(SRC.read_text(encoding="utf-8"))
    assert steps == ["S1", "S2", "S3", "S4", "S5"], f"loop steps, got {steps}"


def test_no_domain_nouns_leak():
    for banned in BANNED_DOMAIN_NOUNS:
        assert not has_word(banned), f"domain noun leaked into the prompt: {banned}"


def test_no_instance_identifying_numbers():
    """Digits are the leak channel a noun black-list misses entirely."""
    for banned in BANNED_NUMERIC_CONSTANTS:
        assert banned not in FLAT, f"instance-identifying constant leaked: {banned}"


def test_tolerance_is_sourced_not_chosen():
    """A tolerance must be sourced from the task, never a constant of the agent's own.

    The old merge clause with its hardcoded tolerance is gone; the surviving
    obligation is that an agent-picked tolerance is a claim about preference,
    not about the target, and an in-bound reading must not be argued away.
    """
    assert "a tolerance you picked, or a limit you remember, is a claim about your preference rather than about the target" in FLAT
    assert "write the limit the task states and the value your own last measurement produced on one line" in FLAT
    assert 'not "within my own tolerance"' in FLAT


def test_values_are_read_off_the_target_not_computed_locally():
    """The specific substitution this rule forbids: reading a value out of your
    own derivation instead of off the delivered bytes."""
    assert "read every shared value off the target by alignment, not locally" in FLAT
    assert "per-component correctness is the seductive error" in FLAT


def test_green_does_not_end_the_enumeration():
    """The obligation that flipped the measured outcome: checks passing is not
    the end of the work, and a green run can still ship the wrong thing."""
    assert "stopping at the first evidence is the commonest way a green run ships the wrong thing" in FLAT
    assert "a green happy path says nothing about a branch it never reached" in FLAT
    assert "would this run still have produced this result if i were wrong" in FLAT


def test_protocol_is_referenced_as_the_core_of_the_section():
    """The loop must be declared the one method, so later sections read as instances."""
    assert "there is one method" in FLAT
    assert "read them as instances, never a second doctrine" in FLAT


def test_evidence_sources_are_ranked():
    """The loop ranks evidence instead of averaging it: task text > recorded
    death > own checker, and an imagined grader weighs zero."""
    for tag in ("e1 the task text's own named authority", "e2 the task text's own wording",
                "e3 a recorded death on this same task class", "e4 your own constructed checker"):
        assert tag in FLAT, f"missing evidence rank: {tag}"
    assert "an imagined grader weighs zero" in FLAT
    # the fix that makes the rank actionable: named authority -> runnable check
    assert "turn every such pointer into a runnable check" in FLAT


def test_reading_anchors_on_the_task_verb_not_consumer_habit():
    """Where a phrase admits several readings, the task's own words pin the one.

    The permission/requirement distinction survives as the superset rule: take
    the reading satisfying EVERY constraint in play, never the convenient one.
    """
    assert "take the one the task's verb and nouns pin, not the convenient one" in FLAT
    assert "take the one satisfying the most (the e2 superset rule)" in FLAT
    assert "an element no stated rule forbids and one reading demands is free" in FLAT


def test_measurement_needs_a_could_have_gone_red_check():
    """A measurement the agent did not read off the artifact is a story.

    The three questions (span fed in, axes, could-this-go-red) are the loop's
    answer to a self-consistent but source-less number.
    """
    assert "could this check have gone red" in FLAT
    assert "check against that, never against your own code, which only restates your assumptions" in FLAT
    assert "a difference must name the baseline it was measured against" in FLAT
    # accumulation claim: bounded or streamed, never unmeasured
    assert "bound it" in FLAT and "stream it" in FLAT


def test_observe_before_derive_bullet_present():
    """The observation-first order is a first-class step: a reading from the
    world precedes the view, and the cheapest refuting experiment runs first.
    (Old 'probe before derive' wording is gone; the concept survives as S1/S3.)"""
    assert "s1 observe" in FLAT
    span = FLAT[FLAT.index("s1 observe"):FLAT.index("s2 hypothesize")]
    assert "get a reading from the world before forming a view" in span
    assert "no reading, not yet work" in span
    assert "run the cheapest experiment that could prove you wrong" in FLAT  # header line 1
    # leak guards: task nouns and invented constants must not ride along
    for noun in ("regex", "chess", "checker.py", "castling", "gpt", "doom"):
        assert noun not in span, f"task noun leaked: {noun}"
    assert not re.search(r"\b\d{3,}\b", span), "numeric anchor leaked"

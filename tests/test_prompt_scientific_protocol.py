"""The scientific protocol (S1-S4) must stay task-agnostic AND stay executable.

Two distinct failure modes are pinned here.

1. LEAKAGE (the old black-list could not catch this). A black-list only fires on
   words someone remembered to list. The measured near-miss: a clause was written
   with "1.39 above a stated maximum of 72" plus "a sub-span rather than the whole",
   which together identify one specific benchmark instance even though not one
   black-listed word appears. So leakage is checked in BOTH directions:
     (a) no domain noun appears (black-list, kept for the known offenders), and
     (b) no bare numeric constant appears (white-list of *numbers allowed*, because
         a story told in specific values is an instance, not a principle).
   Numbers the rule legitimately needs -- tolerance wording, counts of steps --
   are written in words rather than digits, so the digit scan stays clean and any
   future digit is a deliberate act rather than an accident.

2. SOFTENING. A protocol that is only "be careful about definitions" satisfies any
   text check while teaching nothing. Each of the four steps must retain its
   OBLIGATION (a verb that demands an artifact), not merely its topic word.
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
    substring of 'obligation', 'gene' of 'degenerate', 'snap' of 'snapshot'. Using
    a bare `in` here produces false leaks and then pressures a future editor to
    REMOVE a correct English word instead of keeping the clause honest.
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


def test_protocol_section_present():
    """The four-step protocol exists as a named block, not as scattered advice."""
    assert "the scientific protocol" in FLAT
    for tag in ("s1 locate the producer", "s2 enumerate the constructions",
                "s3 merge on arithmetic", "s4 keep enumerating after green"):
        assert tag in FLAT, f"missing protocol step: {tag}"


def test_each_step_carries_an_obligation_and_an_output():
    """Each step must demand a produced artifact, not describe a topic.

    'The output of Sn is ...' is the contract that makes the step executable: a
    stranger can ask to see it. Four steps, four outputs.
    """
    assert FLAT.count("the output of s") == 4, "every protocol step needs a stated output"


def test_no_domain_nouns_leak():
    for banned in BANNED_DOMAIN_NOUNS:
        assert not has_word(banned), f"domain noun leaked into the prompt: {banned}"


def test_no_instance_identifying_numbers():
    """Digits are the leak channel a noun black-list misses entirely."""
    for banned in BANNED_NUMERIC_CONSTANTS:
        assert banned not in FLAT, f"instance-identifying constant leaked: {banned}"


def test_merge_rule_does_not_hardcode_a_tolerance():
    """The tolerance must be sourced from the task, never a constant in the text.

    Owner's ruling: tolerances differ per task, so the clause states where a
    tolerance comes FROM and what to do when the task states none.
    """
    assert "not yours to choose and it is not a constant across tasks" in FLAT
    assert "unresolved" in FLAT
    # Scope the literal scan to the MERGE rule only: the rest of the document
    # legitimately contains budget percentages ("80% of budget"), so a file-wide
    # ban would fail on correct text and teach nothing.
    span = FLAT[FLAT.index("merge only on a number"):FLAT.index("the enumeration is not finished")]
    assert not re.search(r"tolerance\s+(of|is|=)\s+\d", span)
    assert "%" not in span


def test_merge_requires_arithmetic_not_verbal_reasoning():
    """The specific deletion this rule forbids: merging by 'obviously the same'."""
    assert "merge only on a number, never on a hunch" in FLAT
    assert "these two constructions obviously describe the same thing" in FLAT


def test_s4_obligates_enumeration_after_checks_pass():
    """The S4 obligation is the one that flipped the measured outcome."""
    assert "does not end when your checks pass" in FLAT
    assert "what else could be fed to that formula" in FLAT
    # It must be tied to the after-banking moment, not to a generic 'review'.
    assert "already banked" in FLAT or "already at its delivery path" in FLAT


def test_protocol_is_referenced_as_the_core_of_the_section():
    """The four steps must be declared the reusable core, so 2.x reads as instances."""
    assert "reusable core of every clause in this section" in FLAT


import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
RAW = re.sub(r"\s+", " ", SRC.read_text(encoding="utf-8"))
FLAT = RAW.lower()

# Top-level protocol steps: a line starting with "S<n> " (the doctrine).
STEP_RE = re.compile(r"(?m)^\s{2}(S\d+)\s+[A-Z]")
GATE_RE = re.compile(r"(?m)^\s{2}(G\d+)\s+[A-Z]")

STEPS = STEP_RE.findall(SRC.read_text(encoding="utf-8"))
GATES = GATE_RE.findall(SRC.read_text(encoding="utf-8"))


def test_doctrine_has_exactly_four_steps():
    """S1-S4 is the whole doctrine. Nine 'steps' was the ratchet this fixes.

    Measured history: S grew 4 -> 9, one new step per losing batch, each restating
    a 2.x corollary. A doctrine that gains an item per observed loss is no longer a
    doctrine, and the reader cannot tell rule from instance. The count is pinned so
    a future loss cannot quietly add S10.
    """
    assert STEPS == ["S1", "S2", "S3", "S4"], f"doctrine must stay at four steps, got {STEPS}"


def test_delivery_gate_exists_as_five_checks():
    """Everything else lives under ONE gate, as checks that name their parent step."""
    assert "the delivery gate" in FLAT
    assert GATES == ["G1", "G2", "G3", "G4", "G5"], f"gate checks, got {GATES}"
    # Each check declares which doctrine step it applies — otherwise it is a new
    # top-level rule wearing a G label.
    for tag, parent in (
        ("g1 bytes, not model", "s1"),
        ("g2 replay the consumer's construction", "s2"),
        ("g3 a verified value wins over a later story", "s3"),
        ("g4 cover every output class", "s4"),
        ("g5 validate the instrument", "s1"),
    ):
        assert tag in FLAT, f"missing gate check: {tag}"
        span = FLAT[FLAT.index(tag):FLAT.index(tag) + 400]
        assert parent in span, f"{tag} must name its parent step {parent}"


def test_gate_carries_the_containment_rule():
    """The anti-ratchet rule must be stated IN the prompt, not just in tests."""
    assert "does not grow new top-level steps" in FLAT
    assert "doctrine that grows one item per observed loss stops being doctrine" in FLAT


def test_g1_gates_the_delivered_bytes():
    """G1 carries the byte-fact obligations the measured losses lacked.

    Two loss shapes: a model symbol never located on the delivered bytes (the
    consumer found the component missing while every in-model check passed), and a
    stated bound treated as a goal rather than a pre-write gate (the final write was
    never re-measured, and the honest report did not rescue it).
    """
    # (1) bounds become runnable assertions over the DELIVERED file.
    assert "compile every numeric bound" in FLAT
    assert "reads the" in FLAT and "delivered file" in FLAT
    # (2) named components are located on the bytes, with a stated location.
    assert "locate on those bytes" in FLAT
    assert '"it is in my design" is not a location' in FLAT
    # (3) the last write must be followed by a gate run; honesty is not a gate.
    assert "re-run the gate after the last write" in FLAT
    assert "unverified write" in FLAT
    assert "an honest report of an out-of-bound deliverable still fails" in FLAT
    # The gate set is task-derived, never invented constants (anti-leakage).
    assert "comes from the task's own stated constraints" in FLAT
    assert "never from constants you invent" in FLAT
    # (4) the gate is not a wrap-up step — postponed verification dies to the wall.
    assert "the gate is not a wrap-up step" in FLAT
    assert "postpone" in FLAT and "the wall will eat" in FLAT


def test_g4_covers_value_range_not_only_class():
    """The range half is what a shape-correct, value-wrong deliverable dies on."""
    assert "value range the spec names" in FLAT
    assert "outside the range the anchors imply is still wrong" in FLAT

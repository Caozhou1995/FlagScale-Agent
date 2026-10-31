"""Plan-close SECOND-READING obligation — the mechanism for the measured loss.

WHY IT EXISTS (measured, run 2026-10-27__15-08-43, trial dna-assembly, reward=0):
The previous plan-close gate (named arbiter pointed at the delivered file) FIRED
three times in that run and the agent obeyed -- but only formally. The gate
required the command to name the delivered path; the agent rewrote it to

    awk ... /app/primers.fasta | while ...; do a=${s:12}; oligotm -seq "$a"; done

The path appears (gate passed) while ``${s:12}`` is the agent's OWN rule for
cutting the measured substring out of those bytes -- it drops a fixed-length
overhang. Its reading gave 44 (inside the stated max of 45); the consumer's
construction, which includes the overhang-matched characters, gave 48 (outside).
Reward 0.

WHY ONE READING CANNOT CATCH IT: a single definition always agrees with itself.
The failure is not in the number, it is in WHICH bytes were measured -- so the
only observable that can reveal it is a second, genuinely different extraction
over the SAME delivered file.

The gate therefore requires the second reading and refuses a copy: pasting the
same command twice measures nothing. It does NOT judge which reading is right --
that is the consumer's construction, not something the harness can see.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_plan_complete_named_arbiter import (  # noqa: E402
    CORPUS,
    ORACLE,
    _ctx,
    _guard,
    _plan,
)

from flagscale_agent.react.oracle import (  # noqa: E402
    rival_reading_distinct,
    rival_reading_obligation,
)

# The recorded command shape from the admitted run: names the delivered file,
# cuts the substring itself.
CUT_12 = "awk ... primers.fasta | while ... do a=${s:12}; oligotm -seq \"$a\"; done"
CUT_8 = "awk ... primers.fasta | while ... do a=${s:8}; oligotm -seq \"$a\"; done"
CORPUS_BOTH = CORPUS + "44\n48\n"


def _run(act, corpus=CORPUS_BOTH):
    g = _guard(_plan())
    g._oracle_bound = dict(ORACLE)
    g._arbiter_act_seen = act
    return [v.reason for v in
            g._plan_complete_obligations(_ctx({"action": "complete"}, corpus=corpus))]


# ---------------------------------------------------------------- pure helpers

def test_one_reading_is_unmet_not_declared():
    assert rival_reading_obligation({"command": CUT_12, "observed": "44"}) == ""


def test_blank_rival_is_unmet_not_declared():
    act = {"command": CUT_12, "observed": "44", "rival_command": "   "}
    assert rival_reading_obligation(act) == ""


def test_explicit_none_is_a_declaration_not_an_obligation():
    act = {"command": CUT_12, "observed": "44", "rival_command": "none"}
    assert rival_reading_obligation(act) is None
    act["rival_command"] = "  NONE "
    assert rival_reading_obligation(act) is None


def test_rival_dict_carries_the_second_command_and_its_own_output():
    act = {"command": CUT_12, "observed": "44",
           "rival_command": CUT_8, "rival_observed": "48"}
    rival = rival_reading_obligation(act)
    assert rival == {"command": CUT_8, "observed": "48"}


def test_a_copy_is_not_a_rival():
    act = {"command": CUT_12, "observed": "44",
           "rival_command": CUT_12, "rival_observed": "44"}
    assert not rival_reading_distinct(act, rival_reading_obligation(act))


def test_whitespace_only_difference_is_still_a_copy():
    act = {"command": CUT_12, "observed": "44",
           "rival_command": "  " + CUT_12.replace(" ", "  "), "rival_observed": "44"}
    assert not rival_reading_distinct(act, rival_reading_obligation(act))


def test_a_genuinely_different_cut_is_a_rival():
    act = {"command": CUT_12, "observed": "44",
           "rival_command": CUT_8, "rival_observed": "48"}
    assert rival_reading_distinct(act, rival_reading_obligation(act))


# ------------------------------------------------------------- the gate itself

def test_gate_refuses_a_single_reading():
    """The measured loss: the delivered path is named, so the OLD check passed."""
    assert _run({"command": CUT_12, "observed": "44"}) == [
        "plan_complete_single_reading"]


def test_gate_refuses_the_same_command_twice():
    assert _run({"command": CUT_12, "observed": "44",
                 "rival_command": CUT_12, "rival_observed": "48"}) == [
        "plan_complete_rival_not_distinct"]


def test_gate_accepts_two_genuine_readings():
    assert _run({"command": CUT_12, "observed": "44",
                 "rival_command": CUT_8, "rival_observed": "48"}) == []


def test_gate_accepts_an_explicit_no_second_reading_declaration():
    assert _run({"command": CUT_12, "observed": "44",
                 "rival_command": "none"}) == []


def test_gate_refuses_a_rival_output_the_harness_never_recorded():
    """A second reading is an OBSERVATION, not a claim: its output must be real.

    Same corpus-verified standard the primary reading is held to -- otherwise
    the 'rival' is free text the agent can invent to look thorough.
    """
    assert _run({"command": CUT_12, "observed": "44",
                 "rival_command": CUT_8, "rival_observed": "48"},
                corpus=CORPUS + "44\n") == [
        "plan_complete_rival_not_distinct"]

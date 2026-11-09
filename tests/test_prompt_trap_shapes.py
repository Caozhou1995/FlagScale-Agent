"""Contract for the FIVE TRAP SHAPES teaching block (USER RULING 2026-11-08).

Why this exists
---------------
The 19-task replay batches repeatedly lost points to a SMALL, RECURRING set of
traps that no clause in the prompt named as a shape:

  * a self-check that judged the agent's own artifact (a generator agrees with
    itself) — the reading had to come from an independent path;
  * the LAST format item (one char, an exit token, a header, an exact hash):
    the body right, the score zero;
  * one clean sample standing in for the population;
  * the fix landing in a scratch copy / note instead of the delivered bytes;
  * a visible example that fixes FORM but not scale, offset, unit or index base.

Enumerating *the same five shapes* as a named list, and demanding the agent name
ITS OWN instance of each (or record an open bet), is the smallest change that
makes those traps visible without telling the agent any answer.

Two hard constraints, both tested below:

  1. NO LEAKAGE. The block is pure mechanism: no numbers, no task names, no
     pass bars. A number next to a bar word is exactly what the leak guard in
     tests/test_prompt_no_answer_leak.py forbids, and a task name would hand
     the agent the answer outright.
  2. LOWER BOUND, NOT CHECKLIST. The block must say it is a lower bound and
     demand the sixth shape, otherwise a five-item list CAPS exploration at
     five and the next unseen trap is again unenumerated.
"""

import re

from flagscale_agent.react.guard.system_prompt_remind import _REMIND_MESSAGE
from flagscale_agent.react.prompt import prompt_len, render_prompt


def _prompt():
    return render_prompt()


def test_prompt_names_all_five_trap_shapes():
    low = _prompt().lower()
    assert "five trap shapes" in low, "the named trap-shape list is gone"
    for n in (1, 2, 3, 4, 5):
        assert f"({n})" in low, f"trap shape ({n}) is missing from the prompt"


def test_prompt_marks_the_list_as_a_lower_bound_and_asks_for_the_sixth():
    low = _prompt().lower()
    assert "lower bound" in low, (
        "the trap-shape list must call itself a lower bound; a five-item list "
        "that reads as a checklist caps exploration at five"
    )
    assert "sixth" in low, "the prompt must demand the sixth shape, the one this task adds"


def test_beat_and_prompt_agree_on_the_label_set():
    """The RUNTIME label list in the prompt and the beat's slots must match.

    The beat is what the agent sees on every step_done; the prompt's RUNTIME
    paragraph is the contract that tells the agent which labels to fill. If the
    beat grows a slot (W4) while the contract still names W0..W3, the agent is
    invoiced for a label the prompt never promised and may skip it. This test
    fails in BOTH directions: a slot added on either side without the other.
    """
    low_prompt = _prompt().lower()
    low_beat = _REMIND_MESSAGE.lower()
    runtime_start = low_prompt.index("runtime — the first")
    runtime_end = low_prompt.index("before you ship", runtime_start)
    contract = low_prompt[runtime_start:runtime_end]
    for tag in ("w0", "w1", "w2", "w3", "w4"):
        assert f"{tag} " in contract, (
            f"the prompt's RUNTIME contract does not name label {tag.upper()} — "
            "the beat can demand it but the prompt must promise it"
        )
        assert f"{tag} " in low_beat, f"the beat lost slot {tag.upper()}"
    # W4 must be described as the trap-shape slot, not merely mentioned, and the
    # five shapes it refers to must actually be reachable from the prompt.
    assert "five trap shapes" in contract, (
        "the RUNTIME contract must point W4 at the FIVE TRAP SHAPES block, or the "
        "agent is asked for a slot whose content it cannot find"
    )
    assert "open bet" in contract, (
        "the contract must offer the same 'open bet' escape the beat offers, "
        "otherwise an agent that cannot exclude a shape is pushed to invent one"
    )
    for n in (1, 2, 3, 4, 5):
        assert f"({n})" in low_prompt, f"trap shape ({n}) unreachable from the prompt"


def test_prompt_states_the_division_of_labour_with_before_you_ship():
    """The two adjacent blocks must be distinguishable at a glance.

    BEFORE YOU SHIP is the FORWARD enumeration (how a consumer measures);
    FIVE TRAP SHAPES is the REVERSE enumeration (how you lose anyway). Placed
    back to back with no signpost, a reader cannot tell which to use when.
    """
    low = _prompt().lower()
    assert "before you ship enumerates" in low and "lose anyway" in low, (
        "the FIVE TRAP SHAPES block must open by naming its division of labour "
        "with BEFORE YOU SHIP, or the two adjacent lists read as one"
    )


def test_prompt_trap_shapes_are_pure_mechanism_never_an_answer():
    """No digits and no task names inside the block.

    A digit adjacent to a bar word is the leak shape the guard already bans; a
    task name would be worse.  We scan only the block itself so unrelated
    numbers elsewhere in the prompt (section numbers, char budgets) cannot make
    this test vacuous.
    """
    text = _prompt()
    low = text.lower()
    # locate the BLOCK HEADING, not the first mention: the RUNTIME contract also
    # names "five trap shapes" (it points W4 at this block), so a bare substring
    # search would start the window inside the contract and pick up its digits.
    heading = "five trap shapes — before you ship enumerates"
    start = low.index(heading)
    end = low.index("w3 is a reading", start)
    block = text[start:end]
    assert len(block) > 300, f"trap-shape block looks truncated: {len(block)} chars"
    # the five shapes are NUMBERED (1)..(5); those labels are structure, not
    # content, so strip them before asking whether any digit survives.
    body = block
    for n in (1, 2, 3, 4, 5):
        body = body.replace(f"({n})", "")
    assert not re.search(r"\d", body), (
        "the trap-shape block must carry no digits besides its (1)..(5) labels — "
        "a number here is a leaked threshold the agent could hard-code instead "
        f"of measuring; offending block: {body!r}"
    )
    for name in ("raman", "fasttext", "sam-cell-seg", "gcode", "ocaml", "pov-ray", "doom"):
        assert name not in block.lower(), f"task name {name!r} leaked into the trap-shape block"


def test_prompt_still_fits_the_40000_budget():
    n = prompt_len()
    assert n <= 40_000, f"prompt is {n} chars, budget 40000"
    # prompt_len() must measure the RENDERED template, not the raw constant.
    # It pins the two path placeholders (cwd / src_tree) so the reading is a
    # property of the prose rather than of this host's path lengths; the
    # equality therefore holds against a rendering that uses the SAME pinned
    # placeholders, which is what the original assertion was protecting.
    assert n == len(render_prompt(cwd="<cwd>", src_tree="<src_tree>")), (
        "prompt_len() must measure the rendered prompt"
    )


def test_beat_carries_the_w4_slot_and_fits_its_own_bar():
    """The step_done beat must invoice the same five shapes, per step.

    The prompt states them once at the top level; the beat is what the agent
    actually sees on every step_done, so the shapes must be repeated there as a
    slot it has to fill — otherwise the teaching is paid for once and forgotten.
    """
    msg = _REMIND_MESSAGE
    low = msg.lower()
    assert len(msg) < 2400, f"beat is {len(msg)} chars; bar is 2400"
    assert "w4 trap shapes" in low, "the beat lost its W4 slot"
    assert "open bet" in low, (
        "the beat must offer an explicit 'open bet' escape, or an agent that "
        "cannot exclude a shape will fabricate an answer to satisfy the slot"
    )
    assert "never enumerated" in low, "the beat lost the 'a trap never enumerated' warning"
    # ordering: W4 must come after W3 and before the risk sweep
    assert low.index("w4") > low.index("w3"), "W4 must follow W3"
    assert low.index("w4") < low.index("risk sweep"), "W4 must precede the risk sweep"

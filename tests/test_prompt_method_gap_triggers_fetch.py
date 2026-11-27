"""Pin: a named GAP must trigger an immediate fetch, not a note.

Why this test exists
--------------------
A diagnostic run (gpt2-codegolf, 2026-11-27) showed the agent correctly
identifying its gap — "I need a structural size-reduction method" — and then
writing that gap down as a recommendation instead of fetching the method.
The capability was present the whole time (`web_fetch`, `load_knowledge`);
what was missing was the *trigger*, the moment "I can name the gap but not
the method" fires an action.

The prompt already carried "the standard method for this problem class" and
the standing rule that "no better method exists" is a claim about your own
knowledge. Both describe the destination. Neither said what to do at the
instant the destination cannot be named. These tests pin that link, in BOTH
surfaces that must carry it:

  * the per-turn surface (`prompt.py`) — what the agent rereads every turn;
  * the plan template (`plan_create.py` FRAMING) — what it rereads while
    planning.

A previous change shipped into only one surface and the clause vanished for
the rest of the run (see prop_7b57bc84). So each surface is pinned
separately, and either one alone going missing must fail.

ABLATION (run 2026-11-27, each edit made on a scratch copy):
  * drop the sentence from prompt.py          -> 2 tests red
  * drop the sentence from plan_create FRAMING -> 2 tests red
  * restore both                               -> 6 tests green
"""

import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flagscale_agent.react import prompt as prompt_mod  # noqa: E402
from flagscale_agent.react.tools.plan_create import (  # noqa: E402
    _FRAME_STEP_ACCEPTANCE,
)


def _frame_framing_clause() -> str:
    """The FRAMING acceptance entry (identified by its own opening words)."""
    hits = [c for c in _FRAME_STEP_ACCEPTANCE if c.lstrip().startswith("FRAMING.")]
    assert hits, "FRAMING clause disappeared from _FRAME_STEP_ACCEPTANCE"
    return hits[0]


@pytest.fixture(scope="module")
def turn_surface() -> str:
    """The UP paragraph: the per-turn reading surface."""
    text = prompt_mod.SYSTEM_PROMPT_STATIC
    i = text.find("UP — change the level")
    assert i != -1, "the UP paragraph disappeared from the static prompt"
    return text[i:text.find("\n", i)]


# ----------------------------------------------------------------- the trigger
def test_turn_surface_makes_the_fetch_the_next_action(turn_surface):
    """Naming the gap without the method must BE the fetch, in one reading."""
    low = turn_surface.lower()
    assert "gap named, method not" in low, (
        "the per-turn surface no longer names the state in which fetching is "
        "required: 'gap named, method not'"
    )
    assert "now" in low, "the fetch is not stated as immediate ('NOW')"


def test_turn_surface_names_both_tools_that_exist(turn_surface):
    """A rule that names no mechanism is a wish; name the real tools."""
    low = turn_surface.lower()
    assert "web_fetch" in low, "web_fetch is not named — the agent may not think of it"
    assert "load_knowledge" in low, "load_knowledge is not named"


def test_turn_surface_forbids_filing_it_as_a_note(turn_surface):
    """The exact failure: the gap was filed, not fetched."""
    assert "not a note" in turn_surface.lower(), (
        "the clause dropped its prohibition on filing the gap as a note — that "
        "prohibition is the whole point (a diagnosed gap filed as a note is the "
        "delivery you did not make)"
    )


def test_framing_clause_carries_the_same_trigger():
    clause = _frame_framing_clause().lower()
    assert "gap" in clause and "method" in clause, (
        "FRAMING no longer distinguishes the gap from the method"
    )
    assert "web_fetch" in clause, "FRAMING does not name a fetch mechanism"
    assert "now" in clause, "FRAMING does not make the fetch immediate"


# ------------------------------------------------- the semantics it replaced
def test_existing_claims_still_survive(turn_surface):
    """The merge must PRESERVE the two claims it was folded into."""
    assert "standard method for this problem class" in turn_surface, (
        "the 'standard method / problem class' claim was lost in the merge"
    )
    assert "no better method exists" in turn_surface.lower(), (
        "the 'no better method exists is a claim about your knowledge' rule was lost"
    )
    # Wording here is pinned verbatim by an INDEPENDENT test
    # (tests/test_context.py::TestPromptIntegrity::
    #  test_method_switch_cannot_retarget_easier_goal), so this merge must not
    # rephrase it. Assert the pinned form, not a paraphrase of it.
    assert "change HOW" in turn_surface and "never WHAT the task is" in turn_surface, (
        "the 'change HOW, never WHAT' substitution rule was lost or rephrased; "
        "test_context.py pins these exact words"
    )


def test_budget_ratchet_is_respected():
    """This clause was added by compressing existing wording, not by raising it."""
    n = prompt_mod.prompt_len()
    assert n <= prompt_mod.PROMPT_BUDGET, (
        f"prompt exceeds budget: {n} > {prompt_mod.PROMPT_BUDGET}"
    )

"""The source-tree / working-directory distinction is stated, and can go RED.

Why this file exists (2026-11-09, user ruling): the agent kept conflating two
different trees — the SOURCE TREE that defines the agent (a git repo with
tests) and the DEPLOYMENT TREE the container actually installs from (a copy).
The failure was concrete and repeated: an environment fact (a measured proxy
table) was written INTO the product source as if it were a rule. The user's
ruling was explicit: do not inject environment facts into the code tree; the
agent must carry the DISTINCTION as awareness.

So the check is two-sided on purpose:
  * presence — the rendered prompt names the source tree and states the
    distinction, so the agent has the fact at hand;
  * mutation  — removing either half makes a test here FAIL. A presence-only
    assertion would stay green if the clause were softened into something
    that no longer forbids the actual mistake, which is the failure shape this
    repo has been bitten by before.
"""

from __future__ import annotations

import re

import pytest

from flagscale_agent.react.prompt import (
    PROMPT_BUDGET,
    SYSTEM_PROMPT_STATIC,
    prompt_len,
    render_prompt,
    resolve_src_tree,
)


def rendered() -> str:
    return render_prompt(cwd="/work", src_tree="/src")


def test_header_names_the_source_tree_next_to_the_cwd() -> None:
    """The agent must be able to SEE both trees, not just be told they differ."""
    text = rendered()
    assert "Working directory: /work" in text
    assert "Source tree: /src" in text
    # Order matters: the pair reads as one fact, so the source tree must be
    # adjacent to the working directory, not buried elsewhere in the prompt.
    i_cwd = text.index("Working directory: /work")
    i_src = text.index("Source tree: /src")
    assert 0 < i_src - i_cwd < 120, (
        f"Source tree line is {i_src - i_cwd} chars from the working-directory "
        f"line; it must sit with it as one fact"
    )


def test_distinction_clause_states_what_must_not_happen() -> None:
    """A rule that only says 'they differ' does not forbid the real mistake."""
    text = rendered()
    assert "DIFFERENT THINGS" in text
    # Both halves of the actual failure must be named, not implied.
    assert re.search(r"never edit the source tree to make a task pass", text, re.I)
    # WHERE an environment fact belongs must be named, not just forbidden here.
    # Without this the reader is left asking "then where?" -- and §8 says
    # "write the moment you learn anything durable", which reads as a licence
    # to write it into the nearest tree.
    assert re.search(r"an environment fact goes in memory", text, re.I)


def test_deployment_tree_may_be_a_copy() -> None:
    """The specific confusion: an edit in the repo is not an edit in the copy."""
    assert "installed copy" in rendered()


def test_prompt_len_is_independent_of_path_length() -> None:
    """The budget must measure PROSE, not how long this host's paths are.

    Measured before this fix: 39996 at a 2-char source path, 40154 at a
    160-char one -- same prose, different verdict. A gate that flips with the
    deployment path is not a gate, it is a property of the machine.
    """
    readings = {prompt_len(src_tree=p) for p in ("/a", "/x" * 80,
                                                "/nfs/caozhou/glm/agent_stage3")}
    assert len(readings) == 1, f"prompt_len varies with the path: {readings}"


def test_prompt_len_pins_placeholders_against_caller_values() -> None:
    """A caller passing its own cwd must not change the budget reading."""
    a = prompt_len()
    b = prompt_len(cwd="/" + "y" * 100)
    assert a == b, "prompt_len honoured a caller-supplied cwd; it must pin it"


def test_annotations_are_not_evaluated_at_definition_time() -> None:
    """requires-python is >=3.9; PEP 604 in a signature raises TypeError there.

    Guarded two ways on purpose: `from __future__ import annotations` AND the
    quoted annotation. This asserts the future import is present, because that
    is the one that protects every OTHER annotation added later to this file.
    """
    import flagscale_agent.react.prompt as mod
    src = open(mod.__file__, encoding="utf-8").read()
    assert "from __future__ import annotations" in src, (
        "prompt.py declares requires-python >=3.9 but PEP 604 annotations are "
        "evaluated at definition time without this import"
    )
    # And the annotation really is deferred, not merely accompanied by the line.
    ann = mod.render_prompt.__annotations__.get("src_tree")
    assert isinstance(ann, str), f"src_tree annotation was evaluated: {ann!r}"


def test_prompt_still_within_budget() -> None:
    """Guards the C3 budget: the new clause must not push past PROMPT_BUDGET."""
    n = prompt_len()
    assert n <= PROMPT_BUDGET, f"prompt is {n} chars, budget {PROMPT_BUDGET}"


def test_resolve_src_tree_is_derived_not_baked() -> None:
    """Baking a host path would be silently wrong in the other tree."""
    root = resolve_src_tree()
    # It must be an absolute path to the directory CONTAINING the package,
    # derived from this file's own location.
    assert root.startswith("/")
    assert not root.endswith("flagscale_agent"), (
        f"resolve_src_tree() returned the package dir, not its parent: {root}"
    )
    # And it must be the tree this test is actually running from.
    import os
    # __file__ is <repo>/tests/test_...py, so two dirnames up is the repo root,
    # which is exactly what resolve_src_tree() must return (the PARENT of the
    # flagscale_agent package).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(root) == repo_root


# ── Mutation gate ────────────────────────────────────────────────────────────
# Each case removes one half of the clause from the REAL template and asserts
# the corresponding check above goes red. Written as data so a reader can see
# exactly what was ablated.

_MUTATIONS = [
    ("source-tree header line", "Source tree: {src_tree}\n", ""),
    ("never-edit half",
     "Never edit the source tree to make a task pass; ", ""),
    ("environment-fact half",
     "an environment fact goes in memory, not there.", ""),
    ("deployment-copy sentence",
     "A deployment tree may be an installed copy, not the tree you edit. ", ""),
    ("the distinction itself", "are DIFFERENT THINGS: ", "are related. "),
]


@pytest.mark.parametrize("name,old,new", _MUTATIONS, ids=[m[0] for m in _MUTATIONS])
def test_mutation_makes_a_check_go_red(name: str, old: str, new: str) -> None:
    """Prove each assertion above is load-bearing, not decorative."""
    assert old in SYSTEM_PROMPT_STATIC, f"mutation target missing: {name!r}"
    mutated = SYSTEM_PROMPT_STATIC.replace(old, new, 1)
    text = mutated.format(cwd="/work", src_tree="/src", tools="t",
                          skills="s", knowledge="k")
    with pytest.raises(AssertionError):
        # Re-run the same contract checks against the mutated rendering.
        _contract_checks(text)


def _contract_checks(text: str) -> None:
    assert "Working directory: /work" in text
    assert "Source tree: /src" in text
    assert "DIFFERENT THINGS" in text
    assert re.search(r"never edit the source tree to make a task pass", text, re.I)
    assert re.search(r"an environment fact goes in memory", text, re.I)
    assert "installed copy" in text

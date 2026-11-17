"""rule7(a) granularity: a named standard is a LIST of sub-requirements.

Origin (measured, batch 2026-11-15__21-24-25 / dna-assembly):
The statement's last line names an external vendor standard. The reader block
had already told the agent (rule 7(a)) that satisfying a named standard is not
scope creep, so the agent went and satisfied it — but it satisfied ONE element
(1 nt spacer) and never enumerated the second (a clamp of >= 1 nt 5' of the
recognition site). Measured: 'spacer' reasoned about and written into the design
formula; 'clamp' appeared once in the whole trajectory, inside a heredoc, never
in reasoning. The deliverable's first primer began at the recognition site
(`find(site) == 0`), so the grader's `i >= 1` assertion failed. Official
solution carries BOTH (6 nt clamp AND 1 nt spacer) — they are independent
sub-requirements checked independently.

Second measured layer: the vendor site answered 403 from this environment (a
different site on the same machine answered 200 in the same window), so the
standard could not be read at all. The failure mode to forbid is converting
"That read did not happen" into "that element is not required".

Three properties are pinned, each of which could silently regress:
1. PRESENCE  — the two new clauses are in the RENDERED preamble, not a comment.
2. CONTENT   — they say (a1) enumerate sub-requirements and give bytes for each,
               and (a2) an unreadable standard is filled conservatively.
3. NO LEAK   — no task-specific value leaks in (the point is widening the view,
               not handing over the answer).
"""

import os
import re

os.environ["FLAGSCALE_TASK_READERS"] = "1"

from flagscale_agent.react.agent import WorkerAgent


def _render(n=2, word="two", pass_word="passes", votes="two votes"):
    from flagscale_agent.react.tools.read_task_statement import VIEWPOINTS

    return WorkerAgent._READERS_PREAMBLE.format(
        n=n, nv=len(VIEWPOINTS), word=word, pass_word=pass_word, votes=votes
    )


# ── 1/2. Presence + content of the sub-requirement clauses ───────────────────
def test_rule7a1_standard_is_a_list_of_subrequirements():
    pre = _render()
    assert "A STANDARD IS A LIST OF SUB-REQUIREMENTS, NOT ONE REQUIREMENT." in pre
    # the operative instruction: enumerate, then give bytes for each
    assert "write the sub-requirements out" in pre
    assert "give the bytes that satisfy EACH of them" in pre
    # the measured failure mode stated as the reason, so it is not a slogan
    assert "Satisfying the sub-requirement you happen to remember" in pre
    # the grading fact that makes it matter
    assert "graded exactly as if it met none" in pre
    # and the memory-bias explanation (why the missed one is the missed one)
    assert "biased toward its most frequently cited element" in pre
    # verification must be against bytes, not intention
    assert "against the actual deliverable bytes rather than against your intent" in pre


def test_rule7a2_unreadable_standard_is_not_permission_to_do_less():
    pre = _render()
    assert "IF THE NAMED STANDARD CANNOT BE READ FROM THIS ENVIRONMENT" in pre
    assert "THAT IS NOT PERMISSION TO SATISFY LESS" in pre
    # the requirement survives the failed fetch
    assert "the requirement is still in the statement" in pre
    # the conservative fill, named explicitly
    assert "apply the MOST CONSERVATIVE reading" in pre
    assert "satisfy every sub-requirement you can enumerate" in pre
    # and the two conversions that must NOT happen
    assert "Do not downgrade an unreadable requirement into a satisfied one" in pre
    assert "do not treat 'I could not open the page' as evidence that the " \
           "element is not required" in pre
    # a failed read is a gap, stated once as the rule
    assert "A failed read is a gap you must fill conservatively" in pre


# ── 3. No leak ───────────────────────────────────────────────────────────────
_LEAK_BLACKLIST = [
    # dna-assembly grader/task identity
    "ggtctc", "ggctac", "bsai", "clamp", "primer3", "golden gate",
    "neb.com", "neb' s", "spacer base",
    # raman-fitting values (a different task's answers)
    "1580", "2670", "1e7", "6329",
]


def test_new_clauses_leak_no_task_specific_value():
    pre = _render().lower()
    leaked = [tok for tok in _LEAK_BLACKLIST if tok.lower() in pre]
    assert not leaked, f"rule7(a1)/(a2) leak task-specific values: {leaked}"


def test_new_clauses_name_no_benchmark_or_dataset():
    pre = _render().lower()
    for name in ["terminal-bench", "terminalbench", "benchmark", "dataset",
                 "leaderboard"]:
        assert name not in pre, f"preamble names {name!r}"


# ── 4. The assertion must be able to go red ──────────────────────────────────
def test_presence_check_can_go_red():
    """A presence check that cannot fail is not a check.

    Prove each detector rejects a preamble with the clause removed — the shape
    this file would silently degrade into if someone edited the constant.
    """
    pre = _render()
    removed = pre.replace("A STANDARD IS A LIST OF SUB-REQUIREMENTS, NOT ONE "
                          "REQUIREMENT.", "")
    assert "A STANDARD IS A LIST OF SUB-REQUIREMENTS" in pre
    assert "A STANDARD IS A LIST OF SUB-REQUIREMENTS" not in removed

    removed2 = pre.replace("THAT IS NOT PERMISSION TO SATISFY LESS.", "")
    assert "THAT IS NOT PERMISSION TO SATISFY LESS" in pre
    assert "THAT IS NOT PERMISSION TO SATISFY LESS" not in removed2

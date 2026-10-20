"""The named-tool clause must say the task's OPERATION is the task semantics.

A named library keys its task-typed behaviour off what the task asks you to DO
(retrieve / rank / classify / cluster), not only off a task identifier. The
failure this locks, seen in a zero-leak round: the agent reasoned "the text
never names a task, so the plain encode path applies" while the task text said
"retrieve the document with the 5th highest cosine similarity" -- an operation
whose registry entry decides the parameters.
"""

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC as P


def test_operation_semantics_present():
    assert "keys its task-typed behaviour off the OPERATION" in P


def test_names_the_verbs_not_one_task():
    for verb in ("retrieve the closest", "rank", "classify", "cluster"):
        assert verb in P, verb


def test_rejects_the_missing_name_inference():
    assert "never names a task" in P
    assert "plain path applies" in P


def test_derives_from_deliverable():
    assert "derive the operation from what the deliverable must do" in P


def test_no_task_specific_identifiers_in_clause():
    i = P.index("keys its task-typed behaviour off the OPERATION")
    clause = P[i:i + 600]
    for tok in ("mteb", "bge", "sciFact", "HumanEval", "primers", "stolen_A1"):
        assert tok.lower() not in clause.lower(), tok

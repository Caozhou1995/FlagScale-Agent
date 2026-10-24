"""Regression: the in-environment tie-break key for tied candidate conventions
is INFORMATION CONSERVATION (zero undeclared parameters, fewest
transformations) — never an external accuracy/benchmark/canonical label.

Observed failure this guards against: with two preprocessing conventions
tied 12/12 on every binary probe, an agent broke the tie with an external
semantic-accuracy score (92.07% vs 91.61%) plus a "canonical" label and
shipped the parameter-carrying variant — both numbers being parameters the
task text and every provided artifact never contained.
"""

import pathlib
import re

PROMPT = (
    "/nfs/caozhou/fa_eval/FlagScale-Agent/flagscale_agent/react/prompt.py"
)
_SRC = open(PROMPT).read()

_MUST = [
    # the tie-break key itself
    r"tie-break key is INFORMATION CONSERVATION",
    # the two conservation prongs
    r"ZERO undeclared parameters",
    r"FEWEST transformations",
    # the forbidden keys
    r"external accuracy, benchmark score, or .canonical. label must NEVER",
    # the conservative fallback
    r"axis is undecided",
    r"minimal-uncertainty variant",
]


def test_tie_break_key_is_information_conservation():
    m = re.search(
        r"Imported information does not carry imported decisions\..*?(?=\n\n)",
        _SRC,
        re.S,
    )
    assert m, "the imported-info clause is missing"
    clause = m.group(0)
    for phrase in _MUST:
        assert re.search(phrase, clause), f"missing tie-break phrase: {phrase}"


def test_tie_break_has_no_task_literals():
    clause = re.search(
        r"Imported information does not carry imported decisions\..*?(?=\n\n)",
        _SRC,
        re.S,
    ).group(0)
    for forbidden in (
        "MNIST", "127.5", "0.1307", "0.3081", "127", "255", "prediction.txt",
    ):
        assert forbidden not in clause, f"task literal leaked: {forbidden}"

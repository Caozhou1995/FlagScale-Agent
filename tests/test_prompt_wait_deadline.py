"""G13: A WAIT ARGUMENT IS A DEADLINE, NOT A HINT.

History: this file guarded a wait-deadline clause that the previous prompt.py
carried. RULING: 以现在代码为准，修改单测. In the current bytes the clause is
GONE (grep 0 for 'A WAIT ARGUMENT', 'not a hint', 'one-shot', 'no retry',
'idle'/'silence' settle wording) — the prompt rewrite did not re-home it. Per the
ruling: capability genuinely absent => delete that test function with a
one-line comment; do NOT invent capability and do NOT assert a tautology.

Kept: the no-leak gate — prompt.py must stay free of task-identifying words,
which is a property of the CURRENT file and can still go RED.
"""
import re
from pathlib import Path

PROMPT = Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "prompt.py"
TEXT = PROMPT.read_text(encoding="utf-8")

# Generic harness/engineering terms only -- no task-identifying words.
FORBIDDEN = [
    r"\bterminal-bench\b",
    r"\bheadless-terminal\b",
    r"\bhttp\.server\b",
    r"\bECONNREFUSED\b",
    r"\brequests\.get\b",
    r"\bsend_keystrokes\b",
]


# DELETED (case B, capability absent in current prompt.py):
#   test_g13_deadline_clause_present — the clause 'A WAIT ARGUMENT IS A
#   DEADLINE, NOT A HINT' is absent from the current file (grep 0).
#   test_g13_forbids_idle_heuristic_early_exit — 'not a hint' / 'no output'
#   settle wording is absent from the current file.
#   test_g13_requires_one_shot_consumer_shape_selfcheck — 'one-shot' /
#   'no retry' self-check wording is absent from the current file.
#   test_g13_side_effect_examples_are_generic — 'port listening' / 'flushed' /
#   'spawned' side-effect classes are absent from the current file.


def test_no_task_identifying_words_added():
    body = TEXT.lower()
    for pat in FORBIDDEN:
        assert not re.search(pat, body), f"leak pattern in prompt: {pat}"


def test_deadline_terms_when_present_stay_generic():
    """If deadline/wait wording is ever (re-)introduced, it must stay generic.

    Guards against the old leak class re-entering via a future re-home of the
    G13 clause. Passes today only because the current terms are generic.
    """
    for m in re.finditer(r"deadline", TEXT, re.I):
        ctx = TEXT[max(0, m.start() - 60): m.end() + 60].lower()
        for pat in FORBIDDEN:
            assert not re.search(pat, ctx), f"leak near 'deadline': {pat}"

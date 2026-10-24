"""Anti-decay tests: imported-information spec-loyalty clause.

Pins the two landing sites of the 'imported information does not carry
imported decisions' doctrine:
  1. system prompt (prompt.py) — full clause paragraph;
  2. KnowledgeSkill guard three-state info-gain message (knowledge_skill.py)
     — SPEC-LOYALTY CHECK block, appended to the ACQUIRED/MISSING/SELF-FILLABLE
     states.

Also pins the preserved zero-gain stall sentence (regression guard: the
clause was added by REPLACING the tail of the message once; a later
"cleanup" must not drop the stall sentence again).
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROMPT = REPO / "flagscale_agent" / "react" / "prompt.py"
GUARD = REPO / "flagscale_agent" / "react" / "guard" / "knowledge_skill.py"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_prompt_clause_present():
    src = _norm(PROMPT.read_text())
    assert "Imported information does not carry imported decisions" in src
    # the two branches of the axis, verbatim
    assert "external standard \"answering\" it is a smuggling, not a gain" in src
    assert "keep climbing on your best available proxy" in src


def test_prompt_clause_has_no_task_literals():
    src = _norm(PROMPT.read_text())
    clause = src[src.index("Imported information does not carry imported decisions"):
                src.index("Diagnose a failure from the verifier statistic")]
    for lit in ("MNIST", "normalize", "0.1307", "pytorch", "video", "query",
                "takeoff", "G10", "0.3081"):
        assert lit not in clause, f"task literal leaked into clause: {lit}"


def test_guard_message_carries_spec_loyalty_block():
    src = _norm(GUARD.read_text())
    assert "SPEC-LOYALTY CHECK (apply to every imported fact):" in src
    assert "supplies HOW" in src
    assert "never carries a decision about WHAT the deliverable must be" in src or (
        "it never carries a" in src and "decision about WHAT the deliverable" in src
    )
    assert "re-anchor that axis on the task text" in src


def test_guard_stall_sentence_preserved():
    # regression: the stall sentence must survive clause additions
    src = _norm(GUARD.read_text())
    assert ("If the fetch merely confirmed prior knowledge and added nothing" in src)


def test_three_state_names_intact():
    # the original three states must still be present (the existing
    # test_guard_evict_knowledge assertion continues to hold)
    src = _norm(GUARD.read_text())
    for state in ("ACQUIRED", "MISSING", "SELF-FILLABLE"):
        assert state in src

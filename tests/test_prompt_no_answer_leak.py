"""The shipped prompt text must not leak a benchmark's answer.

WHY THIS FILE EXISTS (2026-11-16, user ruling). The W2 UNIT-FORMULA beat and
the `numeric_conventions` viewpoint were first written with a worked example:

    Write the conversion formula explicitly (e.g. 'x_cm^-1 = 1e7 / x_nm')

That example is verbatim the `solution/solve.sh` line of terminal-bench's
`raman-fitting`:

    # X is in Wavelength (nm), Y is in Intensity (a.u.)
    # Convert X to cm-1
    x = 1e7 / x  # Convert nm to cm-1

So the reminder that existed to make agents CONVERT units was handing one
benchmark's answer to every agent, for every task. The user caught it in one
sentence: "为什么要有示例？不泄题了嘛". Correct: an example drawn from a live
benchmark's reference solution is not an illustration, it is a leak — it
removes exactly the reasoning the task is scoring.

The fix keeps the STRUCTURE and drops the CONTENT. A reminder may say "write
x_B = f(x_A)"; it must not say which f, or hand over the constant that makes
f a one-step lookup. The agent has to derive f from the two unit names.

WHAT IS SCANNED, AND WHY NOT THE WHOLE FILE. The object is the text the model
RECEIVES — string constants. A comment mentioning `raman-fitting` is a
provenance note for maintainers and never reaches the model; flagging it would
force me to delete the very record that explains why the rule exists, so the
detector would be "fixed" by destroying evidence. Extraction is therefore by
AST (`ast.Constant` string values, including the literal fragments of
f-strings and docstrings), not by grepping raw bytes.

This test is deliberately a SIGNATURE test (specific constants and domain
nouns), not a general "no examples" test: generic abstract examples are
legitimate and useful — what is forbidden is material lifted from a task's
reference solution.
"""

import ast
import pathlib
import re

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "flagscale_agent"

# Files whose text reaches the model as prompt/reminder/tool-description.
SHIPPING = [
    "react/prompt.py",
    "react/agent.py",
    "react/kernel.py",
    "react/guard/system_prompt_remind.py",
    "react/guard/find_guard.py",
    "react/guard/memory_discipline.py",
    "react/guard/knowledge_skill.py",
    "react/guard/compile_redirect.py",
    "react/tools/read_task_statement.py",
]

# Signatures of a benchmark's own answer, grouped by the task they come from.
# Each entry: (label, regex). A hit in any model-visible STRING is a leak.
LEAK_SIGNATURES = [
    # raman-fitting: solution/solve.sh converts the axis with this exact
    # constant. Seeing it in a reminder tells the agent the conversion.
    ("raman axis constant", r"1e7\s*/\s*x"),
    # raman-fitting: the two peak identities the solution keys on.
    ("raman peak x0", r"\b1580(\.\d+)?\b"),
    ("raman peak x0 (2D)", r"\b2670(\.\d+)?\b"),
    ("raman domain", r"\braman\b"),
    ("raman domain", r"\bgraphene\b"),
    ("raman unit pair", r"cm\s*\^?-1\s*(vs|/|\u2192|->)\s*nm|nm\s*(vs|/|\u2192|->)\s*cm\s*\^?-1"),
    ("raman fit model", r"\blorentzian\b"),
    # large-scale-text-editing (2026-11-16, third instance of the same mistake):
    # the prompt's "a literal must appear character-for-character" rule was
    # illustrated with ``(`wq` is not `:wq`)``.  `:wq`/`:x` ARE that grader's
    # required tokens (`has_exit = any(line in (":wq", ":x") for ...)`), and
    # `setreg` / `:%normal!` / `@a` are its required command forms.  Illustrating
    # a rule with a live grader's exit token hands the answer to that task.
    ("editor exit token", r"(?<![A-Za-z0-9_]):(wq|x)(?![A-Za-z0-9_])"),
    ("editor register cmd", r"setreg\s*\(\s*['\"][abc]['\"]"),
    ("editor normal cmd", r":%normal!"),
    ("editor register", r"(?<![A-Za-z0-9_])@[abc](?![A-Za-z0-9_])"),
    ("editor delivered file", r"apply_macros"),
]


def model_visible_strings(path):
    """Every string constant in `path`, with the line it starts on.

    Covers plain literals, implicit concatenation, f-string literal parts and
    docstrings — anything a caller could feed to the model.
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append((node.lineno, node.value))
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    out.append((part.lineno, part.value))
    return out


def scan(strings):
    """Return leak hits for a list of (lineno, text). Pure, so it is testable."""
    hits = []
    for lineno, text in strings:
        for label, pat in LEAK_SIGNATURES:
            for m in re.finditer(pat, text, re.I):
                hits.append(f"{label}: line {lineno}: {m.group(0)!r}")
    return hits


def test_the_scanner_can_actually_turn_red():
    """ABLATION for this whole file.

    A leak detector that cannot fail is worse than none: it reports PASS on
    every run and buys false confidence. Feed it the exact bytes that were
    shipped before the fix and require a hit. If this test is green while the
    regexes are broken, the security of the other tests means nothing.
    """
    leaked = [
        (1, "Write the conversion formula explicitly (e.g. 'x_cm^-1 = 1e7 / "
            "x_nm')."),
        (2, "the G peak near 1580 and the 2D peak near 2670 of graphene"),
        (3, "this is a Raman spectrum"),
        (4, "fit a Lorentzian"),
    ]
    hits = scan(leaked)
    assert len(hits) >= 4, (
        "the leak scanner missed material copied from the reference solution "
        f"— it cannot protect anything. hits={hits}"
    )
    # And the abstract replacement that we actually ship must be clean, or the
    # test would force a form nobody can write.
    clean = [
        (1, "Write the conversion you used as a formula in the abstract form "
            "'x_B = f(x_A)', with f spelled out from the definitions of the "
            "two units YOU identified."),
        (2, "unit or scale (e.g. a length vs its reciprocal, a rate vs a period)"),
    ]
    assert scan(clean) == [], f"the abstract form is flagged as a leak: {scan(clean)}"


def test_the_scanner_catches_an_exit_token_leak():
    """ABLATION for the third instance (large-scale-text-editing, 2026-11-16).

    The prompt's literal-matching rule was illustrated with a live grader's
    exit token. Assert the scanner turns red on those bytes, and that the
    abstract replacement we now ship stays clean — otherwise the fix would
    forbid a form that is legitimate to write.
    """
    leaked = [
        (1, "an equivalent form is a FAILED check (`wq` is not `:wq`)"),
        (2, "end the script with :x"),
        (3, "call setreg('a', ...) then :%normal! @a"),
        (4, "Delivered /app/apply_macros.vim"),
    ]
    hits = scan(leaked)
    assert len(hits) >= 3, (
        "the scanner missed a live grader's exit token / command form — the "
        f"third instance of this mistake would ship again. hits={hits}"
    )
    clean = [
        (1, "must appear character-for-character, including any delimiter or "
            "prefix the task itself writes"),
        (2, "an equivalent form is a FAILED check, however equivalent it looks"),
    ]
    assert scan(clean) == [], f"the abstract rule is flagged as a leak: {scan(clean)}"


@pytest.mark.parametrize("rel", SHIPPING)
def test_model_visible_strings_carry_no_benchmark_answer(rel):
    """No string that can reach the model may carry a task's answer signature."""
    path = PKG / rel
    assert path.is_file(), f"shipping file vanished: {path}"
    hits = scan(model_visible_strings(path))
    assert not hits, (
        f"{rel} leaks benchmark-answer material into the prompt:\n  "
        + "\n  ".join(hits)
        + "\n\nReplace the concrete example with the abstract form "
        "('x_B = f(x_A)') — the agent must derive f from the unit names."
    )


def test_unit_formula_rule_survives_without_the_leaked_example():
    """The anti-leak fix must not have deleted the RULE itself.

    Ablation partner of the tests above: it is trivial to make a leak test pass
    by deleting the paragraph. This asserts the structural requirement is still
    present and still abstract — the barrier is 'derive f', not 'no rule'.
    """
    text = (PKG / "react/tools/read_task_statement.py").read_text(encoding="utf-8")
    assert "NOT DISCHARGED BY NAMING IT" in text, (
        "the unit-conversion rule vanished with the example — the fix must "
        "abstract the example, not remove the requirement"
    )
    assert "x_B = f(x_A)" in text, (
        "the abstract conversion form is gone; without it the reminder states "
        "no testable requirement"
    )
    low = text.lower()
    assert "from the definitions of the two" in low, (
        "the rule must require deriving f from the unit definitions — "
        "otherwise 'write a formula' is satisfiable by recalling a constant"
    )


def test_w2_beat_names_the_subslot_without_a_domain_example():
    """The beat's UNIT-FORMULA slot must be abstract too."""
    from flagscale_agent.react.guard.system_prompt_remind import _REMIND_MESSAGE

    assert "UNIT-FORMULA" in _REMIND_MESSAGE
    # The slot names the requirement abstractly; the formula illustration was
    # removed from the beat (an illustration sits one step from an answer).
    assert "the conversion is where the error hides" in _REMIND_MESSAGE
    assert scan([(0, _REMIND_MESSAGE)]) == [], (
        "the step_done beat leaks a benchmark answer into every step's reminder"
    )

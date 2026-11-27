"""Meta-test: make anchor DRIFT visible instead of letting it impersonate a regression.

Why this exists (prop_f2d385af, approved 2026-11-27)
---------------------------------------------------
Two assertions in this repo's prompt tests were red for reasons unrelated to any
code change:

  * ``test_prompt_carrier_silence_e8`` pinned ``PROMPT_BUDGET == 51_000`` after
    the user had already ruled the envelope to 51_500;
  * ``test_prompt_method_gap_triggers_fetch`` pinned ``"standard method for the
    problem class"`` while the shipped text says ``"for this problem class"``.

Both were PRE-EXISTING failures. Their real cost was not redness -- it was that,
on the next change to prompt.py, they were indistinguishable from newly
introduced regressions. I spent a whole round computing a baseline to prove they
were pre-existing, and an unwary reading would instead have "fixed" the product
to satisfy the stale test.

So this file does not grade the prompt. It inventories the string literals that
prompt tests use as anchors into the prompt, and demands that any anchor which no
longer matches be DECLARED. Silent drift becomes a named, dated entry; a
deliberate retirement stays possible.

Three defects this scanner had on its way here, each caught by looking at what it
actually flagged rather than at what it was meant to flag:
  1. every string constant in any Compare/Call -> 130+ false hits
     (``_build_memory_keys_summary``, ``'[<0>] wait_woken'``);
  2. case-sensitive compare -> live anchors reported dead, because the tests
     compare against ``FLAT``/``LOW`` which are lower-cased;
  3. any variable named ``text`` treated as the prompt -> anchors into
     ``read_task_statement.py`` and plan FRAMING were attributed to the prompt.

The fix for (3) is the important one: a name is a prompt surface only if it is
DERIVED from ``SYSTEM_PROMPT_STATIC`` by assignment or function return, tracked
to a fixpoint. That is why ``_frame_ledger_text()`` and a file read are correctly
excluded now.
"""

import ast
import pathlib
import re
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC  # noqa: E402

FLAT = re.sub(r"\s+", " ", SYSTEM_PROMPT_STATIC)
FLAT_LOW = FLAT.lower()

_MIN_ANCHOR = 25
_TEST_GLOB = "test_prompt_*.py"

#: The one true root of the prompt surface.
_ROOT_NAME = "SYSTEM_PROMPT_STATIC"


def _mentions(node, names):
    """True if ``node`` references any name in ``names`` (Name or Attribute)."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in names:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in names:
            return True
    return False


def _bindings(tree):
    """name -> list of assigned value-expressions, in source order."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None:
                continue
            for t in targets:
                if isinstance(t, ast.Name):
                    out.setdefault(t.id, []).append(value)
    return out


def _prompt_derived(tree):
    """Names whose binding is UNAMBIGUOUSLY an expression over SYSTEM_PROMPT_STATIC.

    SCOPE, stated so it can be argued with: this audit only judges anchors that
    compare against the STATIC system prompt.  Three earlier versions of this
    scanner tried to widen the surface set and produced false positives every
    time -- whichever shape they used, the failure was the same: the scanner
    could not tell a prompt surface from a plan-FRAMING surface from a plain
    string, and so reported green tests as drift.  A test that reads plan
    FRAMING is a real drift risk, but it is NOT this file's subject; it is
    guarded by its own A-or-B anchor groups.

    Two rules keep this honest:
      * A name assigned MORE THAN ONCE in the file is excluded -- its uses
        cannot be attributed to one surface (`text` is bound to prompt text in
        one test and to ledger text in another; name-level flow conflates them).
      * A function's return value does NOT make its name a prompt surface.  An
        earlier version did this and labelled _frame_ledger_text() -- which
        returns FRAMING text -- as prompt-derived.
    """
    binds = _bindings(tree)
    derived = {_ROOT_NAME}
    changed = True
    while changed:
        changed = False
        for name, values in binds.items():
            if name in derived or len(values) != 1:
                continue
            if _mentions(values[0], derived):
                derived.add(name)
                changed = True
    return derived


def _or_group(node, derived):
    """If ``node`` is an `or` of surface-membership tests, return their literals.

    ``assert "old wording" in f or "current wording" in f`` is ONE anchor with a
    fallback: the assertion is live as long as either alternative matches.  Flagging
    the retired alternative as drift is the scanner lying about a green test.
    """
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
        return None
    lits = []
    for value in node.values:
        if not isinstance(value, ast.Compare):
            return None
        got = [
            l.value
            for l in [value.left] + list(value.comparators)
            if isinstance(l, ast.Constant) and isinstance(l.value, str) and len(l.value) >= _MIN_ANCHOR
        ]
        if not got or not any(_mentions(c, derived) for c in value.comparators):
            return None
        lits.extend(got)
    return lits


def _anchor_groups(path: pathlib.Path):
    """Lists of literals; a group is live if ANY member matches the prompt."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    derived = _prompt_derived(tree)
    consumed = set()
    for node in ast.walk(tree):
        grp = _or_group(node, derived)
        if not grp:
            continue
        yield ("OR", grp)
        for value in getattr(node, "values", []):
            consumed.add(id(value))
    for lit in _anchors_from(tree, derived, consumed):
        yield ("SINGLE", [lit])


def _anchors_from(tree, derived, consumed=()):
    consumed = consumed or set()
    for node in ast.walk(tree):
        # "<lit>" in SURFACE   (affirmative only: "not in" asserts ABSENCE)
        if isinstance(node, ast.Compare):
            if id(node) in consumed:
                continue
            for op, right in zip(node.ops, node.comparators):
                if not isinstance(op, ast.In):
                    continue
                if not _mentions(right, derived):
                    continue
                left = node.left
                if isinstance(left, ast.Constant) and isinstance(left.value, str):
                    if len(left.value) >= _MIN_ANCHOR:
                        yield left.value
        # SURFACE.find("<lit>") / .index(...) / .count(...)
        if isinstance(node, ast.Call):
            f = node.func
            if not isinstance(f, ast.Attribute):
                continue
            if f.attr not in {"find", "index", "count"}:
                continue
            if not _mentions(f.value, derived):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if len(arg.value) >= _MIN_ANCHOR:
                        yield arg.value


def _anchors(path: pathlib.Path):
    """Every literal this test compares against a prompt surface, flattened.

    Thin wrapper over _anchor_groups() so the control tests can exercise the
    scanner's literal extraction on their own synthetic files.
    """
    for _kind, lits in _anchor_groups(path):
        for lit in lits:
            yield lit


def _stale_report():
    stale = []
    for path in sorted((_ROOT / "tests").glob(_TEST_GLOB)):
        if path.name == pathlib.Path(__file__).name:
            continue
        for _kind, lits in _anchor_groups(path):
            needles = [re.sub(r"\s+", " ", l).lower() for l in lits]
            if not any(FLAT_LOW.count(n) for n in needles):
                stale.append((path.name, " | ".join(lits)))
    return stale


def test_no_undeclared_stale_anchor():
    """Every anchor that no longer matches must be a DECLARED retirement."""
    undeclared = []
    for name, lit in _stale_report():
        body = (_ROOT / "tests" / name).read_text(encoding="utf-8")
        if "STALE ANCHOR" not in body:
            undeclared.append((name, lit[:70]))
    assert not undeclared, (
        "anchors no longer present in the shipped prompt, with no STALE ANCHOR "
        f"declaration in the owning test: {undeclared}"
    )


def test_scanner_can_go_red(tmp_path):
    """Negative control: an injected stale anchor MUST be caught."""
    fake = tmp_path / "test_prompt_injected.py"
    fake.write_text(
        "from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC\n"
        "def test_x():\n"
        "    assert 'a sentence definitely not in the shipped prompt at all' in SYSTEM_PROMPT_STATIC\n",
        encoding="utf-8",
    )
    lits = list(_anchors(fake))
    assert lits, "the scanner found no literals -- it is not scanning"
    assert all(FLAT_LOW.count(re.sub(r"\s+", " ", l).lower()) == 0 for l in lits)


def test_scanner_sees_a_live_anchor(tmp_path):
    """Positive control: a literal that IS in the prompt must not be flagged."""
    live = "never overwrite the green bytes with a candidate"
    assert FLAT.count(live) == 1
    fake = tmp_path / "test_prompt_live.py"
    fake.write_text(
        "from flagscale_agent.react.prompt import SYSTEM_PROMPT_STATIC\n"
        f"def test_x():\n    assert {live!r} in SYSTEM_PROMPT_STATIC\n",
        encoding="utf-8",
    )
    lits = list(_anchors(fake))
    assert lits == [live], lits
    assert FLAT_LOW.count(live.lower()) == 1


def test_scanner_ignores_other_surfaces(tmp_path):
    """A name not derived from the prompt is NOT an anchor into it.

    This is the defect that produced two false "drift" reports: anchors into
    ``read_task_statement.py`` and plan FRAMING share the variable name ``text``.
    """
    fake = tmp_path / "test_prompt_other.py"
    fake.write_text(
        "def test_x():\n"
        "    text = open('some_other_file.txt').read()\n"
        "    assert 'a long sentence that lives in a completely different file' in text\n",
        encoding="utf-8",
    )
    assert list(_anchors(fake)) == []

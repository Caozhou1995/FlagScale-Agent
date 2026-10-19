"""Every GuardVerdict.block() call site must use only its real keywords.

A guard that is ABOUT TO block builds its verdict with
GuardVerdict.block(message, reason, category, overridable). A stray keyword
(e.g. `slug=`, which duplicated `reason`) makes that construction raise
TypeError at the exact moment the guard should have produced a diagnosis: the
agent then sees "Error executing tool: ... unexpected keyword argument", with
none of the missing-constraint content the block existed to deliver. The failure
is silent in the worst way -- tests that never reach that branch stay green while
production degrades to an empty error.

This test inspects every call site, so a new one cannot reintroduce the shape.
"""
import ast
import pathlib

GUARD_DIR = pathlib.Path(__file__).resolve().parents[1] / "flagscale_agent" / "react" / "guard"

# The real signature of GuardVerdict.block.
ALLOWED = {"message", "reason", "category", "overridable"}


def _block_call_kwargs(node):
    """Keyword names passed to a <x>.block(...) call, or None if not one."""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "block":
        return {kw.arg for kw in node.keywords if kw.arg is not None}
    return None


def test_no_guard_verdict_block_call_uses_unknown_keywords():
    offenders = []
    for path in sorted(GUARD_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            kwargs = _block_call_kwargs(node)
            if kwargs is None:
                continue
            unknown = kwargs - ALLOWED
            if unknown:
                offenders.append(
                    f"{path.name}:{node.lineno}: unknown kwarg(s) {sorted(unknown)}"
                )
    assert not offenders, (
        "GuardVerdict.block() accepts only "
        f"{sorted(ALLOWED)}; bad call sites: {offenders}"
    )


def test_the_scanner_actually_sees_block_calls():
    """Falsifier: a scanner that finds nothing would pass vacuously."""
    found = 0
    for path in sorted(GUARD_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _block_call_kwargs(node) is not None:
                found += 1
    assert found >= 20, f"expected many GuardVerdict.block() call sites, saw {found}"


def test_unknown_keyword_is_detected_in_a_synthetic_site():
    """Falsifier: prove the detector rejects the exact defect that shipped."""
    src = "class V:\n    @classmethod\n    def block(cls, message, reason, category, overridable=True):\n        pass\n\nV.block('m', reason='r', slug='r')\n"
    tree = ast.parse(src)
    bad = [n for n in ast.walk(tree)
           if isinstance(n, ast.Call) and (_block_call_kwargs(n) or set()) - ALLOWED]
    assert bad, "the detector must flag a slug= keyword"

"""The assembled-deliverable replay must be able to FAIL, and must report margins.

This tool exists because a self-check that lays the pieces out in the ORDER you
designed them cannot catch a consumer that assembles them canonically and
locates a boundary by SEARCHING. The tests therefore pin the property that makes
it worth running: the same two pieces that look fine to their author are
rejected when their derived flanks fall outside the stated bound, and the report
carries the margin rather than a bare verdict.

Fixtures are synthetic with a known construction; the file names no target.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TOOL = (Path(__file__).resolve().parents[1]
        / "flagscale_agent" / "tools" / "assembled_deliverable_replay.py")


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    d = tmp_path_factory.mktemp("replaykit")
    (d / "mk.py").write_text(textwrap.dedent("""
        import random
        random.seed(3)
        ALPH = "ACGT"
        comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
        inp = "".join(random.choice(ALPH) for _ in range(600))
        insert = "".join(random.choice(ALPH) for _ in range(40))
        P = 300
        out = inp[:P] + insert + inp[P:]
        open("input.fasta", "w").write(">x\\n" + inp + "\\n")
        open("output.fasta", "w").write(">y\\n" + out + "\\n")
        open("insert.fasta", "w").write(">i\\n" + insert + "\\n")
        fwd = insert + inp[P:P + 17]
        def rc(s):
            return "".join(comp[c] for c in s[::-1])
        open("good.fasta", "w").write(">F\\n" + fwd + "\\n>R\\n" + rc(inp[P - 42:P]) + "\\n")
        open("bad.fasta", "w").write(">F\\n" + fwd + "\\n>R\\n" + rc(inp[P - 82:P]) + "\\n")
    """), encoding="utf-8")
    subprocess.run([sys.executable, "mk.py"], cwd=d, check=True)
    return d


def _run(d, deliverable, *extra):
    return subprocess.run(
        [sys.executable, str(TOOL), "--input", "input.fasta",
         "--insert-fasta", "insert.fasta", "--deliverable", deliverable,
         "--bound", "15:45", *extra],
        cwd=d, capture_output=True, text=True, timeout=300,
    )


def test_tool_lives_inside_the_package():
    assert TOOL.is_file(), f"missing {TOOL} — a tool outside the package is unreachable in a container"


def test_good_construction_passes(kit):
    r = _run(kit, "good.fasta")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "VERDICT: PASS" in r.stdout


def test_long_flank_is_rejected_with_a_margin(kit):
    r = _run(kit, "bad.fasta")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "VERDICT: FAIL" in r.stdout
    assert "OUT OF BOUND" in r.stdout
    assert "margin -" in r.stdout          # the size of the miss, not just a verdict


def test_missing_pieces_is_a_failure_not_a_crash(kit, tmp_path):
    p = tmp_path / "one.fasta"
    p.write_text(">F\nACGTACGTACGTACGT\n", encoding="utf-8")
    r = _run(kit, str(p))
    assert r.returncode == 1
    assert "VERDICT: FAIL" in r.stdout


def test_underivable_insert_is_unmeasured(kit):
    r = subprocess.run(
        [sys.executable, str(TOOL), "--input", "input.fasta", "--deliverable", "good.fasta"],
        cwd=kit, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 2, r.stdout + r.stderr
    assert "VERDICT: UNMEASURED" in r.stdout
    assert "VERDICT: PASS" not in r.stdout


def test_tool_names_no_target():
    txt = TOOL.read_text(encoding="utf-8").lower()
    for tok in ("mteb", "primers", "stolen_a1", "cli_tool", "oligotm", "seed5", "terminal-bench"):
        assert tok not in txt, tok


def test_bare_upper_bound_is_accepted(tmp_path):
    """`--bound 45` (a caller who knows only the upper bound) must not crash.

    Measured: the replay tool raised
    `ValueError: invalid literal for int() with base 10: ''`
    on that form. A check that crashes on the obvious invocation is unusable
    at exactly the moment the agent needs it.
    """
    import subprocess
    import sys
    import pathlib as _pl

    seq = "ACGT" * 60
    comp = str.maketrans("ACGT", "TGCA")
    rc = lambda s: s.translate(comp)[::-1]
    insert = "TTGGCCAA" * 3

    inp = tmp_path / "seq.fasta"
    ins = tmp_path / "ins.fasta"
    deliv = tmp_path / "deliv.fasta"
    inp.write_text(">s\n" + seq + "\n")
    ins.write_text(">i\n" + insert + "\n")
    deliv.write_text(">f\n" + seq[60:75] + "\n>r\n" + rc(seq[40:60]) + insert + "\n")

    tool = _pl.Path(__file__).resolve().parents[1] / "flagscale_agent" / "tools" / "assembled_deliverable_replay.py"
    for bound in ("15:45", "45"):
        r = subprocess.run(
            [sys.executable, str(tool), "--input", str(inp), "--insert-fasta", str(ins),
             "--deliverable", str(deliv), "--bound", bound],
            capture_output=True, text=True,
        )
        assert "Traceback" not in r.stderr, (bound, r.stderr)
        assert "VERDICT:" in r.stdout, (bound, r.stdout)

"""The post-edit inject reports the written file's SIZE — a measurement only.

User directive (2026-11-28):
    "不要这么加，而是post-edit inject，对文件自动进行测量一下size吧，然后inject
     size情况，提醒LLM注意下是否有超出size要求。harness侧自己不要做语义判断。"

WHY (measured, gpt2-codegolf Kdan3fi, job 2026-11-28__14-09-38):
    The task states a hard ceiling ("a C file under 5000 bytes"). The agent
    measured ITSELF once (`wc -c` -> 5713) and its reasoning immediately said
    "over 5000 ... then compact" — yet it never compacted, and the size then GREW
    5713 -> 6083 -> 6193 across 40+ further tool calls, ending at
    `assert 6193 < 5000`. A comparison run the day before compacted
    6452 -> 6197 -> 5016 -> 4970 and cleared that same gate in 16 tool calls
    instead of 43. The number that decided the reward was the one number nothing
    in the harness kept in front of the agent.

DESIGN BOUNDARY — the point of this file:
    The guard reports the NUMBER. It does NOT parse a task's bound, and does not
    decide whether the number violates anything. "Does 6193 exceed <5000" is a
    semantic judgement about the TASK, and belongs to the reader. These tests
    pin that boundary in BOTH directions: the line must carry the number, and
    the guard must not be seen asserting pass/fail against a bound.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flagscale_agent.react.guard import GuardContext  # noqa: E402
from flagscale_agent.react.guard.post_edit_far_end import (  # noqa: E402
    PostEditFarEndGuard,
)


def _inject(path: str) -> str:
    ctx = GuardContext(tool_name="write_file", tool_args={"path": path},
                       tool_result="ok", messages=[])
    v = PostEditFarEndGuard(session_dir_fn=None).check_post(ctx)
    return getattr(v, "message", None) or str(v)


# ── the number is present and EXACT ─────────────────────────────────────────

def test_size_line_reports_the_exact_byte_count(tmp_path) -> None:
    p = tmp_path / "gpt2.c"
    p.write_bytes(b"x" * 6193)  # the size the real trial died at
    assert os.path.getsize(p) == 6193
    line = PostEditFarEndGuard._size_line(str(p))
    assert line is not None
    assert "6193 bytes" in line


def test_size_line_is_in_the_path_branch_inject(tmp_path) -> None:
    """Not just a helper: it must reach the bytes the agent actually reads."""
    p = tmp_path / "small.c"
    p.write_bytes(b"y" * 4846)  # a size that CLEARED the gate in the 11-27 run
    txt = _inject(str(p))
    assert "size: 4846 bytes" in txt


def test_the_number_tracks_the_file_not_a_constant(tmp_path) -> None:
    """A hard-coded/offset value would pass a single check; rewriting the file
    must move the reported number."""
    p = tmp_path / "a.c"
    p.write_bytes(b"x" * 6193)
    first = PostEditFarEndGuard._size_line(str(p))
    p.write_bytes(b"x" * 4970)
    second = PostEditFarEndGuard._size_line(str(p))
    assert "6193 bytes" in first and "4970 bytes" in second
    assert first != second


def test_units_are_included_so_a_bound_needs_no_conversion(tmp_path) -> None:
    """Task bounds are written as bytes, k or KiB. Carrying all three means the
    reader compares directly instead of converting first (the conversion step is
    exactly where '5713 is fine' gets rationalized)."""
    p = tmp_path / "a.c"
    p.write_bytes(b"x" * 5000)
    line = PostEditFarEndGuard._size_line(str(p))
    assert "5000 bytes" in line
    assert "5.0 k" in line
    assert "KiB" in line


# ── quiet failure: no file, no line, no crash ───────────────────────────────

def test_missing_file_is_silent(tmp_path) -> None:
    assert PostEditFarEndGuard._size_line(str(tmp_path / "nope.c")) is None


def test_directory_is_silent(tmp_path) -> None:
    assert PostEditFarEndGuard._size_line(str(tmp_path)) is None


def test_inject_still_renders_when_size_is_unavailable(tmp_path, monkeypatch) -> None:
    """The size line is an enhancement; its absence must not break the inject."""
    p = tmp_path / "gone.c"
    p.write_bytes(b"x" * 100)
    monkeypatch.setattr(PostEditFarEndGuard, "_size_line",
                        classmethod(lambda cls, path: None))
    txt = _inject(str(p))
    assert "[Post-edit]" in txt
    assert "size measured now" not in txt


# ── the boundary: measure, do not judge ─────────────────────────────────────

def test_no_verdict_against_a_bound_is_asserted() -> None:
    """The guard must not decide 'over/under 5000'. It states the count and lets
    the reader judge — the user's explicit constraint. A line that says
    'EXCEEDS the limit' would be the harness doing semantic interpretation."""
    src = Path(
        "flagscale_agent/react/guard/post_edit_far_end.py"
    ).read_text(encoding="utf-8")
    start = src.index("def _size_line")
    end = src.index("def _message", start)
    body = src[start:end]
    for verdict in ("EXCEEDS", "exceeds the", "VIOLATES", "TOO LARGE",
                    "over the limit", "within limit"):
        assert verdict not in body, (
            f"size line asserts a verdict ({verdict!r}); it must only measure")


def test_the_line_points_the_reader_at_their_own_bound() -> None:
    """Boundary stated POSITIVELY: the measurement must hand the comparison to
    the reader (their task's ceiling), not perform it. Without this half, a
    future edit could strip the invitation and leave a bare number that reads
    like trivia."""
    line = PostEditFarEndGuard._size_line("/etc/hostname") or ""
    low = line.lower()
    assert "if the task states a ceiling" in low
    assert "compare this number against it" in low

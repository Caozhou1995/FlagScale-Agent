"""Regression tests for the two read_file defects found on 2026-11-13.

Defect 1 (stale cache): FileCache keyed on the RAW caller path string, so
  `read_file('sub/k.py')` cached under 'sub/k.py' while `edit_file('./sub/k.py')`
  invalidated only './sub/k.py' (edit_file does NOT do the abspath double-key
  that write_file does). A subsequent read returned pre-edit content.
  Measured on the old code: read -> A=1 while the file on disk said A=555.

Defect 2 (whole-file read + unbounded cache): execute() called f.readlines()
  regardless of how few lines were requested, and cached the entire line list
  per path with no byte bound. Measured on the old code: 500 lines requested
  out of a 49.4MB log -> +90MB peak RSS and one full 900k-line list cached;
  8 large paths -> 323MB RSS.
"""
import os
import subprocess
import sys

import pytest

from flagscale_agent.react.tools.read_file import FileCache, ReadFileTool, get_file_cache
from flagscale_agent.react.tools.write_file import WriteFileTool


@pytest.fixture(autouse=True)
def _clean_cache():
    get_file_cache().invalidate_all()
    yield
    get_file_cache().invalidate_all()


def test_cache_key_normalises_path_spellings(tmp_path, monkeypatch):
    """Same file reached through two spellings must be ONE cache entry."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    p = tmp_path / "sub" / "k.py"
    p.write_text("A=1\n")
    tool = ReadFileTool()
    tool.execute(path="sub/k.py")
    tool.execute(path="./sub/k.py")
    tool.execute(path=str(p))
    tool.execute(path="sub/../sub/k.py")
    cache = get_file_cache()
    assert len(cache._store) == 1, f"expected 1 entry, got {list(cache._store)}"


def test_stale_read_after_edit_file_relative_path(tmp_path, monkeypatch):
    """The exact defect: read, edit via a different spelling, read again.

    This is the shape that fails on the pre-fix code because edit_file issues a
    single un-normalised invalidate().
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    p = tmp_path / "sub" / "k.py"
    p.write_text("A=1\n")
    tool = ReadFileTool()
    first = tool.execute(path="sub/k.py")
    assert "A=1" in first

    p.write_text("A=555\n")  # what edit_file ultimately does to the bytes
    get_file_cache().invalidate("./sub/k.py")  # edit_file's un-normalised call

    second = tool.execute(path="sub/k.py")
    assert "A=555" in second, f"stale cache served: {second}"
    assert "A=1" not in second


def test_stale_read_after_write_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "w.txt"
    p.write_text("OLD\n")
    ReadFileTool().execute(path="w.txt")
    WriteFileTool().execute(path="./w.txt", content="NEW\n")
    assert "NEW" in ReadFileTool().execute(path="w.txt")


def test_filecache_get_put_invalidate_are_normalised(tmp_path):
    p = tmp_path / "x" / "y.txt"
    p.parent.mkdir()
    p.write_text("hi\n")
    cache = FileCache()
    cache.put(str(p), ["hi\n"])
    assert cache.get(str(p)) == ["hi\n"]
    assert cache.get(str(p.parent / "." / "y.txt")) == ["hi\n"]
    cache.invalidate(str(p.parent) + "/./y.txt")
    assert cache.get(str(p)) is None


def test_cache_never_exceeds_byte_cap(tmp_path, monkeypatch):
    """A file above _MAX_CACHE_BYTES must not be cached at all."""
    big = tmp_path / "big.log"
    with open(big, "w") as f:
        line = "x" * 99 + "\n"
        for _ in range((ReadFileTool._MAX_CACHE_BYTES // 100) + 5000):
            f.write(line)
    assert os.path.getsize(big) > ReadFileTool._MAX_CACHE_BYTES
    ReadFileTool().execute(path=str(big), start_line=1, end_line=10)
    assert len(get_file_cache()._store) == 0


def test_small_file_still_cached(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text("a\nb\nc\n")
    ReadFileTool().execute(path=str(p))
    assert len(get_file_cache()._store) == 1


def test_ranged_read_memory_is_bounded(tmp_path):
    """Ranged read of a large file must not scale peak RSS with file size.

    Runs in a fresh subprocess so ru_maxrss is this operation's peak only.
    """
    big = tmp_path / "huge.log"
    with open(big, "w") as f:
        line = "line padding-padding-padding-padding abcdefghij\n"
        for i in range(400000):
            f.write(f"{i} {line}")
    size_mb = os.path.getsize(big) / 1e6
    assert size_mb > 15, f"test file too small to be meaningful: {size_mb:.1f}MB"

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import resource,sys;"
        "sys.path.insert(0,%r);"
        "from flagscale_agent.react.tools.read_file import ReadFileTool,get_file_cache;"
        "base=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;"
        "out=ReadFileTool().execute(path=%r,start_line=1,end_line=500);"
        "peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;"
        "print((peak-base)/1024.0, len(out.splitlines()), len(get_file_cache()._store))"
    ) % (repo, str(big))
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    delta_mb, n_lines, n_cached = res.stdout.split()
    delta_mb = float(delta_mb)
    # header + 500 numbered lines + blank line + truncated notice
    assert int(n_lines) == 503, res.stdout
    assert int(n_cached) == 0
    # Old code needed ~2x file size; the windowed read needs O(window).
    assert delta_mb < size_mb / 2, f"peak {delta_mb:.1f}MB vs file {size_mb:.1f}MB"


def test_output_identical_to_readlines_reference(tmp_path):
    """Behavioural equivalence: windowed read must match the old algorithm."""
    p = tmp_path / "f.txt"
    with open(p, "w") as f:
        for i in range(1, 1201):
            f.write(f"row{i}\n")

    def reference(path, start=1, end=None, numbered=True, MAX_LINES=500):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        start = max(1, start)
        if end is None:
            end = min(start + MAX_LINES - 1, total)
        end = min(end, total)
        if start > total:
            return f"File has {total} lines, requested start_line={start} is past end."
        selected = lines[start - 1:end]
        truncated = end < total and (end - start + 1) >= MAX_LINES
        if numbered:
            width = len(str(end))
            output_lines = [f"{i:{width}d}| {ln}" for i, ln in enumerate(selected, start=start)]
        else:
            output_lines = selected
        result = f"[{path}] lines {start}-{end} of {total}\n" + "".join(output_lines)
        if truncated:
            result += f"\n... truncated at {MAX_LINES} lines. Use start_line={end + 1} to continue."
        return result

    tool = ReadFileTool()
    for st, en in [(1, None), (1, 500), (1, 1200), (500, 700), (1199, 1300),
                   (1200, 1200), (1201, None), (700, 650), (0, None), (3, None)]:
        get_file_cache().invalidate_all()
        assert tool.execute(path=str(p), start_line=st, end_line=en) == reference(str(p), st, en), (
            f"mismatch at start={st} end={en}"
        )
    get_file_cache().invalidate_all()
    assert tool.execute(path=str(p), start_line=2, end_line=5, numbered=False) == reference(
        str(p), 2, 5, numbered=False
    )


@pytest.mark.parametrize("content,start,end", [
    ("a\nb\nc", 1, None),          # no trailing newline
    ("a\nb\nc", 3, None),
    ("only\n", 1, None),           # single line
    ("", 1, None),                 # empty
    ("中文行\nemoji \U0001F600 line\n\tTabbed\n", 1, None),  # unicode/tabs
])
def test_boundary_inputs(tmp_path, content, start, end):
    p = tmp_path / "b.txt"
    p.write_text(content, encoding="utf-8")
    out = ReadFileTool().execute(path=str(p), start_line=start, end_line=end)
    if not content:
        # Matches pre-fix behaviour exactly: an empty file reports past-end.
        assert out == "File has 0 lines, requested start_line=1 is past end."
        return
    assert out.startswith(f"[{p}] lines ")
    assert "ERROR" not in out

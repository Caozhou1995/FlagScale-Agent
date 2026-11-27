"""Post-edit inject must cover writes the workspace delta is blind to.

WHY these tests exist. The effect branch detects a write by diffing a snapshot
of os.getcwd() before/after a shell call. That is blind to two real shapes:

  1. a generator script written OUTSIDE the workspace (`cat > /tmp/gen.py`),
     which is the normal shape while an agent iterates;
  2. a deliverable whose EXTENSION is not in _SNAPSHOT_EXTS (`.txt`, `.fasta`),
     which includes real task answers (`/app/out.txt`, `/app/primers.fasta`).

Both were measured in a live trial: a task spent 16 minutes writing
/tmp/*.py and received ZERO reminders, and another wrote its final
`> /app/out.txt` with no Post-edit inject at all — only unrelated advisories.

The fix parses the shell command for the file paths it NAMES and stats those
exact paths. These tests pin that behaviour, including the negative controls
that keep it from spraying reminders at /dev/null or at a command that merely
RUNS a program.
"""

from __future__ import annotations

import pytest

from flagscale_agent.react.guard import GuardContext as _GC
from flagscale_agent.react.guard.post_edit_far_end import (
    PostEditFarEndGuard,
    _parse_shell_write_targets,
)


def _shell(command: str, result: str = "ok"):
    return _GC(tool_name="shell", tool_args={"command": command}, tool_result=result)


@pytest.fixture()
def guard():
    return PostEditFarEndGuard()


# ── the parser ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd,expected", [
    ("cat > /tmp/x.py <<'EOF'", ["/tmp/x.py"]),
    ("cat >> /tmp/x.py <<EOF", ["/tmp/x.py"]),
    ("printf 'a\\n' > /app/out.txt", ["/app/out.txt"]),
    ("echo hi | tee /tmp/a.py", ["/tmp/a.py"]),
    ("echo hi | tee -a /tmp/b.py", ["/tmp/b.py"]),
    ("cp /tmp/a.py /app/del.py", ["/app/del.py"]),
    ("mv /tmp/a.py /app/keep.py", ["/app/keep.py"]),
    ("> /tmp/first.py python3 x", ["/tmp/first.py"]),
    ("cmd 2> /tmp/err.log", ["/tmp/err.log"]),
    ("dd if=a of=/tmp/blob.bin", ["/tmp/blob.bin"]),
])
def test_parser_names_the_written_file(cmd, expected):
    assert _parse_shell_write_targets(cmd) == expected


@pytest.mark.parametrize("cmd", [
    "cmd > /dev/null 2>&1",       # fd duplication + device
    "cmd 2>&1",                   # fd duplication only
    "ls -la /app",                # no write at all
    "python3 /tmp/gen.py",        # RUNS a program; the shell names no target
    'cmd > "$OUT"',               # unresolvable variable — do not guess
    "cat > /proc/self/x",         # procfs pseudo-file
])
def test_parser_stays_silent_when_nothing_is_named(cmd):
    assert _parse_shell_write_targets(cmd) == []


# ── the trigger ───────────────────────────────────────────────────────────

def test_write_outside_workspace_now_fires(guard, monkeypatch, tmp_path):
    """THE regression: a /tmp script the workspace walk cannot see."""
    work = tmp_path / "app"
    work.mkdir()
    monkeypatch.chdir(work)
    target = tmp_path / "elsewhere" / "design.py"
    target.parent.mkdir()

    guard.check_pre(_shell(f"cat > {target} <<'EOF'"))
    target.write_text("print('hi')\n", encoding="utf-8")
    v = guard.check_post(_shell(f"cat > {target} <<'EOF'"))

    assert v is not None and v.action == "inject"
    assert str(target) in v.message


def test_deliverable_extension_outside_snapshot_exts_fires(guard, monkeypatch, tmp_path):
    """`.txt` is not in _SNAPSHOT_EXTS, so the delta never saw the real answer.

    Measured in a live trial: the step that wrote `> /app/out.txt` got NO
    Post-edit inject, while every /app/*.py step did.
    """
    monkeypatch.chdir(tmp_path)
    guard.check_pre(_shell("printf 'flag{x}\\n' > answer.txt"))
    (tmp_path / "answer.txt").write_text("flag{x}\n", encoding="utf-8")
    v = guard.check_post(_shell("printf 'flag{x}\\n' > answer.txt"))
    assert v is not None and "answer.txt" in v.message


def test_named_target_unchanged_does_not_fire(guard, monkeypatch, tmp_path):
    """A command that names a file but writes nothing must stay silent."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "same.py").write_text("x=1\n", encoding="utf-8")
    guard.check_pre(_shell("cat > same.py <<'EOF'"))
    v = guard.check_post(_shell("cat > same.py <<'EOF'"))
    assert v is None


def test_dev_null_and_fd_dup_never_fire(guard, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    guard.check_pre(_shell("ls /app > /dev/null 2>&1"))
    v = guard.check_post(_shell("ls /app > /dev/null 2>&1"))
    assert v is None


def test_running_a_program_is_not_a_named_write(guard, monkeypatch, tmp_path):
    """Stated boundary, not an oversight: `python gen.py` writes inside the
    program, so the shell names no target and the NAMED-target path cannot see
    it. Isolated from the workspace delta by writing a file whose extension is
    outside _SNAPSHOT_EXTS — so a pass here proves the delta did not fire
    either, i.e. this shape is genuinely uncovered (and documented as such)."""
    monkeypatch.chdir(tmp_path)
    guard.check_pre(_shell("python3 /tmp/gen.py"))
    (tmp_path / "produced.artifact").write_text("y=2\n", encoding="utf-8")
    v = guard.check_post(_shell("python3 /tmp/gen.py"))
    assert v is None


def test_same_path_fires_once_per_turn(guard, monkeypatch, tmp_path):
    """_fired_paths dedup must survive: /tmp writes are frequent, and a
    reminder on every single one would be noise."""
    work = tmp_path / "app"
    work.mkdir()
    monkeypatch.chdir(work)
    target = tmp_path / "tmpd" / "s.py"
    target.parent.mkdir()

    guard.check_pre(_shell(f"cat > {target} <<'EOF'"))
    target.write_text("a=1\n", encoding="utf-8")
    assert guard.check_post(_shell(f"cat > {target} <<'EOF'")) is not None

    guard.check_pre(_shell(f"cat > {target} <<'EOF'"))
    target.write_text("a=2\n", encoding="utf-8")
    assert guard.check_post(_shell(f"cat > {target} <<'EOF'")) is None


def test_named_target_delivers_the_shared_semantic_nudges(guard, monkeypatch, tmp_path):
    """A /tmp write must get the SAME reminders the path branch delivers."""
    work = tmp_path / "app"
    work.mkdir()
    monkeypatch.chdir(work)
    target = tmp_path / "tmpx" / "gen.py"
    target.parent.mkdir()

    guard.check_pre(_shell(f"cat > {target} <<'EOF'"))
    target.write_text("z=3\n", encoding="utf-8")
    v = guard.check_post(_shell(f"cat > {target} <<'EOF'"))
    assert v is not None
    assert "THINKING↔BYTES DIFF" in v.message
    assert "COLD-CONSUMER probe" in v.message


def test_never_blocks(guard, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert guard.check_pre(_shell("cat > /tmp/q.py <<'EOF'")) is None

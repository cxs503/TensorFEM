"""Regression net for the diff-scoped lint gate (``scripts/lint_scope.py``).

The gate decides what the CI ``lint (advisory)`` job actually measures, so its
two failure modes matter more than ruff itself: it must never fall back to
"whole tree" (that red is the pre-existing backlog) and it must never report
"clean" when it cannot run (missing ruff, unresolvable base).  These tests pin
the scope selection and the fail-closed exit codes with a stub ``ruff``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lint_scope.py"

STUB_RUFF = """#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
if args == ["--version"]:
    print("ruff 0.0.0 (stub)")
    sys.exit(0)
log = os.environ.get("RUFF_LOG")
if log:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(" ".join(args) + "\\n")
print("stub ruff:", " ".join(args))
sys.exit(int(os.environ.get("RUFF_EXIT", "0")))
"""


def load_gate():
    spec = importlib.util.spec_from_file_location("lint_scope", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def gate():
    return load_gate()


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def repo(tmp_path):
    """A git repository with one committed module and one modified module."""
    git(tmp_path, "init", "-q")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "untouched.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "legacy.py").write_text("OLD = 1\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "pkg" / "legacy.py").write_text("OLD = 2\n", encoding="utf-8")
    git(tmp_path, "commit", "-qam", "touch legacy")
    return tmp_path


@pytest.fixture()
def stub_ruff(tmp_path, monkeypatch):
    path = tmp_path / "stub_ruff"
    path.write_text(STUB_RUFF, encoding="utf-8")
    path.chmod(0o755)
    monkeypatch.setenv("RUFF", str(path))
    monkeypatch.setenv("RUFF_LOG", str(tmp_path / "ruff.log"))
    return path


def ruff_calls(tmp_path):
    log = tmp_path / "ruff.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def test_scope_is_the_diff_not_the_whole_tree(repo, gate):
    merge_base = gate.resolve_base(repo, "HEAD~1")
    files = gate.changed_python_files(repo, merge_base)
    assert files == ["pkg/legacy.py"]


def test_merge_base_keeps_already_merged_work_out_of_scope(repo, gate):
    """A branch that merged a moved main is judged only on its own change."""
    trunk = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "pkg" / "side.py").write_text("SIDE = 1\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "side work")
    git(repo, "checkout", "-q", trunk)
    (repo / "pkg" / "main_moved.py").write_text("MAIN = 1\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "main moved on")
    git(repo, "checkout", "-q", "side")
    git(repo, "merge", "-q", "--no-edit", trunk)
    # main_moved.py arrived from the trunk merge: not this branch's lint debt.
    assert gate.changed_python_files(repo, gate.resolve_base(repo, trunk)) == ["pkg/side.py"]


def test_clean_change_passes(repo, stub_ruff, tmp_path, gate, monkeypatch):
    report = tmp_path / "report.json"
    code = gate.main(
        [
            "--repo",
            str(repo),
            "--base",
            "HEAD~1",
            "--report",
            str(report),
        ]
    )
    assert code == 0
    payload = report.read_text(encoding="utf-8")
    assert '"clean": true' in payload
    assert '"pkg/legacy.py"' in payload
    calls = ruff_calls(tmp_path)
    assert any(c.startswith("check pkg/legacy.py") for c in calls)
    assert any(c.startswith("format --check pkg/legacy.py") for c in calls)
    # The untouched file is never linted: that is the whole point of the gate.
    assert all("untouched.py" not in c for c in calls)


def test_violation_fails_with_exit_code_1(repo, stub_ruff, tmp_path, gate, monkeypatch):
    monkeypatch.setenv("RUFF_EXIT", "1")
    report = tmp_path / "report.json"
    code = gate.main(["--repo", str(repo), "--base", "HEAD~1", "--report", str(report)])
    assert code == 1
    assert '"clean": false' in report.read_text(encoding="utf-8")


def test_no_python_change_passes_without_running_ruff(tmp_path, stub_ruff, gate):
    git(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("docs only\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "README.md").write_text("docs only, edited\n", encoding="utf-8")
    git(tmp_path, "commit", "-qam", "docs")
    assert gate.main(["--repo", str(tmp_path), "--base", "HEAD~1"]) == 0
    assert ruff_calls(tmp_path) == []


def test_untracked_files_count_with_worktree(repo, stub_ruff, tmp_path, gate):
    (repo / "pkg" / "brand_new.py").write_text("NEW = 1\n", encoding="utf-8")
    without = gate.changed_python_files(repo, gate.resolve_base(repo, "HEAD~1"))
    with_worktree = gate.changed_python_files(
        repo, gate.resolve_base(repo, "HEAD~1"), include_worktree=True
    )
    assert "pkg/brand_new.py" not in without
    assert "pkg/brand_new.py" in with_worktree


def test_deleted_files_are_not_linted(repo, stub_ruff, tmp_path, gate):
    (repo / "pkg" / "gone.py").write_text("GONE = 1\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "add gone")
    (repo / "pkg" / "gone.py").unlink()
    git(repo, "commit", "-qam", "delete gone")
    assert gate.changed_python_files(repo, gate.resolve_base(repo, "HEAD~1")) == []


def test_missing_ruff_fails_closed(repo, tmp_path, gate, monkeypatch):
    monkeypatch.delenv("RUFF", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    code = gate.main(["--repo", str(repo), "--base", "HEAD~1"])
    assert code == 2


def test_unresolvable_base_fails_closed(repo, stub_ruff, gate):
    assert gate.main(["--repo", str(repo), "--base", "no-such-ref"]) == 2


def test_missing_base_fails_closed(repo, stub_ruff, gate, monkeypatch):
    monkeypatch.delenv("LINT_BASE", raising=False)
    assert gate.main(["--repo", str(repo)]) == 2


def test_backlog_is_measured_but_never_gates(repo, stub_ruff, tmp_path, gate):
    report = tmp_path / "report.json"
    code = gate.main(
        ["--repo", str(repo), "--base", "HEAD~1", "--measure-backlog", "--report", str(report)]
    )
    assert code == 0
    payload = report.read_text(encoding="utf-8")
    assert '"backlog"' in payload
    assert any("--statistics" in c for c in ruff_calls(tmp_path))


def test_script_is_runnable_as_a_module_entry_point():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0
    assert "diff" in proc.stdout.lower()

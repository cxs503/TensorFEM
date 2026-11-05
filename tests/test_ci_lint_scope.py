"""Tests for scripts/ci_lint_scope.py -- the changed-files ruff ratchet scope.

The helper decides *which* files the CI lint gate runs on.  If it silently
returned an empty list for a real code change the gate would be dead, so these
tests pin the selection logic (roots, filters, base resolution, fail-open
behaviour) against throwaway git repositories.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_lint_scope.py"
_spec = importlib.util.spec_from_file_location("ci_lint_scope", SCRIPT)
ci_lint_scope = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_lint_scope)

ZERO = "0" * 40


def _run(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=ci@example.invalid",
            "-c",
            "user.name=ci",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "scripts").mkdir()
    (repo / "examples").mkdir()
    (repo / "docs").mkdir()
    (repo / "src" / "kept.py").write_text("A = 1\n", encoding="utf-8")
    (repo / "src" / "dropped.py").write_text("B = 2\n", encoding="utf-8")
    (repo / "docs" / "note.md").write_text("base\n", encoding="utf-8")
    _run(repo, "init", "-q")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "base")
    return repo


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_scope_selects_only_changed_python_in_linted_roots(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    base = _head(repo)
    # touched: src/kept.py (modify), tests/new_test.py (add), scripts/tool.py (add),
    #          src/dropped.py (delete -> not linted), docs/note.md (not python)
    (repo / "src" / "kept.py").write_text("A = 2\n", encoding="utf-8")
    (repo / "tests" / "new_test.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8"
    )
    (repo / "scripts" / "tool.py").write_text("print(1)\n", encoding="utf-8")
    (repo / "src" / "dropped.py").unlink()
    (repo / "docs" / "note.md").write_text("changed\n", encoding="utf-8")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "work")

    files = ci_lint_scope.changed_files(repo, base)
    assert files == ["scripts/tool.py", "src/kept.py", "tests/new_test.py"]


def test_scope_is_empty_for_docs_only_change(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    base = _head(repo)
    (repo / "docs" / "note.md").write_text("only docs\n", encoding="utf-8")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "docs")
    assert ci_lint_scope.changed_files(repo, base) == []


def test_zero_base_falls_back_to_parent_commit(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "src" / "kept.py").write_text("A = 3\n", encoding="utf-8")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "work")

    base, why = ci_lint_scope.resolve_base(repo, ZERO)
    assert base == "HEAD^"
    assert "HEAD^" in why
    assert ci_lint_scope.changed_files(repo, base) == ["src/kept.py"]


def test_unknown_base_fails_open_with_empty_scope(tmp_path: Path, capsys) -> None:
    repo = _make_repo(tmp_path)
    rc = ci_lint_scope.main(["--repo", str(repo), "--base", "deadbeef" * 5])
    out = capsys.readouterr().out
    assert rc == 0  # never kill the pipeline
    assert "::warning" in out
    assert not out.strip().splitlines()[-1:] == ["src/kept.py"]


def test_strict_mode_signals_unresolvable_base(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert ci_lint_scope.main(["--repo", str(repo), "--base", "deadbeef" * 5, "--strict"]) == 2


def test_out_file_receives_newline_separated_scope(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    base = _head(repo)
    (repo / "src" / "kept.py").write_text("A = 4\n", encoding="utf-8")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "work")

    out = tmp_path / "scope.txt"
    assert ci_lint_scope.main(["--repo", str(repo), "--base", base, "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "src/kept.py\n"


def test_custom_root_restricts_scope(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    base = _head(repo)
    (repo / "src" / "kept.py").write_text("A = 5\n", encoding="utf-8")
    (repo / "scripts" / "tool.py").write_text("print(2)\n", encoding="utf-8")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-qm", "work")
    assert ci_lint_scope.changed_files(repo, base, roots=("scripts",)) == ["scripts/tool.py"]

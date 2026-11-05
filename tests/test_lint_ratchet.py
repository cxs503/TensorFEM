"""Unit tests for the lint ratchet gate (``scripts/lint_ratchet.py``).

The gate is what makes the CI ``lint`` job meaningful on a tree that still
carries a large pre-existing ruff backlog: a change set may not *add* findings
and may not add newly unformatted code.  These tests pin both halves with
positive controls (clean changes pass) and negative controls (new debt fails),
using throw-away git repositories so the real tree is never touched.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "lint_ratchet.py"

pytestmark = pytest.mark.skipif(
    shutil.which("ruff") is None or shutil.which("git") is None,
    reason="lint ratchet needs ruff and git on PATH",
)

CLEAN = '"""Clean module."""\n\nVALUE = 1\n'
DIRTY = '"""Dirty module."""\n\n\ndef f(a):\n    b = 1; return a + b\n'  # E702, also unformatted
UNFORMATTED = '"""Unformatted but lint-clean."""\n\n\nVALUE = {"a":1}\n'  # format-only finding


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=tester", *args],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


def _commit(repo: Path, message: str = "change") -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _make_repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy(REPO_ROOT / "pyproject.toml", repo / "pyproject.toml")
    (repo / ".gitignore").write_text("")
    _git(repo, "init", "-q", "-b", "main")
    for name, body in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return repo, _commit(repo, "base")


def _ratchet(repo: Path, base: str, *extra: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    summary = repo / "artifacts" / "ratchet.json"
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--base", base, "--summary", str(summary), *extra],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    payload = json.loads(summary.read_text()) if summary.is_file() else {}
    return res, payload


def test_new_clean_file_passes(tmp_path: Path):
    repo, base = _make_repo(tmp_path, {"pkg/legacy.py": CLEAN})
    (repo / "pkg" / "fresh.py").write_text(CLEAN)
    _commit(repo)
    res, payload = _ratchet(repo, base)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "PASS" in res.stdout
    assert [row["file"] for row in payload["files"]] == ["pkg/fresh.py"]
    assert payload["regressions"] == 0


def test_new_dirty_file_fails(tmp_path: Path):
    repo, base = _make_repo(tmp_path, {"pkg/legacy.py": CLEAN})
    (repo / "pkg" / "fresh.py").write_text(DIRTY)
    _commit(repo)
    res, payload = _ratchet(repo, base)
    assert res.returncode == 1, res.stdout + res.stderr
    assert "REGRESSION" in res.stdout
    assert [row["file"] for row in payload["files"]] == ["pkg/fresh.py"]
    assert payload["files"][0]["head_findings"] >= 1
    assert payload["regressions"] == 1


def test_new_unformatted_file_fails(tmp_path: Path):
    repo, base = _make_repo(tmp_path, {"pkg/legacy.py": CLEAN})
    (repo / "pkg" / "fresh.py").write_text(UNFORMATTED)
    _commit(repo)
    res, payload = _ratchet(repo, base)
    assert res.returncode == 1, res.stdout + res.stderr
    assert "not `ruff format` clean" in res.stdout
    assert payload["files"][0]["head_findings"] == 0
    assert payload["files"][0]["head_format_clean"] is False


def test_untouched_legacy_backlog_is_not_a_gate(tmp_path: Path):
    # The legacy file already carries a finding and is unformatted; leaving it
    # alone must not even be inspected.
    repo, base = _make_repo(tmp_path, {"pkg/legacy.py": DIRTY})
    (repo / "pkg" / "fresh.py").write_text(CLEAN)
    _commit(repo)
    res, payload = _ratchet(repo, base)
    assert res.returncode == 0, res.stdout + res.stderr
    assert [row["file"] for row in payload["files"]] == ["pkg/fresh.py"]


def test_editing_legacy_file_without_growth_passes(tmp_path: Path):
    repo, base = _make_repo(tmp_path, {"pkg/legacy.py": DIRTY})
    (repo / "pkg" / "legacy.py").write_text(DIRTY + "\n\n# keep the debt flat\n")
    _commit(repo)
    res, payload = _ratchet(repo, base)
    assert res.returncode == 0, res.stdout + res.stderr
    row = payload["files"][0]
    assert row["base_findings"] == row["head_findings"] >= 1
    assert row["base_format_clean"] is False and row["head_format_clean"] is False


def test_editing_legacy_file_that_grows_fails(tmp_path: Path):
    repo, base = _make_repo(tmp_path, {"pkg/legacy.py": DIRTY})
    (repo / "pkg" / "legacy.py").write_text(DIRTY + "\n\ndef g(a):\n    b = 2; return a - b\n")
    _commit(repo)
    res, _ = _ratchet(repo, base)
    assert res.returncode == 1, res.stdout + res.stderr
    assert "findings 1 -> 2" in res.stdout


def test_unusable_base_skips_instead_of_failing(tmp_path: Path):
    repo, _ = _make_repo(tmp_path, {"pkg/legacy.py": CLEAN})
    res, _ = _ratchet(repo, "0" * 40)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "skipping the ratchet" in res.stdout


def test_backlog_snapshot_is_advisory_only(tmp_path: Path):
    repo, base = _make_repo(tmp_path, {"src/pkg/legacy.py": DIRTY})
    (repo / "src" / "pkg" / "fresh.py").write_text(CLEAN)
    _commit(repo)
    res, payload = _ratchet(repo, base)
    assert res.returncode == 0, res.stdout + res.stderr
    # The legacy finding is counted by the advisory snapshot but never gates.
    assert payload["backlog"]["findings"] >= 1
    assert payload["regressions"] == 0

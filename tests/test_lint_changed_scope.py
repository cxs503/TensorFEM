"""Scope tests for ``scripts/lint_changed.py``, the incremental ruff ratchet.

The ratchet is what turns the CI lint job from "always red on a pre-existing
4.2k-finding backlog" into a gate on the files a change actually touches, so the
selection logic itself needs a regression net: a silent widening (linting the
whole tree again) would restore the old always-red behaviour, and a silent
narrowing (gating nothing) would make the job a rubber stamp.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "lint_changed.py"

pytestmark = pytest.mark.skipif(
    shutil.which("ruff") is None
    and subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True).returncode
    != 0,
    reason="ruff is not installed (install the dev extra)",
)

LEGACY = '''"""Grandfathered legacy module: two statements on one line (E702)."""

X = 1; Y = 2
'''
"""Pre-existing debt: ruff reports E702 and would reformat the file."""

CLEAN = '''"""Clean module gated by the ratchet."""

VALUE = 1
'''


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A tiny repository: one grandfathered file, one clean file, one commit."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "ci@example.com")
    _git(root, "config", "user.name", "ci")
    _write(root, "src/legacy.py", LEGACY)
    _write(root, "src/module.py", CLEAN)
    _write(root, "docs/notes.md", "# notes\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=root, capture_output=True, text=True
    )


def _report(path: Path) -> dict:
    return json.loads(path.read_text())


def test_untouched_legacy_debt_is_grandfathered(repo: Path, tmp_path: Path) -> None:
    """A clean edit passes even though the repository-wide baseline is dirty."""
    _write(repo, "src/module.py", CLEAN.replace("VALUE = 1", "VALUE = 2"))
    report = tmp_path / "lint.json"
    proc = _run(repo, "--base", "HEAD", "--report", str(report))

    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = _report(report)
    assert payload["mode"] == "changed"
    assert payload["files"] == ["src/module.py"], payload["files"]
    assert payload["clean"] is True
    assert [r["rule"] for r in payload["results"]] == ["check", "format"]


def test_new_violation_in_changed_file_fails(repo: Path, tmp_path: Path) -> None:
    """Fresh E702 debt is caught, and only the changed file is named."""
    _write(repo, "src/module.py", CLEAN + "\nA = 1; B = 2\n")
    proc = _run(repo, "--base", "HEAD")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "E702" in proc.stdout
    assert "src/module.py" in proc.stdout
    assert "src/legacy.py" not in proc.stdout


def test_unformatted_change_fails_on_format_check(repo: Path) -> None:
    """A formatted-but-not-normalised edit trips the second command."""
    _write(repo, "src/module.py", CLEAN + "\n\ndef f(a):\n    return a\n\nRESULT=f(1)\n")
    proc = _run(repo, "--base", "HEAD")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Would reformat" in proc.stdout


def test_non_python_change_lints_nothing(repo: Path, tmp_path: Path) -> None:
    """Documentation-only changes stay green and report an empty set."""
    _write(repo, "docs/notes.md", "# notes\n\nmore\n")
    report = tmp_path / "lint.json"
    proc = _run(repo, "--base", "HEAD", "--report", str(report))

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _report(report)["files"] == []
    assert "--fail-on-empty" not in proc.args


def test_untracked_new_file_is_gated(repo: Path) -> None:
    """New files are linted before their first commit, when fixing is cheapest."""
    _write(repo, "tests/test_new.py", '"""New test module."""\nX = 1; Y = 2\n')
    proc = _run(repo, "--base", "HEAD")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "tests/test_new.py" in proc.stdout


def test_all_mode_exposes_the_backlog(repo: Path, tmp_path: Path) -> None:
    """`--all` is the repo-wide invocation, so the legacy debt resurfaces."""
    report = tmp_path / "lint.json"
    proc = _run(repo, "--all", "--report", str(report))

    assert proc.returncode == 1, proc.stdout + proc.stderr
    payload = _report(report)
    assert payload["mode"] == "all"
    assert "src/legacy.py" in payload["files"]


def test_unresolvable_base_fails_closed(repo: Path) -> None:
    """A broken comparison base must not be read as 'nothing to lint'."""
    proc = _run(repo, "--base", "does-not-exist")

    assert proc.returncode == 2
    assert "does not resolve" in proc.stderr


def test_missing_ruff_fails_closed(repo: Path) -> None:
    """A lint job that cannot run ruff must be an error, not a pass."""
    proc = _run(repo, "--base", "HEAD", "--ruff", "/nonexistent/ruff")

    assert proc.returncode == 2
    assert "ruff is not available" in proc.stderr


def test_fail_on_empty_flags_a_hollow_change_set(repo: Path) -> None:
    """`--fail-on-empty` distinguishes 'nothing changed' from 'all clean'."""
    proc = _run(repo, "--base", "HEAD", "--fail-on-empty")

    assert proc.returncode == 2
    assert "empty" in proc.stderr

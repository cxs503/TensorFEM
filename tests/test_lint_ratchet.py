"""Tests for the lint ratchet gate (``scripts/lint_ratchet.py``).

The gate's contract: a change may keep inherited lint debt, but it may not add
any -- and it must fail closed when it cannot measure.  Every case below builds
a throw-away git repository with a pre-existing backlog on ``main`` and a small
feature branch, then runs the gate exactly as CI does.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lint_ratchet.py"
RUFF = shutil.which("ruff")

pytestmark = pytest.mark.skipif(RUFF is None, reason="ruff is not installed")

# Debt that already exists on ``main``: two packed statements plus an unused
# import, and one-pass formatting would rewrite the file.
LEGACY = "import os\n\n\ndef legacy(a):\n    x = 1; y = 2\n    return a + x + y\n"


def git(repo: Path, *args: str) -> str:
    ident = ["-c", "user.name=ratchet", "-c", "user.email=ratchet@example.invalid"]
    cp = subprocess.run(
        ["git", *ident, *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return cp.stdout


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """``main`` carries the backlog; ``HEAD`` is a feature branch off it."""
    root = tmp_path / "repo"
    for d in ("src", "tests", "scripts", "examples"):
        (root / d).mkdir(parents=True)
    (root / "src" / "legacy.py").write_text(LEGACY, encoding="utf-8")
    (root / "README.md").write_text("baseline\n", encoding="utf-8")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "baseline with inherited debt")
    git(root, "checkout", "-qb", "feature")
    return root


def ratchet(
    repo: Path, base: str = "main", json_out: Path | None = None
) -> subprocess.CompletedProcess:
    args = [sys.executable, str(SCRIPT), "--base", base]
    if json_out is not None:
        args += ["--json-out", str(json_out)]
    return subprocess.run(args, cwd=repo, capture_output=True, text=True, check=False)


def commit(repo: Path, message: str = "change") -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)


def test_gate_is_live_and_inherited_debt_does_not_fail(repo: Path, tmp_path: Path) -> None:
    """The headline case: a clean change on a debt-laden tree passes."""
    out = tmp_path / "report.json"
    (repo / "src" / "clean.py").write_text('"""Clean module."""\n\nVALUE = 1\n', encoding="utf-8")
    commit(repo)

    cp = ratchet(repo, json_out=out)
    report = json.loads(out.read_text())

    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert report["regressions"] == []
    assert report["probe"]["findings"] >= 2  # the measurement is not blind
    assert report["probe"]["unformatted"] == 1
    # the backlog itself is reported, not gated
    assert report["whole_tree"]["findings"] >= 2
    assert report["whole_tree"]["unformatted"] >= 1
    assert [row["path"] for row in report["files"]] == ["src/clean.py"]


def test_touching_a_debt_carrying_file_without_adding_debt_passes(repo: Path) -> None:
    """Editing a legacy file is allowed as long as its debt does not grow."""
    (repo / "src" / "legacy.py").write_text("# edited\n" + LEGACY, encoding="utf-8")
    commit(repo)

    cp = ratchet(repo)

    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "OK: this change adds no lint debt" in cp.stdout


def test_new_findings_in_a_new_file_fail(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    (repo / "src" / "newmod.py").write_text('"""Module."""\n\nimport sys\n', encoding="utf-8")
    commit(repo)

    cp = ratchet(repo, json_out=out)
    report = json.loads(out.read_text())

    assert cp.returncode == 1
    assert "FAIL" in cp.stdout
    assert report["regressions"][0]["growth"] == {"F401": 1}
    assert "new findings" in report["regressions"][0]["reasons"]


def test_new_findings_in_a_legacy_file_fail(repo: Path) -> None:
    (repo / "src" / "legacy.py").write_text(LEGACY + "    q = 3; w = 4\n", encoding="utf-8")
    commit(repo)

    cp = ratchet(repo)

    assert cp.returncode == 1
    assert "E702" in cp.stdout


def test_unformatted_new_file_fails(repo: Path) -> None:
    (repo / "src" / "clean.py").write_text('"""Module."""\n\nVALUE   =   1\n', encoding="utf-8")
    commit(repo)

    cp = ratchet(repo)

    assert cp.returncode == 1
    assert "newly unformatted" in cp.stdout


def test_out_of_scope_change_passes_without_measuring(repo: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    (repo / "README.md").write_text("edited\n", encoding="utf-8")
    commit(repo)

    cp = ratchet(repo, json_out=out)
    report = json.loads(out.read_text())

    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert report["files"] == []
    assert report["note"] == "no in-scope Python file changed"


def test_unresolvable_base_ref_fails_closed(repo: Path) -> None:
    (repo / "src" / "clean.py").write_text('"""Module."""\n\nVALUE = 1\n', encoding="utf-8")
    commit(repo)

    cp = ratchet(repo, base="refs/heads/does-not-exist")

    assert cp.returncode == 2
    assert "cannot resolve base ref" in cp.stderr

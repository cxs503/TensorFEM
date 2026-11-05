"""Tests for the ruff lint ratchet in ``scripts/lint_ratchet.py``.

The ratchet is what keeps the frozen ruff backlog from growing, so the
comparison logic and the shipped baseline are both under test here:

* the comparison must reject new violations, new rules, new files and new
  format debt while accepting (and reporting) debt that was paid down;
* debt that the *base branch* already carries is tolerated, so a pull request
  built on top of a base that moved on (and re-froze its own baseline) is not
  blamed for it, while debt the branch itself adds is still rejected;
* the shipped ``scripts/lint_baseline.json`` must describe the tree it ships
  with, and the CLI must return 1 as soon as the tree grows past it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lint_ratchet", ROOT / "scripts" / "lint_ratchet.py")
lint_ratchet = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("lint_ratchet", lint_ratchet)
SPEC.loader.exec_module(lint_ratchet)

BASELINE_PATH = ROOT / "scripts" / "lint_baseline.json"
GIT = shutil.which("git")


def _git(repo: Path, *args: str) -> None:
    """Run git in ``repo`` with a throwaway identity (tests never touch $HOME config)."""
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=lint",
        "-c",
        "user.email=lint@example.invalid",
        "commit",
        "-q",
        "-m",
        message,
    )


def _state(
    files: dict[str, dict[str, int]],
    unformatted: list[str] | None = None,
    ruff_version: str = "0.15.0",
) -> dict:
    totals: dict[str, int] = {}
    for counts in files.values():
        for rule, count in counts.items():
            totals[rule] = totals.get(rule, 0) + count
    return {
        "schema": 1,
        "ruff_version": ruff_version,
        "paths": ["src", "tests", "scripts", "examples"],
        "check": {"totals": totals, "files": files},
        "format": {"unformatted": sorted(unformatted or [])},
    }


BASELINE = _state({"src/a.py": {"E702": 3, "F401": 1}}, ["src/b.py"])


def test_identical_tree_passes():
    assert lint_ratchet.regression_problems(BASELINE, BASELINE) == []


def test_paid_down_debt_passes():
    better = _state({"src/a.py": {"E702": 2}}, [])
    assert lint_ratchet.regression_problems(better, BASELINE) == []


def test_grown_file_debt_is_rejected():
    worse = _state({"src/a.py": {"E702": 4, "F401": 1}}, ["src/b.py"])
    problems = lint_ratchet.regression_problems(worse, BASELINE)
    assert any("src/a.py: E702 3 -> 4 (+1)" in problem for problem in problems)


def test_new_rule_in_existing_file_is_rejected():
    worse = _state({"src/a.py": {"E702": 3, "F401": 1, "E741": 1}}, ["src/b.py"])
    problems = lint_ratchet.regression_problems(worse, BASELINE)
    assert any("src/a.py: E741 0 -> 1 (+1)" in problem for problem in problems)


def test_new_file_debt_is_rejected():
    worse = _state({"src/a.py": {"E702": 3, "F401": 1}, "src/new.py": {"E741": 2}}, ["src/b.py"])
    problems = lint_ratchet.regression_problems(worse, BASELINE)
    assert any("src/new.py: E741 0 -> 2 (+2)" in problem for problem in problems)


def test_new_format_debt_is_rejected():
    worse = _state({"src/a.py": {"E702": 3, "F401": 1}}, ["src/b.py", "src/c.py"])
    problems = lint_ratchet.regression_problems(worse, BASELINE)
    assert any("src/c.py: ruff format would rewrite" in problem for problem in problems)
    assert len(problems) == 1


def test_ruff_version_drift_is_rejected():
    drifted = _state({"src/a.py": {"E702": 3, "F401": 1}}, ["src/b.py"], ruff_version="0.16.0")
    problems = lint_ratchet.regression_problems(drifted, BASELINE)
    assert len(problems) == 1
    assert "ruff version changed" in problems[0]


def test_base_branch_debt_is_tolerated():
    """Debt the base branch already carries must not fail a branch built on it."""
    base = _state(
        {"src/a.py": {"E702": 3, "F401": 1}, "src/moved_on.py": {"E741": 2}}, ["src/new.py"]
    )
    current = _state(
        {"src/a.py": {"E702": 3, "F401": 1}, "src/moved_on.py": {"E741": 2}},
        ["src/b.py", "src/new.py"],
    )
    assert lint_ratchet.regression_problems(current, BASELINE) != []
    allowed = lint_ratchet.tolerated_baseline(BASELINE, base)
    assert lint_ratchet.regression_problems(current, allowed) == []


def test_branch_debt_above_the_base_is_rejected():
    """The allowance is a floor, not an amnesty: growing a base file is still debt."""
    base = _state({"src/moved_on.py": {"E741": 2}})
    current = _state(
        {"src/a.py": {"E702": 3, "F401": 1, "E741": 1}, "src/moved_on.py": {"E741": 3}},
        ["src/b.py", "src/own.py"],
    )
    problems = lint_ratchet.regression_problems(
        current, lint_ratchet.tolerated_baseline(BASELINE, base)
    )
    assert any("src/a.py: E741 0 -> 1 (+1)" in problem for problem in problems)
    assert any("src/moved_on.py: E741 2 -> 3 (+1)" in problem for problem in problems)
    assert any("src/own.py: ruff format would rewrite" in problem for problem in problems)
    assert len(problems) == 3


def test_tolerated_baseline_without_a_base_is_the_shipped_baseline():
    assert lint_ratchet.tolerated_baseline(BASELINE, None) == BASELINE


def test_tolerated_baseline_ignores_a_version_mismatched_base():
    base = _state({"src/moved_on.py": {"E741": 2}}, ruff_version="0.16.0")
    assert lint_ratchet.tolerated_baseline(BASELINE, base) == BASELINE


@pytest.mark.skipif(GIT is None, reason="git is not installed")
def test_read_baseline_from_git(tmp_path: Path):
    (tmp_path / "scripts").mkdir()
    payload = _state({"src/a.py": {"E702": 3}})
    (tmp_path / "scripts" / "lint_baseline.json").write_text(json.dumps(payload), encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _commit(tmp_path, "freeze")
    assert lint_ratchet.read_baseline_from_git("HEAD", tmp_path) == payload
    assert lint_ratchet.read_baseline_from_git("no-such-ref", tmp_path) is None


def test_render_summary_reports_counts():
    text = lint_ratchet.render_summary("frozen baseline", BASELINE)
    assert "4 ruff violations in 1 files" in text
    assert "1 files ruff format would rewrite" in text


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
def test_cli_end_to_end_accepts_baseline_and_rejects_new_debt(tmp_path: Path):
    """Freeze a throwaway tree, then grow it and watch the CLI turn red."""
    for directory in lint_ratchet.LINT_PATHS:
        (tmp_path / directory).mkdir()
    sample = tmp_path / "src" / "module.py"
    sample.write_text("value = 1;other = 2\n", encoding="utf-8")

    freeze = lint_ratchet.main(["--root", str(tmp_path), "--update"])
    assert freeze == 0
    assert (tmp_path / lint_ratchet.BASELINE_PATH).exists()

    assert lint_ratchet.main(["--root", str(tmp_path), "--summary"]) == 0
    assert lint_ratchet.main(["--root", str(tmp_path)]) == 0

    sample.write_text("value = 1;other = 2;third = 3\n", encoding="utf-8")
    assert lint_ratchet.main(["--root", str(tmp_path)]) == 1

    (tmp_path / "src" / "fresh.py").write_text("unused = 1;\n", encoding="utf-8")
    assert lint_ratchet.main(["--root", str(tmp_path)]) == 1


@pytest.mark.skipif(
    shutil.which("ruff") is None or GIT is None, reason="ruff and git are both needed"
)
def test_cli_tolerates_debt_inherited_from_the_base_branch(tmp_path: Path):
    """A branch whose base re-froze a grown baseline stays green, its own debt does not."""
    for directory in lint_ratchet.LINT_PATHS:
        (tmp_path / directory).mkdir()
    (tmp_path / "src" / "module.py").write_text("value = 1;other = 2\n", encoding="utf-8")
    assert lint_ratchet.main(["--root", str(tmp_path), "--update"]) == 0
    _git(tmp_path, "init", "-q")
    _commit(tmp_path, "freeze the baseline")
    shipped = (tmp_path / lint_ratchet.BASELINE_PATH).read_text(encoding="utf-8")

    # The base branch moves on: a new file with debt lands there and the base
    # re-freezes its own baseline.  This branch still ships the older freeze.
    (tmp_path / "src" / "inherited.py").write_text("import os\n", encoding="utf-8")
    assert lint_ratchet.main(["--root", str(tmp_path), "--update"]) == 0
    _commit(tmp_path, "base moves on")
    (tmp_path / lint_ratchet.BASELINE_PATH).write_text(shipped, encoding="utf-8")

    assert lint_ratchet.main(["--root", str(tmp_path)]) == 1
    assert lint_ratchet.main(["--root", str(tmp_path), "--base-ref", "HEAD"]) == 0

    (tmp_path / "src" / "own.py").write_text("import sys\n", encoding="utf-8")
    assert lint_ratchet.main(["--root", str(tmp_path), "--base-ref", "HEAD"]) == 1


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
def test_shipped_baseline_describes_the_tree():
    """The committed baseline must match the tree it ships with.

    On CI the tree under test already contains its base branch, which may have
    moved on after this baseline was frozen; ``$LINT_BASELINE_REF`` names that
    base so its committed baseline joins the allowance.
    """
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    current = lint_ratchet.snapshot(ROOT)
    if baseline["ruff_version"] != current["ruff_version"]:
        pytest.skip(
            f"ruff {current['ruff_version']} != baseline {baseline['ruff_version']}; "
            "the pin in pyproject.toml is what guarantees this match"
        )
    ref = os.environ.get(lint_ratchet.BASE_REF_ENV)
    allowed = lint_ratchet.tolerated_baseline(
        baseline, lint_ratchet.read_baseline_from_git(ref, ROOT) if ref else None
    )
    assert lint_ratchet.regression_problems(current, allowed) == []

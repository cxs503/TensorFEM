"""Tests for the ruff lint ratchet in ``scripts/lint_ratchet.py``.

The ratchet is what keeps the frozen ruff backlog from growing, so the
comparison logic and the shipped baseline are both under test here:

* the comparison must reject new violations, new rules, new files and new
  format debt while accepting (and reporting) debt that was paid down;
* the shipped ``scripts/lint_baseline.json`` must describe the tree it ships
  with, and the CLI must return 1 as soon as the tree grows past it.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lint_ratchet", ROOT / "scripts" / "lint_ratchet.py")
lint_ratchet = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault("lint_ratchet", lint_ratchet)
SPEC.loader.exec_module(lint_ratchet)

BASELINE_PATH = ROOT / "scripts" / "lint_baseline.json"


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


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
def test_shipped_baseline_describes_the_tree():
    """The committed baseline must match the tree it ships with."""
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    current = lint_ratchet.snapshot(ROOT)
    if baseline["ruff_version"] != current["ruff_version"]:
        pytest.skip(
            f"ruff {current['ruff_version']} != baseline {baseline['ruff_version']}; "
            "the pin in pyproject.toml is what guarantees this match"
        )
    assert lint_ratchet.regression_problems(current, baseline) == []

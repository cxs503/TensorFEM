"""Tests for the ruff debt ratchet (``scripts/lint_gate.py``).

Two layers:

* pure comparison logic - frozen counts, grown counts, new ``file -> rule``
  pairs, the unformatted-file set, and the progress side of the ledger;
* an end-to-end run against a throwaway tree with the real ``ruff`` binary, so
  "clean tree passes / one new violation fails / formatting a frozen file
  counts as progress" is demonstrated rather than asserted by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import lint_gate  # noqa: E402  (path bootstrap above)


needs_ruff = pytest.mark.skipif(
    __import__("shutil").which("ruff") is None, reason="ruff not installed"
)


# --------------------------------------------------------------------------- #
# pure comparison logic
# --------------------------------------------------------------------------- #
def test_identical_ledger_is_green() -> None:
    ledger = {"src/a.py": {"E702": 2}, "tests/b.py": {"F401": 1}}
    unformatted = ["src/a.py"]
    assert lint_gate.regressions(ledger, unformatted, ledger, unformatted) == []
    assert lint_gate.improvements(ledger, unformatted, ledger, unformatted) == []


def test_grown_count_is_new_debt() -> None:
    base = {"src/a.py": {"E702": 2}}
    now = {"src/a.py": {"E702": 5}}
    problems = lint_gate.regressions(base, [], now, [])
    assert problems == ["src/a.py: E702 2 -> 5"]


def test_new_rule_in_known_file_is_new_debt() -> None:
    base = {"src/a.py": {"E702": 2}}
    now = {"src/a.py": {"E702": 2, "F401": 1}}
    assert lint_gate.regressions(base, [], now, []) == ["src/a.py: F401 new (1x)"]


def test_violation_in_untouched_file_is_new_debt() -> None:
    base = {"src/a.py": {"E702": 2}}
    now = {"src/a.py": {"E702": 2}, "src/shipped.py": {"E501": 1}}
    assert lint_gate.regressions(base, [], now, []) == ["src/shipped.py: E501 new (1x)"]


def test_regressions_report_in_stable_order() -> None:
    base: dict[str, dict[str, int]] = {}
    now = {
        "tests/z.py": {"E702": 1},
        "src/a.py": {"E702": 1, "E701": 1},
    }
    problems = lint_gate.regressions(base, [], now, [])
    assert problems == [
        "src/a.py: E701 new (1x)",
        "src/a.py: E702 new (1x)",
        "tests/z.py: E702 new (1x)",
    ]


def test_unformatted_file_outside_frozen_set_is_new_debt() -> None:
    problems = lint_gate.regressions({}, ["src/legacy.py"], {}, ["src/legacy.py", "src/new.py"])
    assert problems == ["src/new.py: needs 'ruff format' (not in frozen set)"]


def test_debt_reduction_is_progress_not_failure() -> None:
    base = {"src/a.py": {"E702": 4, "F401": 1}}
    now = {"src/a.py": {"E702": 1}}
    assert lint_gate.regressions(base, ["src/a.py"], now, []) == []
    wins = lint_gate.improvements(base, ["src/a.py"], now, [])
    assert wins == ["src/a.py: E702 4 -> 1", "src/a.py: F401 1 -> 0", "src/a.py: formatted"]


def test_totals_and_rule_totals() -> None:
    ledger = {"src/a.py": {"E702": 2, "F401": 1}, "tests/b.py": {"E702": 3}}
    assert lint_gate.totals(ledger, ["src/a.py", "tests/b.py"]) == {
        "check": 6,
        "files_with_check_debt": 2,
        "unformatted": 2,
    }
    assert lint_gate.rule_totals(ledger) == {"E702": 5, "F401": 1}


def test_signature_is_order_independent_and_content_sensitive() -> None:
    one = {"src/b.py": {"E702": 1}, "src/a.py": {"F401": 2}}
    two = {"src/a.py": {"F401": 2}, "src/b.py": {"E702": 1}}
    other = {"src/a.py": {"F401": 3}}
    assert lint_gate.signature(one, ["src/a.py"]) == lint_gate.signature(two, ["src/a.py"])
    assert lint_gate.signature(one, ["src/a.py"]) != lint_gate.signature(other, ["src/a.py"])
    assert lint_gate.signature(one, ["src/a.py"]) != lint_gate.signature(one, [])


def test_load_baseline_errors(tmp_path: Path) -> None:
    with pytest.raises(lint_gate.RuffError):
        lint_gate.load_baseline(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(lint_gate.RuffError):
        lint_gate.load_baseline(bad)
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"schema": 99}), encoding="utf-8")
    with pytest.raises(lint_gate.RuffError):
        lint_gate.load_baseline(stale)


# --------------------------------------------------------------------------- #
# end-to-end against the real ruff binary
# --------------------------------------------------------------------------- #
def _seed_tree(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    # E702: statement separated by semicolon.
    (root / "pkg" / "packed.py").write_text("x = 1; y = 2\n", encoding="utf-8")
    # Formatted and clean: must never show up as debt.
    (root / "pkg" / "clean.py").write_text("VALUE = 1\n", encoding="utf-8")


@needs_ruff
def test_collect_reports_rule_ledger_and_unformatted_set(tmp_path: Path) -> None:
    _seed_tree(tmp_path)
    ruff = lint_gate.shutil.which("ruff")
    assert ruff is not None
    ledger = lint_gate.collect_check(ruff, ("pkg",), tmp_path)
    assert ledger == {"pkg/packed.py": {"E702": 1}}
    unformatted = lint_gate.collect_unformatted(ruff, ("pkg",), tmp_path)
    assert "pkg/packed.py" in unformatted
    assert "pkg/clean.py" not in unformatted


@needs_ruff
def test_gate_green_then_red_on_new_violation(tmp_path: Path) -> None:
    _seed_tree(tmp_path)
    baseline = tmp_path / "docs" / "lint-baseline.json"
    args = [
        "--root",
        str(tmp_path),
        "--baseline",
        str(baseline),
        "--scopes",
        "pkg",
    ]
    assert lint_gate.main([*args, "--update-baseline"]) == 0
    frozen = json.loads(baseline.read_text(encoding="utf-8"))
    assert frozen["totals"]["check"] == 1
    assert frozen["schema"] == lint_gate.SCHEMA

    # Same tree, nothing changed -> green.
    assert lint_gate.main(args) == 0

    # One new semicolon-packed statement in a file that had none -> red.
    (tmp_path / "pkg" / "fresh.py").write_text("a = 1; b = 2\n", encoding="utf-8")
    assert lint_gate.main(args) == 1

    # Re-freezing accepts the new state (deliberate, reviewable step).
    assert lint_gate.main([*args, "--update-baseline"]) == 0
    assert lint_gate.main(args) == 0


@needs_ruff
def test_gate_rejects_newly_unformatted_frozen_offender(tmp_path: Path) -> None:
    """Formatting regressions in already-clean files are also new debt."""
    (tmp_path / "pkg").mkdir()
    clean = tmp_path / "pkg" / "clean.py"
    clean.write_text("VALUE = 1\n", encoding="utf-8")
    baseline = tmp_path / "docs" / "lint-baseline.json"
    args = ["--root", str(tmp_path), "--baseline", str(baseline), "--scopes", "pkg"]
    assert lint_gate.main([*args, "--update-baseline"]) == 0
    clean.write_text("VALUE = {'a':1}\n", encoding="utf-8")  # needs ruff format
    assert lint_gate.main(args) == 1


@needs_ruff
def test_gate_write_report_payload(tmp_path: Path) -> None:
    _seed_tree(tmp_path)
    baseline = tmp_path / "docs" / "lint-baseline.json"
    report = tmp_path / "runs" / "lint-report.json"
    args = ["--root", str(tmp_path), "--baseline", str(baseline), "--scopes", "pkg"]
    assert lint_gate.main([*args, "--update-baseline"]) == 0
    assert lint_gate.main([*args, "--report", str(report)]) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert payload["regressions"] == []
    assert payload["rule_totals"] == {"E702": 1}


def test_step_summary_is_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    lint_gate.emit_step_summary("hello")  # no summary env -> silent no-op
    target = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
    lint_gate.emit_step_summary("hello")
    assert target.read_text(encoding="utf-8") == "hello\n"

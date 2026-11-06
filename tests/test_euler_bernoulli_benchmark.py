"""Gate on the committed Euler-Bernoulli / thin-plate / Boussinesq benchmark.

The heavy ladders live in ``benchmarks/euler_bernoulli/run.py`` (minutes of
half-space solves), so CI does not re-run them.  It instead

* re-derives every verdict from ``benchmarks/euler_bernoulli/results.json`` and
  cross-checks the numbers quoted in ``docs/FEM_VERIFY.md`` (``validate.py``);
* re-runs the *cheap* parts — the analytic beam reference, the frame ladder and
  the Navier series — so a regression in the formulas cannot ride along on a
  stale JSON file.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "euler_bernoulli"


def _load(name: str):
    """Import a benchmark script by path (``benchmarks/`` is not a package).

    The module has to be registered in ``sys.modules`` before execution, otherwise
    the ``@dataclass`` inside ``run.py`` cannot resolve its own module.
    """
    module_name = f"euler_bernoulli_{name}"
    spec = importlib.util.spec_from_file_location(module_name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return _load("run")


@pytest.fixture(scope="module")
def validator():
    return _load("validate")


def test_results_json_exists_and_verdicts_follow_raw_numbers(validator):
    checks = validator.validate()
    failures = [check for check in checks if not check.ok]
    assert not failures, [f"{check.name}: {check.detail}" for check in failures]


def test_documented_doc_present_with_convergence_tables():
    doc = ROOT / "docs" / "FEM_VERIFY.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "BEGIN CONVERGENCE TABLES" in text and "END CONVERGENCE TABLES" in text
    rows = [line for line in text.splitlines() if line.strip().startswith("|")]
    assert len(rows) >= 12


def test_beam_analytic_reference(bench):
    analytic = bench.beam_analytic()
    cfg = bench.BEAM
    expected = cfg["load"] * cfg["length"] ** 3 / (3 * cfg["young"] * cfg["width"] * cfg["height"] ** 3 / 12)
    assert analytic["tip_deflection"] == pytest.approx(expected, rel=1e-14)
    assert analytic["slenderness"] == pytest.approx(100.0)


def test_frame_ladder_reproduces_beam_theory(bench):
    rows = bench.beam_frame_levels(level_elements=(4, 8, 16))
    analytic = bench.beam_analytic()["tip_deflection"]
    for row in rows:
        assert row["tip_deflection"] == pytest.approx(analytic, rel=1e-10)
    assert [row["n_elements"] for row in rows] == [4, 8, 16]


def test_plate_navier_series_matches_textbook_coefficient(bench):
    reference = bench.plate_navier_series(terms=199)
    assert reference["centre_coefficient"] == pytest.approx(0.00406, rel=2e-3)


def test_committed_results_agree_with_formulas(bench):
    """The committed JSON must still satisfy the formulas it claims to grade."""
    results = json.loads((BENCH / "results.json").read_text(encoding="utf-8"))
    analytic = bench.beam_analytic()["tip_deflection"]
    assert results["beam"]["analytic"]["tip_deflection"] == pytest.approx(analytic, rel=1e-12)
    for row in results["beam"]["solid_levels"]:
        assert row["relative_error"] == pytest.approx(
            (row["tip_deflection"] - analytic) / analytic, rel=1e-9, abs=1e-12)
    reference = bench.plate_navier_series()["centre_deflection"]
    for row in results["plate"]["levels"]:
        assert row["relative_error"] == pytest.approx(
            (row["centre_deflection"] - reference) / reference, rel=1e-6, abs=1e-12)
    for row in results["halfspace"]["levels"]:
        assert row["relative_error"] == pytest.approx(
            (row["w_centre"] - row["w_centre_exact"]) / row["w_centre_exact"],
            rel=1e-9, abs=1e-12)
        assert row["reaction_balance_rel"] < 1e-6
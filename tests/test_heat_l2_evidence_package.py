"""Regression: the CASE-FEM-HEAT-L2 evidence package must exist in the tree.

The six-section report ``docs/CASE-FEM-HEAT-L2-steady-conduction.md`` quotes
numbers from ``artifacts/CASE-FEM-HEAT-L2/evidence.json``.  A previous revision
of the study committed the report but not the evidence package (the files lived
only on an unmerged branch), so the merged report pointed at files that were not
in the merged tree.  These tests fail closed if that debt returns:

* the evidence file and the tables file exist;
* the evidence verdict is green (all predeclared gates passed);
* the evidence covers the six-case pool at the full ladder;
* the gate constants recorded in the evidence equal the module constants
  (no silent relaxation of the predeclared acceptance);
* the headline rate of every non-control case sits in the predeclared band.

The test does NOT re-run the ladder (that is what
``benchmarks/heat_l2_convergence.py`` is for); it only binds the report to the
committed evidence.
"""
import json
from pathlib import Path

import pytest

from tensorfem import heat_l2_convergence as hlc

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "artifacts" / "CASE-FEM-HEAT-L2" / "evidence.json"
TABLES = REPO / "artifacts" / "CASE-FEM-HEAT-L2" / "tables.md"
REPORT = REPO / "docs" / "CASE-FEM-HEAT-L2-steady-conduction.md"


@pytest.fixture(scope="module")
def evidence() -> dict:
    assert EVIDENCE.is_file(), (
        "artifacts/CASE-FEM-HEAT-L2/evidence.json is missing from the tree: the "
        "merged report quotes it, so it must be committed together with the "
        "report. Regenerate with: PYTHONPATH=src python3 "
        "benchmarks/heat_l2_convergence.py")
    return json.loads(EVIDENCE.read_text())


def test_tables_file_exists():
    assert TABLES.is_file(), "artifacts/CASE-FEM-HEAT-L2/tables.md is missing"


def test_report_exists():
    assert REPORT.is_file(), "docs/CASE-FEM-HEAT-L2-steady-conduction.md is missing"


def test_evidence_verdict_is_green(evidence):
    verdict = evidence["verdict"]
    assert verdict["passed"] is True, f"failed gates: {verdict['failed']}"
    assert verdict["failed"] == []


def test_evidence_covers_the_six_case_pool(evidence):
    studies = evidence["studies"]
    expected = {"sine_generation", "exp_trig_harmonic", "cubic_harmonic",
                "sine_aspect2", "quadratic_harmonic", "bilinear_control"}
    assert set(studies) == expected, set(studies) ^ expected
    for key, study in studies.items():
        assert study["levels"][-1] == 64, (key, study["levels"])


def test_target_order_is_the_manual_two(evidence):
    assert float(evidence["target"]["order"]) == 2.0


def test_recorded_gates_equal_the_module_constants(evidence):
    """The evidence must carry the same predeclared gates as the module."""
    assert evidence["gates"]["l2_pair"] == hlc.GATE_L2_PAIR
    assert evidence["gates"]["l2_finest"] == list(hlc.GATE_L2_FINEST)
    assert evidence["gates"]["h1_pair"] == hlc.GATE_H1_PAIR
    assert evidence["gates"]["power_law"] == hlc.GATE_POWER_LAW
    assert evidence["gates"]["control_abs"] == hlc.GATE_CONTROL_ABS
    assert evidence["gates"]["field_rel"] == hlc.GATE_FIELD_REL
    assert evidence["gates"]["quadrature"] == hlc.GATE_QUADRATURE


def test_headline_rates_in_the_predeclared_band(evidence):
    """The acceptance is a rate: finest-pair L2 order in band, power law clean."""
    for key, study in evidence["studies"].items():
        if study["representable"]:
            continue
        finest = study["l2_orders"][-1]
        lo, hi = hlc.GATE_L2_FINEST
        assert lo <= finest <= hi, (key, finest)
        assert study["richardson_residual"]["theoretical"] <= hlc.GATE_POWER_LAW, key


def test_report_quotes_the_evidence(evidence):
    """The report must point at the evidence package it quotes."""
    text = REPORT.read_text()
    assert "evidence.json" in text, "report must point at the evidence package"
    assert "31" in text, "report no longer states the 31-check gate count"

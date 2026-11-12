"""The two refinement-ladder anchors must ship *recomputable* gate numbers.

Why this test exists
--------------------
``fem_pinched_cylinder`` and ``fem_shell_scordelis_lo`` grade themselves with
:func:`tensorfem.refinement_gate.evaluate_reference_ladder`.  The gates it
emitted were not machine-checkable:

* no gate carried a comparison ``operator``, so a reader could not decide
  whether ``value`` and ``threshold`` were to be compared with ``<=`` or ``>=``
  (the auditor could only *infer* it from prose);
* the monotonicity gate reported ``actual=None`` when it passed and
  ``actual=1.0`` when it failed, with ``threshold=None`` -- a pass with no
  criterion at all (the auditor flags this as ``E_PASS_WITHOUT_CRITERION``);
* the asymptotic-range gate wrote the GCI *band* into both ``actual`` and
  ``threshold``, so the gate reproduced its own threshold instead of reporting
  the measured finest-mesh error.

These tests pin the repaired contract from the shipped artifacts (no solver run:
the numbers come from ``result.json`` / ``gates_verified.json`` on disk).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfem.refinement_gate import FAIL, PASS, evaluate_reference_ladder

REPO = Path(__file__).resolve().parents[1]
VERIFIED = REPO / "benchmarks" / "verified"
ANCHORS = ("fem_pinched_cylinder", "fem_shell_scordelis_lo")

#: Gates that are allowed to carry no measurement, because the ladder itself
#: failed to support the question (blocked / withheld *accuracy* estimate).
MAY_BE_UNMEASURED = {"accuracy_asymptotic"}


def _gates(anchor: str) -> list[dict]:
    payload = json.loads((VERIFIED / anchor / "gates_verified.json").read_text())
    return payload["gates"]


@pytest.mark.parametrize("anchor", ANCHORS)
def test_every_measured_gate_declares_an_operator(anchor):
    for gate in _gates(anchor):
        bare_id = gate["id"].split("_", 1)[1] if gate["id"][:1] == "G" else gate["id"]
        if gate["actual"] is None:
            continue
        assert gate.get("operator") in ("<=", ">=", "<", ">"), gate["id"]
        assert bare_id  # sanity: id parsing above must keep something


@pytest.mark.parametrize("anchor", ANCHORS)
def test_monotonicity_gate_has_a_numeric_criterion(anchor):
    for gate in _gates(anchor):
        if not gate["id"].endswith("monotone_qoi"):
            continue
        assert gate["threshold"] == 0.0, gate
        assert gate["actual"] is not None and math.isfinite(float(gate["actual"])), gate
        # count of offending ladder steps: PASS <=> zero of them
        assert (gate["status"] == PASS) == (float(gate["actual"]) <= 0.0), gate


@pytest.mark.parametrize("anchor", ANCHORS)
def test_asymptotic_band_gate_reports_the_measured_error(anchor):
    """``actual`` must be the measured finest-mesh error, not a copy of the band."""
    result = json.loads((VERIFIED / anchor / "result.json").read_text())
    gates_payload = json.loads((VERIFIED / anchor / "gates_verified.json").read_text())
    certified = gates_payload.get("reported_vs_certified")
    for gate in _gates(anchor):
        if not gate["id"].endswith("asymptotic_range_consistency"):
            continue
        assert gate["actual"] is not None and gate["threshold"] is not None
        assert gate["actual"] != gate["threshold"], f"{gate['id']}: value copies its threshold"
        assert gate["operator"] == "<="
        assert (gate["status"] == PASS) == (float(gate["actual"]) <= float(gate["threshold"]))
    assert result  # the anchor still ships its numeric result artifact
    assert certified  # ... and still separates reported from certified accuracy


def test_scordelis_flat_facet_band_gate_matches_its_own_detail():
    """The flat-facet ladder is pre-asymptotic: 49.4 % error vs a 3.17 % band."""
    for gate in _gates("fem_shell_scordelis_lo"):
        if gate["id"] != "flat_facet.asymptotic_range_consistency":
            continue
        assert float(gate["actual"]) == pytest.approx(0.4940176798061, rel=1e-9)
        assert float(gate["threshold"]) < 0.05
        assert gate["status"] != PASS
        assert "outside the band" in gate["detail"]
        return
    pytest.fail("flat-facet asymptotic-range gate disappeared")


def test_monotone_ladder_yields_zero_violations():
    """Synthetic ladder of known order: monotone, so the gate must certify 0."""
    levels = [(1.0 / n, 1.0 + 0.1 * (1.0 / n) ** 2) for n in (4, 8, 16, 32)]
    report = evaluate_reference_ladder(levels, reference_value=1.0, tolerance=0.01)
    gate = next(g for g in report["gates"] if g["id"] == "monotone_qoi")
    assert gate["status"] == PASS and gate["actual"] == 0.0 and gate["threshold"] == 0.0


def test_non_monotone_ladder_counts_the_offending_steps():
    levels = [(1.0 / 4, 1.10), (1.0 / 8, 0.90), (1.0 / 16, 1.02), (1.0 / 32, 1.01)]
    report = evaluate_reference_ladder(levels, reference_value=1.0, tolerance=0.10)
    gate = next(g for g in report["gates"] if g["id"] == "monotone_qoi")
    assert gate["status"] == FAIL
    assert gate["actual"] == 2.0 and gate["threshold"] == 0.0 and gate["operator"] == "<="

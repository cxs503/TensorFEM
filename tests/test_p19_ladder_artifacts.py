"""Artifact gates for the P19-N01 FEM verification ladder.

The ladder ships machine-readable evidence:

* ``benchmarks/verified/fem_beam_static/{result,gates_verified}.json``  -- L1
* ``benchmarks/verified/fem_plate_static/{result,gates_verified}.json`` -- L2
* ``benchmarks/verified/fem_modal/{result_p19,gates_verified}.json``    -- L3

These tests re-read the committed JSON and enforce the campaign contract
without re-running the solvers: 23 graded data points, every one under its
tolerance, three refinement levels ``N = 10/20/40``, a documented convergence
order, and no silently relaxed gate.  The legacy ``fem_modal/result.json``
(cantilever rung of ``benchmarks/structural_ladder.py``) must stay untouched so
the older audit chain keeps validating.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

VERIFIED = Path(__file__).resolve().parents[1] / "benchmarks" / "verified"

ANCHORS = {
    "fem_beam_static": ("result.json", 5.0e-3, 7),
    "fem_plate_static": ("result.json", 2.0e-2, 6),
    "fem_modal": ("result_p19.json", 3.0e-2, 10),
}


def load(anchor: str) -> tuple[dict, dict]:
    name, _, _ = ANCHORS[anchor]
    result = json.loads((VERIFIED / anchor / name).read_text())
    gates = json.loads((VERIFIED / anchor / "gates_verified.json").read_text())
    return result, gates


@pytest.mark.parametrize("anchor", sorted(ANCHORS))
def test_anchor_passes_every_gate(anchor: str):
    result, gates = load(anchor)
    _, tolerance, expected_points = ANCHORS[anchor]
    assert result["passed"] is True
    assert gates["passed"] is True, gates["failed_gates"]
    assert gates["schema"] == "tensorfem.verified-gates.v1"
    assert gates["anchor"]
    assert len(result["data_points"]) == expected_points
    for gate in gates["gates"]:
        assert gate["passed"], gate
        assert isinstance(gate["value"], float)
    for point in result["data_points"]:
        assert point["passed"], point
        assert point["relative_error"] <= point["threshold"]
        assert point["threshold"] <= tolerance + 1e-12


@pytest.mark.parametrize("anchor", sorted(ANCHORS))
def test_anchor_has_three_level_refinement_ladder(anchor: str):
    result, _ = load(anchor)
    assert result["parameters"]
    levels = result["parameters"].get("levels") or result["parameters"].get("levels_elements")
    if levels is None:  # beam driver stores the list under a different key
        levels = result["parameters"]["levels_elements"]
    assert list(levels)[:3] == [10, 20, 40]
    assert result["convergence"]


def test_total_graded_points_at_least_fifteen():
    total = sum(len(load(anchor)[0]["data_points"]) for anchor in ANCHORS)
    assert total >= 15


def test_plate_anchor_records_navier_truncation_and_table_floor():
    result, _ = load("fem_plate_static")
    assert result["parameters"]["navier_terms"] == 11
    assert result["parameters"]["navier_terms_converged"] >= 100
    reference = result["reference"]
    # the mandated 11x11 truncation is itself within 1e-4 of the converged series
    relative = abs(
        reference["navier_11_centre_deflection_m"]
        - reference["navier_301_centre_deflection_m"]
    ) / reference["navier_301_centre_deflection_m"]
    assert relative < 1e-4
    # the N=80 clamped level sits on the 0.00126 rounding floor of the table
    extended = result["extended_levels"]["cccc"][0]
    assert abs(extended["alpha"] - 0.00126) / 0.00126 < 5e-3


def test_modal_anchor_keeps_legacy_cantilever_result_intact():
    legacy = json.loads((VERIFIED / "fem_modal" / "result.json").read_text())
    assert legacy["generated_by"] == "benchmarks/structural_ladder.py"
    assert legacy["verdict"] == "PASS"
    current = json.loads((VERIFIED / "fem_modal" / "result_p19.json").read_text())
    assert current["generated_by"] == "benchmarks/verified/fem_modal/run.py"


def test_convergence_orders_meet_campaign_thresholds():
    beam, _ = load("fem_beam_static")
    assert beam["convergence"]["field_l2_order"] >= 2.0
    plate, _ = load("fem_plate_static")
    assert plate["convergence"]["order_ssss_deflection"] >= 1.5
    assert plate["convergence"]["order_cccc_self"] >= 1.5
    modal, _ = load("fem_modal")
    assert modal["convergence"]["beam_order_min"] >= 2.0
    assert modal["convergence"]["plate_order_min"] >= 1.5
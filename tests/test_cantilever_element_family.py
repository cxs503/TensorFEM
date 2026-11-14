"""CASE-CANTILEVER-TIP element-family ladder -- gates and fail-closed checks.

Layers gated here:

* the two oracles against an *independent* arithmetic statement of themselves
  (the symbolic EB/TI separation ``E/(4 kappa G) (H/L)^2`` and the quadrature
  residual), so a break in the driver's closed forms is caught;
* the family ranking on a fast sub-ladder (HEX20 inside the anchor band, the
  fully integrated brick far outside it);
* the pre-registered expectation that turned out false is *recorded*, not
  deleted: the through-depth refinement ratio must stay near 1 in the artifact;
* the generated artifact itself: every gate in ``gates_verified.json`` passes,
  its ``passed`` flag is recomputable from ``value``/``threshold``, and the
  values agree with ``result.json``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks" / "verified" / "fem_cantilever_hex8"
for extra in (REPO / "src", BENCH):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import run as driver  # noqa: E402


# --------------------------------------------------------------------------- #
# Oracles
# --------------------------------------------------------------------------- #
def test_oracle_matches_unit_load_quadrature():
    for L_over_h in (100.0, 20.0, 10.0):
        residual = driver.oracle_residual(L_over_h * driver.H)
        assert residual["relative_residual"] < 1e-6
        # bending and shear terms are both positive and the shear share grows
        # as (H/L)^2 -- the reason the thick rung is the discriminating one
        assert residual["quadrature_bending_term"] > 0.0
        assert residual["quadrature_shear_term"] > 0.0


def test_eb_ti_separation_matches_closed_form():
    """``(TI - EB)/EB = E/(4 kappa G) (H/L)^2`` against the driver's oracle."""
    for L_over_h in (100.0, 20.0, 10.0):
        L = L_over_h * driver.H
        expected = (driver.E / (4.0 * driver.KAPPA * driver.G)) * (driver.H / L) ** 2
        got = (driver.delta_ti(L) - driver.delta_eb(L)) / driver.delta_eb(L)
        assert got == pytest.approx(expected, rel=1e-12)


def test_deflection_profile_ends_at_the_tip_oracle():
    L = 10.0 * driver.H
    profile = driver.deflection_profile([L], L)[0]
    assert profile == pytest.approx(driver.delta_ti(L), rel=1e-12)
    assert driver.deflection_profile([0.0], L)[0] == 0.0


# --------------------------------------------------------------------------- #
# Family ranking on a fast sub-ladder
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def quick_ladder():
    return driver.run_ladder(rungs=(100.0,), quick=True)["records"]


def test_quadratic_solid_meets_the_anchor_criterion(quick_ladder):
    hex20 = [r for r in quick_ladder if r["family"] == "hex20"]
    assert hex20, "the quick ladder must contain a HEX20 rung"
    assert min(r["relative_error_vs_timoshenko"] for r in hex20) <= driver.TOLERANCE


def test_first_order_brick_is_far_outside_the_criterion(quick_ladder):
    hex8 = [r for r in quick_ladder if r["family"] == "hex8"]
    assert hex8, "the quick ladder must contain a HEX8 rung"
    best_hex8 = min(r["relative_error_vs_timoshenko"] for r in hex8)
    best_hex20 = min(r["relative_error_vs_timoshenko"]
                     for r in quick_ladder if r["family"] == "hex20")
    # an order of magnitude at least: this is the whole point of the ladder
    assert best_hex8 > 10.0 * best_hex20
    assert not any(r["criterion_3pct_met"] for r in hex8)


def test_tet4_is_not_better_than_its_parent_brick(quick_ladder):
    tet = [r for r in quick_ladder if r["family"] == "tet4"]
    hex8 = [r for r in quick_ladder if r["family"] == "hex8"]
    if not tet or not hex8:
        pytest.skip("quick ladder carries no TET4 record")
    assert (min(r["relative_error_vs_timoshenko"] for r in tet)
            >= min(r["relative_error_vs_timoshenko"] for r in hex8))


def test_load_resultant_is_exactly_p():
    """The end-face traction is re-summed from the applied vector."""
    import torch

    nodes, _ = driver.structured_hex_mesh(0.1, driver.B, driver.H, 4, 2, 2)
    forces = driver._tip_load(nodes, 0.1)
    assert float(forces.sum()) == pytest.approx(driver.P, rel=1e-15)
    assert float(forces[2::3].sum()) == pytest.approx(driver.P, rel=1e-15)
    assert torch.count_nonzero(forces[0::3]) == 0
    assert torch.count_nonzero(forces[1::3]) == 0


# --------------------------------------------------------------------------- #
# Generated artifact
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def artifacts():
    result_path = BENCH / "result.json"
    gates_path = BENCH / "gates_verified.json"
    if not (result_path.is_file() and gates_path.is_file()):
        pytest.skip("run benchmarks/verified/fem_cantilever_hex8/run.py first")
    return json.loads(result_path.read_text()), json.loads(gates_path.read_text())


def test_every_gate_passes_and_is_recomputable(artifacts):
    _, gates_doc = artifacts
    for gate in gates_doc["gates"]:
        value, threshold, operator = gate["value"], gate["threshold"], gate["operator"]
        recomputed = (value <= threshold) if operator == "<=" else (value >= threshold)
        assert gate["passed"] is recomputed, gate["id"]
        assert gate["passed"], f"failed gate {gate['id']}: {value} {operator} {threshold}"
    assert gates_doc["passed"] is True
    assert gates_doc["failed_gates"] == []


def test_gate_values_agree_with_the_records(artifacts):
    result, gates_doc = artifacts
    by_id = {g["id"]: g["value"] for g in gates_doc["gates"]}
    rows = result["records"]
    l100 = [r for r in rows if r["L_over_H"] == 100.0]
    hex20_best = min(r["relative_error_vs_timoshenko"]
                     for r in l100 if r["family"] == "hex20")
    hex8 = sorted((r for r in l100 if r["family"] == "hex8"), key=lambda r: r["n_dofs"])
    assert by_id["hex20_anchor_rung_within_3pct"] == pytest.approx(hex20_best, rel=1e-12)
    assert by_id["first_order_brick_fails_criterion"] == pytest.approx(
        hex8[0]["relative_error_vs_timoshenko"], rel=1e-12)
    assert by_id["brick_through_depth_refinement_is_flat"] == pytest.approx(
        hex8[-1]["relative_error_vs_timoshenko"]
        / hex8[0]["relative_error_vs_timoshenko"], rel=1e-12)


def test_falsified_expectation_is_recorded_not_hidden(artifacts):
    result, _ = artifacts
    items = result["falsified_expectations"]
    assert items, "the falsified pre-registration must stay on the record"
    ratio = items[0]["observed_ratio"]
    assert 0.9 < ratio < 1.1, "through-depth refinement was flat; the ratio must say so"
    assert "falsified" in items[0]["verdict"]


def test_aspect_sweep_decreases_monotonically(artifacts):
    result, _ = artifacts
    aspect = result["aspect_sweep"]
    assert [r["mesh"][0] for r in aspect] == sorted(r["mesh"][0] for r in aspect)
    errs = [r["relative_error_vs_timoshenko"] for r in aspect]
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
    order = result["aspect_order"]
    assert math.isfinite(order["observed_order"]) and order["observed_order"] > 0.0
    # the extrapolation must be a statement about a mesh denser than any solved
    assert order["dx_over_H_needed_for_tolerance"] < aspect[-1]["dx_over_H"]
    assert order["nx_needed_for_tolerance_per_H"] > 1.0
"""Fail-closed gates for the Saint-Venant torsion verification.

The gate set is deliberately asymmetric: the fine HEX8 ring must reach the 3 %
band while the coarse rings must keep failing, so a mesh refinement cannot be
silently replaced by a looser tolerance.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorfem.torsion_reference import (
    annular_hex_mesh,
    faceted_torsional_constant,
    faceting_ratio,
    project_hex20_to_cylinders,
    shear_modulus,
    solve_torsion_hex8,
    solve_torsion_hex20,
    surface_shear_stress,
    tube_torsional_constant,
)
from tensorfem.hex20 import upgrade_hex8_to_hex20

INNER, OUTER, LENGTH, TWIST = 0.05, 0.10, 0.4, 0.01
YOUNG, POISSON = 210.0e9, 0.3
GATE = 0.03
ROOT = Path(__file__).resolve().parents[1]


def torsional(summary, shear):
    return summary["torque"] / (shear * TWIST)


@pytest.fixture(scope="module")
def shear():
    return shear_modulus(YOUNG, POISSON)


@pytest.fixture(scope="module")
def hex8_fine(shear):
    return solve_torsion_hex8(INNER, OUTER, LENGTH, TWIST, YOUNG, POISSON, 32, 4, 2)


@pytest.fixture(scope="module")
def hex8_mid(shear):
    return solve_torsion_hex8(INNER, OUTER, LENGTH, TWIST, YOUNG, POISSON, 16, 4, 2)


def test_references_are_analytically_consistent():
    circle = tube_torsional_constant(INNER, OUTER)
    assert circle == pytest.approx(math.pi * (OUTER**4 - INNER**4) / 2.0, rel=1e-15)
    # inscribed polygon converges to the circle from below, and the deficit is 2nd order
    previous = 0.0
    for n in (8, 16, 32, 64, 128, 256):
        faceted = faceted_torsional_constant(INNER, OUTER, n)
        assert faceted < circle
        assert faceted > previous
        previous = faceted
        assert faceting_ratio(n) == pytest.approx(faceted / circle, rel=1e-14)
    assert faceting_ratio(256) == pytest.approx(1.0, abs=3e-4)
    # faceting is second order: quadrupling n cuts the deficit by ~16
    deficit_16 = 1.0 - faceting_ratio(16)
    deficit_32 = 1.0 - faceting_ratio(32)
    assert deficit_16 / deficit_32 == pytest.approx(4.0, rel=0.05)
    # thin-wall limit of the polar moment of a 4-gon is a^4 / 6 with a = sqrt(2) R
    assert faceted_torsional_constant(0.0, 1.0, 4) == pytest.approx(2.0 / 3.0, rel=1e-14)


def test_surface_stress_scales_linearly_with_radius(shear):
    inner = surface_shear_stress(YOUNG, POISSON, TWIST, INNER)
    outer = surface_shear_stress(YOUNG, POISSON, TWIST, OUTER)
    assert outer / inner == pytest.approx(OUTER / INNER, rel=1e-14)
    assert outer == pytest.approx(shear * TWIST * OUTER, rel=1e-15)


def test_hex8_mesh_is_watertight_and_periodic():
    nodes, cells = annular_hex_mesh(INNER, OUTER, LENGTH, 8, 2, 2)
    assert nodes.shape == (8 * 3 * 3, 3)
    # every element must have four in-plane corners at the two radii of its layer
    corners = nodes[cells.long()]
    radius = torch.linalg.vector_norm(corners[:, :, :2], dim=2)
    mid = INNER + (OUTER - INNER) / 2.0
    for element in radius:
        assert float(element.max() - element.min()) > 0.0
        assert float(element.min()) == pytest.approx(INNER, rel=1e-12) or float(
            element.min()
        ) == pytest.approx(mid, rel=1e-12)


def test_hex8_patch_and_stress_are_exact(hex8_fine, hex8_mid, shear):
    for summary in (hex8_fine, hex8_mid):
        assert summary["displacement_patch_error"] < 1e-12
        radius = torch.linalg.vector_norm(summary["probe"][:, :2], dim=1)
        exact = surface_shear_stress(YOUNG, POISSON, TWIST, radius)
        error = summary["tau_probe"] / exact - 1.0
        assert float(error.abs().max()) < 1e-10
        # the twist ramp is reproduced level by level
        assert float(
            (summary["twist_angle"] - TWIST * summary["twist_levels"]).abs().max()
        ) < 1e-12


def test_fine_hex8_ring_passes_the_three_percent_gate(hex8_fine, hex8_mid, shear):
    circle = tube_torsional_constant(INNER, OUTER)
    fine = torsional(hex8_fine, shear) / circle - 1.0
    assert abs(fine) < GATE, f"HEX8 n_theta=32 error {fine:.5f} exceeds the gate"
    # the whole residual is geometric: the faceted reference is met to round-off
    for summary, n_theta in ((hex8_fine, 32), (hex8_mid, 16)):
        faceted = faceted_torsional_constant(INNER, OUTER, n_theta)
        assert abs(torsional(summary, shear) / faceted - 1.0) < 1e-9
        # and the error is the predicted faceting law
        assert torsional(summary, shear) / circle - 1.0 == pytest.approx(
            faceting_ratio(n_theta) - 1.0, rel=1e-6, abs=1e-9
        )


def test_coarse_failures_are_retained_and_ordered():
    circle = tube_torsional_constant(INNER, OUTER)
    shear = shear_modulus(YOUNG, POISSON)
    errors = {}
    for n_theta in (8, 16, 32):
        summary = solve_torsion_hex8(INNER, OUTER, LENGTH, TWIST, YOUNG, POISSON, n_theta, 4, 2)
        errors[n_theta] = torsional(summary, shear) / circle - 1.0
    assert errors[8] < -GATE and errors[16] < -GATE, "coarse HEX8 rings must not pass"
    assert errors[8] < errors[16] < errors[32] < 0.0
    order = math.log(abs(errors[16]) / abs(errors[32])) / math.log(2.0)
    assert order == pytest.approx(2.0, abs=0.15), f"observed faceting order {order:.3f}"


def test_one_point_stress_quadrature_is_a_documented_trap():
    """The stress torque must not be used as the verdict: int r^2 dA != A r_c^2."""
    shear = shear_modulus(YOUNG, POISSON)
    coarse = solve_torsion_hex8(INNER, OUTER, LENGTH, TWIST, YOUNG, POISSON, 8, 4, 2)
    fine = solve_torsion_hex8(INNER, OUTER, LENGTH, TWIST, YOUNG, POISSON, 32, 4, 2)
    coarse_bias = coarse["stress_torque_one_point"] / coarse["torque"] - 1.0
    fine_bias = fine["stress_torque_one_point"] / fine["torque"] - 1.0
    assert coarse_bias < -0.01  # 5.8 % under-report at n_theta = 8
    assert abs(fine_bias) < abs(coarse_bias)


def test_hex20_faceted_matches_hex8_and_curved_edges_recover_the_circle():
    shear = shear_modulus(YOUNG, POISSON)
    circle = tube_torsional_constant(INNER, OUTER)
    faceted = solve_torsion_hex20(
        INNER, OUTER, LENGTH, TWIST, YOUNG, POISSON, 16, 4, 1, curved=False
    )
    hex8 = solve_torsion_hex8(INNER, OUTER, LENGTH, TWIST, YOUNG, POISSON, 16, 4, 2)
    assert torsional(faceted, shear) == pytest.approx(torsional(hex8, shear), rel=1e-10)
    assert abs(torsional(faceted, shear) / circle - 1.0) > GATE

    curved = solve_torsion_hex20(
        INNER, OUTER, LENGTH, TWIST, YOUNG, POISSON, 8, 2, 1, curved=True
    )
    error = torsional(curved, shear) / circle - 1.0
    assert abs(error) < GATE
    assert abs(error) < 0.02 * abs(faceted_torsional_constant(INNER, OUTER, 8) / circle - 1.0)


def test_projection_only_moves_surface_mid_edge_nodes():
    nodes, cells = annular_hex_mesh(INNER, OUTER, LENGTH, 16, 4, 2)
    hex20_nodes, hex20_cells = upgrade_hex8_to_hex20(nodes, cells)
    projected = project_hex20_to_cylinders(hex20_nodes, INNER, OUTER, hex20_cells)
    moved = (projected - hex20_nodes).abs().max(dim=1).values > 1e-14
    radius = torch.linalg.vector_norm(hex20_nodes[:, :2], dim=1)
    # only nodes strictly inside a cylindrical surface can move, and only outwards
    assert bool(moved.any())
    # chord midpoints sit just inside the cylinder they are projected onto
    assert (radius[moved] < OUTER + 1e-12).all()
    assert (radius[moved] > INNER * math.cos(math.pi / 16) - 1e-12).all()
    assert float(torch.linalg.vector_norm(projected[moved][:, :2], dim=1).sub(
        torch.where(
            (radius[moved] - INNER).abs() < (radius[moved] - OUTER).abs(),
            torch.full_like(radius[moved], INNER),
            torch.full_like(radius[moved], OUTER),
        )
    ).abs().max()) < 1e-12
    # idempotent
    again = project_hex20_to_cylinders(projected, INNER, OUTER, hex20_cells)
    assert float((again - projected).abs().max()) < 1e-15


def test_evidence_artifact_is_consistent_with_the_gate():
    evidence_path = ROOT / "benchmarks" / "torsion_tube_verification_evidence.json"
    if not evidence_path.exists():
        pytest.skip("run benchmarks/torsion_tube_verification.py to publish evidence")
    evidence = json.loads(evidence_path.read_text())
    circle = evidence["references"]["torsional_constant_circle"]
    assert evidence["verdicts"]["hex8_min_qualified_n_theta"] == 24
    for case in evidence["hex8_ladder"]:
        expected = abs(case["torsional_constant"] / circle - 1.0) <= GATE
        assert case["passes_gate"] is expected
    assert evidence["orders"]["hex8"]["p_16_32"] == pytest.approx(2.0, abs=0.15)
    assert evidence["orders"]["hex20"]["p_16_32"] > 3.0
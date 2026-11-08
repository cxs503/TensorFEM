"""Gates for the TET10 integration-accuracy audit.

These tests lock the numbers reported in ``docs/tet10-integration-accuracy.md`` and, more
importantly, lock the *method*: the Duffy reference is verified against closed-form
monomial integrals before it is allowed to judge the shipped rule, the shipped rule is
shown to agree with the reference to round-off where it must (affine geometry, where the
integrand is constant), and only then is the curved-geometry deficiency quantified.

Fail-closed intent: if someone changes the shipped rule, the curvature geometry or the
audit reference, the recorded error windows break instead of silently re-baselining.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorfem.quadratic_solid import tet10_quadrature, tet10_stiffness
from tensorfem.solid3d import elasticity_matrix_3d
from tensorfem.tet10_integration_audit import (
    KEAST_POINTS,
    KEAST_WEIGHTS,
    duffy_rule,
    duffy_self_verification,
    elasticity_matrix,
    element_mass,
    element_stiffness,
    integration_sweep,
    mass_matrix_audit,
    monomial_integral,
    relative_frobenius,
    strain_energy_audit,
    tet10_geometry,
    tet10_shape,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)  # shared host: thread oversubscription dominates runtime

YOUNG = 200.0e9
POISSON = 0.3
RHO = 7850.0
GATE = 0.03
JSON_PATH = (Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "tutorials"
             / "tet10-integration-accuracy.json")


def _d():
    return elasticity_matrix(YOUNG, POISSON)


def _evidence() -> dict:
    return json.loads(JSON_PATH.read_text(encoding="utf8"))


# --------------------------------------------------------------------------------------
# 1. The reference must verify itself before it judges anything.
# --------------------------------------------------------------------------------------
def test_duffy_rule_reproduces_closed_form_monomials():
    """Collapsed-cube rule: exact monomials and exact reference volume from order 6."""
    for order in (6, 8, 16):
        points, weights = duffy_rule(order)
        assert points.shape == (order ** 3, 3)
        assert weights.sum() == pytest.approx(monomial_integral((0, 0, 0)), rel=1e-14)
        for a in range(4):
            for b in range(4 - a):
                for c in range(4 - a - b):
                    reference = monomial_integral((a, b, c))
                    measured = float(np.sum(weights * points[:, 0] ** a * points[:, 1] ** b
                                            * points[:, 2] ** c))
                    if reference == 0.0:
                        assert abs(measured) < 1e-16
                    else:
                        assert measured == pytest.approx(reference, rel=1e-13)
    records = {r["order"]: r for r in duffy_self_verification()}
    assert records[6]["exact_through_total_degree"] == 6
    assert records[16]["worst_monomial_relative_error"] < 1e-14


def test_duffy_rule_rejects_nonsense_order():
    with pytest.raises(ValueError):
        duffy_rule(0)


def test_audit_keast_copy_equals_shipped_rule():
    """The locally restated rule must be the production rule, not a cousin of it."""
    points, weights = tet10_quadrature()
    assert np.max(np.abs(points.numpy() - KEAST_POINTS)) == 0.0
    assert np.max(np.abs(weights.numpy() - KEAST_WEIGHTS)) == 0.0
    assert weights.numpy().sum() == pytest.approx(1.0 / 6.0, rel=1e-15)


def test_shape_functions_are_a_partition_of_unity_with_zero_gradient_sum():
    rng = np.random.default_rng(7)
    rst = rng.random((20, 3)) * 0.4
    rst = rst[rst.sum(axis=1) < 2.0]
    n, dn = tet10_shape(rst)
    assert np.allclose(n.sum(axis=1), 1.0, atol=1e-14)
    assert np.allclose(dn.sum(axis=1), 0.0, atol=1e-13)


# --------------------------------------------------------------------------------------
# 2. Cross-implementation agreement (torch production vs independent NumPy oracle).
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind,curvature", [
    ("regular", 0.0), ("distorted", 0.0), ("curved", 0.05), ("curved", 0.11), ("warped", 0.02)
])
def test_production_stiffness_matches_independent_oracle(kind, curvature):
    nodes = tet10_geometry(kind, curvature)
    d3 = elasticity_matrix_3d(torch.tensor([YOUNG]), torch.tensor([POISSON]))
    k_prod, _ = tet10_stiffness(torch.tensor(nodes[None]), d3)
    k_oracle = element_stiffness(nodes, _d(), KEAST_POINTS, KEAST_WEIGHTS)
    assert relative_frobenius(k_prod[0].numpy(), k_oracle) < 1e-12


def test_geometry_curvature_moves_only_the_mid_edge_nodes():
    straight = tet10_geometry("curved", 0.0)
    curved = tet10_geometry("curved", 0.06)
    assert np.allclose(straight[:4], curved[:4])
    assert np.allclose(straight[:4], tet10_geometry("distorted")[:4])
    assert np.linalg.norm(curved[4:] - straight[4:], axis=1).max() > 0.0


# --------------------------------------------------------------------------------------
# 3. Where the shipped rule is exact, it must be exact — otherwise the audit is broken.
# --------------------------------------------------------------------------------------
def test_shipped_rule_is_exact_on_straight_edged_geometry():
    """Affine geometry: the integrand is constant, so Keast must hit the reference."""
    d = _d()
    points, weights = duffy_rule(16)
    for kind in ("regular", "distorted"):
        nodes = tet10_geometry(kind)
        k_keast = element_stiffness(nodes, d, KEAST_POINTS, KEAST_WEIGHTS)
        k_reference = element_stiffness(nodes, d, points, weights)
        assert relative_frobenius(k_keast, k_reference) < 1e-12


def test_cantilever_benchmark_mesh_is_affine_so_quadrature_is_not_its_error_source():
    """The shipped TET10 benchmark upgrades a structured HEX mesh: straight edges only."""
    from tensorfem.quadratic_solid import tet4_to_tet10
    from tensorfem.solid3d import hex_to_tet_mesh, structured_hex_mesh

    nodes, hexes = structured_hex_mesh(10.0, 1.0, 1.0, 4, 1, 1, dtype=torch.float64)
    _, tet4 = hex_to_tet_mesh(nodes, hexes)
    nodes10, cells = tet4_to_tet10(nodes, tet4)
    points, weights = duffy_rule(12)
    d = _d()
    for cell in cells[:4].tolist():
        element = nodes10[cell].numpy()
        assert relative_frobenius(element_stiffness(element, d, KEAST_POINTS, KEAST_WEIGHTS),
                                  element_stiffness(element, d, points, weights)) < 1e-12


# --------------------------------------------------------------------------------------
# 4. The measured deficiency on curved geometry (the verification debt itself).
# --------------------------------------------------------------------------------------
def test_curved_geometry_error_matches_recorded_evidence():
    """Recompute the two headline curvatures and compare with the committed JSON."""
    evidence = _evidence()["curved_sweep"]
    recomputed = {round(r["curvature"], 3): r for r in
                  integration_sweep(kind="curved", curvatures=(0.05, 0.11))}
    for curvature in (0.05, 0.11):
        recorded = next(r for r in evidence if abs(r["curvature"] - curvature) < 1e-12)
        assert recomputed[curvature]["stiffness_keast_vs_reference"] == pytest.approx(
            recorded["stiffness_keast_vs_reference"], rel=1e-9
        )


def test_curved_error_windows_and_three_percent_gate_crossing():
    rows = integration_sweep(kind="curved", curvatures=(0.05, 0.11, 0.15))
    error_005, error_011, error_015 = (r["stiffness_keast_vs_reference"] for r in rows)
    # Documented: sub-3 % at moderate curvature, but far from round-off ...
    assert 3.0e-3 < error_005 < 6.0e-3
    assert 2.0e-2 < error_011 < 3.2e-2
    # ... and past the acceptance gate once the curvature is heavy.
    assert error_015 > GATE
    assert error_005 < error_011 < error_015


def test_random_warp_confirms_the_error_is_not_a_hand_picked_pattern():
    rows = integration_sweep(kind="warped", curvatures=(0.02, 0.05))
    assert 5.0e-4 < rows[0]["stiffness_keast_vs_reference"] < 5.0e-3
    assert 5.0e-3 < rows[1]["stiffness_keast_vs_reference"] < 2.0e-2


def test_affine_energy_is_exact_so_the_error_is_the_b_matrix_not_the_jacobian():
    """detJ integration is fine even on curved elements; ``B`` variation is the culprit."""
    rows = strain_energy_audit()
    for row in rows:
        assert row["affine_energy_relative_error"] < 1e-12
    bending = {round(r["curvature"], 3): r["bending_energy_relative_error"] for r in rows}
    assert 2.0e-2 < bending[0.11] < 3.2e-2
    assert bending[0.15] > GATE


def test_shipped_rule_cannot_integrate_the_consistent_mass_matrix():
    """Quartic integrand vs degree-three rule: 19.7 % Frobenius on a straight element."""
    rows = mass_matrix_audit(curvatures=(0.0, 0.11))
    straight, curved = rows
    assert straight["relative_frobenius_error"] == pytest.approx(0.19677, abs=5e-4)
    assert straight["diagonal_relative_error_max"] > 0.5
    assert curved["relative_frobenius_error"] > straight["relative_frobenius_error"]
    # Total mass is still exact: the rule integrates detJ, only the N_i N_j weights fail.
    assert straight["total_mass_relative_error"] < 1e-12


# --------------------------------------------------------------------------------------
# 5. The opt-in high-order hook in production code.
# --------------------------------------------------------------------------------------
def test_opt_in_rule_leaves_the_straight_element_unchanged():
    d3 = elasticity_matrix_3d(torch.tensor([YOUNG]), torch.tensor([POISSON]))
    x = torch.tensor(tet10_geometry("distorted")[None])
    k_default, bc_default = tet10_stiffness(x, d3)
    points, weights = duffy_rule(6)
    k_opt, bc_opt = tet10_stiffness(x, d3, rule=(points, weights))
    assert relative_frobenius(k_opt.numpy()[0], k_default.numpy()[0]) < 1e-12
    assert torch.allclose(bc_opt, bc_default)


def test_opt_in_rule_removes_the_curved_geometry_deficiency():
    d3 = elasticity_matrix_3d(torch.tensor([YOUNG]), torch.tensor([POISSON]))
    d = _d()
    reference_points, reference_weights = duffy_rule(24)
    nodes = tet10_geometry("curved", 0.11)
    x = torch.tensor(nodes[None])
    k_reference = element_stiffness(nodes, d, reference_points, reference_weights)
    norm = np.linalg.norm(k_reference)
    k_default, _ = tet10_stiffness(x, d3)
    assert np.linalg.norm(k_default[0].numpy() - k_reference) / norm > 2.0e-2
    points, weights = duffy_rule(6)
    k_opt, _ = tet10_stiffness(x, d3, rule=(points, weights))
    assert np.linalg.norm(k_opt[0].numpy() - k_reference) / norm < 1e-4


def test_opt_in_rule_validates_its_shapes():
    d3 = elasticity_matrix_3d(torch.tensor([YOUNG]), torch.tensor([POISSON]))
    x = torch.tensor(tet10_geometry("regular")[None])
    with pytest.raises(ValueError):
        tet10_stiffness(x, d3, rule=(np.zeros((5, 2)), np.full(5, 1.0 / 30.0)))
    with pytest.raises(ValueError):
        tet10_stiffness(x, d3, rule=(KEAST_POINTS, np.full(4, 1.0 / 24.0)))


# --------------------------------------------------------------------------------------
# 6. Committed evidence staleness guards.
# --------------------------------------------------------------------------------------
def test_evidence_json_matches_the_module_outputs():
    evidence = _evidence()
    assert evidence["acceptance_gate"] == GATE
    verdict = evidence["verdict"]
    assert verdict["straight_edges_exact"] is True
    recomputed = integration_sweep(kind="curved", curvatures=(0.0,))
    assert recomputed[0]["stiffness_keast_vs_reference"] < 1e-12
    mass = mass_matrix_audit(curvatures=(0.0,))[0]
    assert mass["relative_frobenius_error"] == pytest.approx(
        verdict["mass_matrix_error_straight"], rel=1e-9
    )
    assert evidence["reference_rule"]["points"] == 24 ** 3
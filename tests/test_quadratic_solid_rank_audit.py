"""Tests for the quadratic-solid rank audit (HEX20 rules, TET10, mesh scans).

These gates lock the numbers reported in ``docs/quadratic-solid-rank-audit.md``.
The audit-only 8-point rule is the negative control: the classifier must report six
spurious mechanisms for it, which proves that the zero counts on the shipped rule
are a measurement and not a blind spot of the classifier.
"""
from __future__ import annotations

import pytest
import torch

from tensorfem.element_rank_audit import (
    _hex20_cube,
    _hex20_curved_geometry,
    _hex20_distorted_geometry,
    _tet10_geometry,
    audit_hex20_mesh_scan,
    audit_library,
    audit_quadratic_solids,
    classify_null_space,
    hex20_quadrature_convergence,
    hex20_reduced_stiffness,
    hex20_stiffness_with_rule,
    quadrature_exactness,
    rigid_body_basis_solid3d,
)
from tensorfem.hex20 import hex20_stiffness
from tensorfem.quadratic_solid import tet10_stiffness
from tensorfem.solid3d import elasticity_matrix_3d

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)  # shared host: thread oversubscription dominates runtime

YOUNG = 200.0e9
POISSON = 0.3


def _d3():
    return elasticity_matrix_3d(torch.tensor([YOUNG]), torch.tensor([POISSON]))


def _records():
    return {r.element: r for r in audit_quadratic_solids(young=YOUNG, poisson=POISSON)}


def test_shipped_hex20_rule_is_rank_sufficient_on_all_geometries():
    """27-point rule: exactly the six rigid-body modes, straight or curved geometry."""
    records = _records()
    for shape in ("unit cube", "distorted, straight mid-edges", "distorted, curved edges"):
        record = records[f"hex20 3x3x3 ({shape})"]
        assert record.dofs == 60
        assert record.null_dim == 6
        assert record.expected_rigid_dim == 6
        assert record.spurious == 0, f"{shape}: {record.spurious} spurious modes"


def test_reduced_eight_point_rule_has_six_mechanisms():
    """Negative control: the cheaper rule must be caught with exactly six mechanisms."""
    records = _records()
    for shape in ("unit cube", "distorted, straight mid-edges", "distorted, curved edges"):
        record = records[f"hex20 2x2x2 reduced ({shape})"]
        assert record.null_dim == 12
        assert record.spurious == 6, f"{shape}: {record.spurious} spurious modes"


def test_tet10_is_rank_sufficient_including_curved_geometry():
    """TET10 had never been audited; straight and curved edges must show 6/6."""
    records = _records()
    for shape in ("straight edges", "curved edges"):
        record = records[f"tet10 ({shape})"]
        assert record.dofs == 30
        assert record.null_dim == 6
        assert record.spurious == 0


def test_tet10_enters_the_library_level_audit():
    """The library audit gained the quadratic tetrahedron instead of silently skipping it."""
    records = {r.element: r for r in audit_library(young=YOUNG, poisson=POISSON)}
    assert records["tet10 (3-D, straight edges)"].spurious == 0
    assert records["hex20 (3x3x3 integration)"].null_dim == 6


def test_rigid_body_basis_is_annihilated_on_curved_quadratic_geometry():
    """A rigid motion of a curved brick/tetra must carry no strain energy at all."""
    d3 = _d3()
    for nodes, k in (
        (_hex20_curved_geometry(), hex20_stiffness(_hex20_curved_geometry().unsqueeze(0), d3)[0][0]),
        (_tet10_geometry(True), tet10_stiffness(_tet10_geometry(True).unsqueeze(0), d3)[0][0]),
    ):
        basis = rigid_body_basis_solid3d(nodes)
        report = classify_null_space(k, basis)
        assert report["explained"] == basis.shape[1]
        energy = basis.transpose(0, 1) @ k @ basis
        scale = float(torch.linalg.matrix_norm(k)) * float(torch.linalg.matrix_norm(basis)) ** 2
        assert float(torch.linalg.matrix_norm(energy)) <= 1e-12 * scale


def test_mesh_scan_suppresses_element_mechanisms():
    """Element mechanisms that cannot cross a conforming interface leave no global mode."""
    records = audit_hex20_mesh_scan(sizes=(2,), young=YOUNG, poisson=POISSON)
    assert {r.element for r in records} == {
        "hex20 3x3x3 warped mesh",
        "hex20 2x2x2 reduced warped mesh",
    }
    for record in records:
        assert record.expected_rigid_dim == 6
        assert record.null_dim == 6
        assert record.spurious == 0


def test_derived_rules_reproduce_the_shipped_stiffness():
    """The order-3 audit rule must be bit-comparable with production, or the table lies."""
    d3 = _d3()
    for nodes in (_hex20_cube(), _hex20_distorted_geometry(), _hex20_curved_geometry()):
        shipped, _ = hex20_stiffness(nodes.unsqueeze(0), d3)
        audited = hex20_stiffness_with_rule(nodes.unsqueeze(0), d3, 3)
        assert torch.allclose(shipped, audited, rtol=0.0, atol=1e-12 * float(shipped.abs().max()))
        reduced = hex20_reduced_stiffness(nodes.unsqueeze(0), d3)
        assert reduced.shape == shipped.shape


def test_quadrature_exactness_matches_the_documented_orders():
    """Machine-checked exactness: 27-point degree 5, 8-point degree 3, Keast degree 3."""
    rows = {row["rule"]: row for row in quadrature_exactness(max_degree=5)}
    assert rows["hex20 3x3x3 Gauss (shipped)"]["points"] == 27
    assert rows["hex20 3x3x3 Gauss (shipped)"]["exact_through_total_degree"] == 5
    assert rows["hex20 2x2x2 Gauss (audit only)"]["points"] == 8
    assert rows["hex20 2x2x2 Gauss (audit only)"]["exact_through_total_degree"] == 3
    assert rows["tet10 five-point Keast (shipped)"]["exact_through_total_degree"] == 3
    assert rows["tet4 centroid rule (shipped)"]["exact_through_total_degree"] == 1


def test_quadrature_convergence_bounds_the_shipped_rule_error():
    """27-point rule is exact on the regular brick and under-integrates curved ones <1%."""
    rows = {row["geometry"]: row for row in hex20_quadrature_convergence(young=YOUNG, poisson=POISSON)}
    cube = rows["unit cube"]
    assert cube["order3_vs_order4"] < 1e-14
    curved = rows["distorted, curved edges"]
    assert curved["order3_vs_order4"] < 1e-2
    assert curved["order4_vs_order5"] < curved["order3_vs_order4"]
    # the audit-only 8-point rule is off by more than the 3 % acceptance gate
    for row in rows.values():
        assert row["order2_vs_order4"] > 0.03


def test_unknown_quadrature_order_is_rejected():
    """Silently falling back to a default rule is exactly the failure mode under audit."""
    with pytest.raises(ValueError):
        hex20_stiffness_with_rule(_hex20_cube().unsqueeze(0), _d3(), 7)
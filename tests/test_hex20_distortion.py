"""HEX20 under distortion/curvature, plus fail-closed controls.

Closes the debt recorded in ``docs/element-rank-deficiency-audit.md`` ("the
20-node brick was only audited on the regular cube; distorted bricks and the
reduced-integration variant are pending") and guards the exactness properties
that the structured-mesh tests in ``tests/test_hex20.py`` cannot see.
"""

import pytest
import torch

from tensorfem.element_rank_audit import audit_element, rigid_body_basis_solid3d
from tensorfem.hex20 import hex20_shape_gradients, structured_hex20_mesh
from tensorfem.hex20_distortion_audit import (
    boundary_nodes,
    cylindrical_bend,
    element_stiffness,
    element_stiffness_reduced,
    element_volume,
    energy_identity_residual,
    graded_shear,
    patch_residual,
    rank_audit,
    warp,
)
from tensorfem.solid3d import elasticity_matrix_3d

D = torch.float64
GRAD = torch.tensor(
    [[0.012, 0.007, -0.003], [-0.002, -0.009, 0.005], [0.004, 0.006, 0.015]], dtype=D
)
OFFSET = torch.tensor([0.1, -0.2, 0.05], dtype=D)
ROUNDOFF = 1e-11


def _mesh(kind, amplitude=0.0, radius=0.0, nx=2, ny=2, nz=2):
    nodes, cells = structured_hex20_mesh(2.0, 1.0, 1.0, nx, ny, nz, dtype=D)
    ref = nodes.clone()
    if kind == "warp":
        nodes = warp(nodes, amplitude)
    elif kind == "graded_shear":
        nodes = graded_shear(nodes, amplitude)
    elif kind == "bent":
        nodes = cylindrical_bend(nodes, radius, axis_x=0.0)
    elif kind != "regular":
        raise ValueError(kind)
    return nodes, cells, boundary_nodes(ref, 2.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "kind,amplitude,radius",
    [
        ("regular", 0.0, 0.0),
        ("warp", 0.1, 0.0),
        ("warp", 0.4, 0.0),
        ("graded_shear", 0.1, 0.0),
        ("graded_shear", 0.4, 0.0),
        ("bent", 0.0, 1.0),
        ("bent", 0.0, 2.0),
        ("bent", 0.0, 8.0),
    ],
)
def test_linear_patch_exact_for_distorted_and_curved_meshes(kind, amplitude, radius):
    """Isoparametric completeness: linear field must be reproduced at roundoff."""
    nodes, cells, bnd = _mesh(kind, amplitude, radius)
    out = patch_residual(nodes, cells, bnd, GRAD, OFFSET)
    assert out["free_dofs"] > 0
    assert out["displacement_relative_error"] < ROUNDOFF
    assert out["centroid_stress_relative_error"] < 1e-10


@pytest.mark.parametrize("kind,amplitude,radius", [("warp", 0.3, 0.0), ("graded_shear", 0.4, 0.0), ("bent", 0.0, 1.5)])
def test_constant_strain_energy_identity(kind, amplitude, radius):
    """u^T K u = V sigma:epsilon exactly for a constant-strain field."""
    nodes, cells, _ = _mesh(kind, amplitude, radius)
    assert energy_identity_residual(nodes, cells, GRAD) < 1e-11


def test_center_jacobian_control_is_rejected_on_curved_elements():
    """Fail-closed: the classic single-Jacobian bug must fail the patch test.

    A correct curved element is patch-exact to roundoff, the broken variant is
    not -- so the gate has teeth on curved geometry.
    """
    for radius in (1.0, 2.0, 5.0):
        nodes, cells, bnd = _mesh("bent", radius=radius)
        good = patch_residual(nodes, cells, bnd, GRAD, OFFSET)
        bad = patch_residual(nodes, cells, bnd, GRAD, OFFSET, center_jacobian=True)
        assert good["displacement_relative_error"] < ROUNDOFF
        assert bad["displacement_relative_error"] > 1e-5
        assert bad["displacement_relative_error"] > 1e10 * good["displacement_relative_error"]


def test_center_jacobian_control_is_blind_on_affine_meshes():
    """Documented limitation of that control: affine maps have a constant Jacobian."""
    nodes, cells, bnd = _mesh("graded_shear", amplitude=0.4)
    good = patch_residual(nodes, cells, bnd, GRAD, OFFSET)
    bad = patch_residual(nodes, cells, bnd, GRAD, OFFSET, center_jacobian=True)
    assert good["displacement_relative_error"] < ROUNDOFF
    assert bad["displacement_relative_error"] < ROUNDOFF


@pytest.mark.parametrize("kind,amplitude,radius", [("warp", 0.3, 0.0), ("graded_shear", 0.4, 0.0), ("bent", 0.0, 2.0)])
def test_no_hourglass_on_distorted_and_curved_bricks(kind, amplitude, radius):
    nodes, cells, _ = _mesh(kind, amplitude, radius, nx=1, ny=1, nz=1)
    rec = rank_audit(nodes, cells, label=f"hex20 ({kind})")
    assert rec.null_dim == 6
    assert rec.spurious == 0
    assert rec.first_positive_ratio > 1e-4


def test_reduced_2x2x2_brick_is_rank_deficient():
    """Reduced integration of HEX20 creates 6 hourglass modes -> inadmissible."""
    for kind, amplitude, radius in (("regular", 0.0, 0.0), ("warp", 0.3, 0.0), ("bent", 0.0, 2.0)):
        nodes, cells, _ = _mesh(kind, amplitude, radius, nx=1, ny=1, nz=1)
        x = nodes[cells.long()]
        d = elasticity_matrix_3d(x.new_full((len(x),), 70e9), x.new_full((len(x),), 0.27))
        ke, _ = element_stiffness_reduced(x, d)
        rec = audit_element(f"hex20 reduced ({kind})", ke[0], rigid_body_basis_solid3d(x[0]))
        assert rec.spurious == 6
        kfull, _ = element_stiffness(x, d)
        full = audit_element(f"hex20 full ({kind})", kfull[0], rigid_body_basis_solid3d(x[0]))
        assert full.spurious == 0


def test_bent_sector_volume_matches_analytic_shell_volume():
    """Sum of element volumes vs L*W*(H + H^2/(2R)) for the curved block."""
    for radius in (1.0, 2.0, 8.0, 20.0):
        nodes, cells, _ = _mesh("bent", radius=radius, nx=4, ny=2, nz=2)
        vol = float(element_volume(nodes[cells.long()]).sum())
        exact = 2.0 * 1.0 * (1.0 + 1.0 / (2.0 * radius))
        assert abs(vol - exact) / exact < 1e-3


def test_shape_gradient_partition_still_holds_under_warp():
    """Sanity: warping nodes does not enter the parent-domain partition identity."""
    dummy = torch.zeros(1, dtype=D)
    for r, s, t in ((0.0, 0.0, 0.0), (0.37, -0.21, 0.43)):
        n, dn = hex20_shape_gradients(r, s, t, like=dummy)
        assert abs(float(n.sum()) - 1.0) < 1e-13
        assert torch.linalg.vector_norm(dn.sum(0)) < 1e-13
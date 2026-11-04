"""Gates for the three-dimensional Lamé thick-cylinder qualification."""
from __future__ import annotations

import math

import pytest
import torch

# the dense reference solves must not oversubscribe a busy machine
torch.set_num_threads(2)

from tensorfem.thick_cylinder_3d import (
    LameCylinderReference, field_errors, qualify_thick_cylinder, quarter_cylinder_mesh,
    solve_thick_cylinder, stress_in_cylindrical_frame)

MESHES = ((3, 8, 3), (4, 12, 4), (5, 16, 5))


@pytest.fixture(scope="module")
def reference() -> LameCylinderReference:
    return LameCylinderReference()


@pytest.fixture(scope="module")
def runs(reference):
    solves = [solve_thick_cylinder(reference, nr=a, ntheta=b, nz=c) for a, b, c in MESHES]
    short = solve_thick_cylinder(reference, nr=5, ntheta=16, nz=5, half_length=.6)
    return solves, short


def test_closed_form_satisfies_traction_boundary_conditions(reference):
    r = torch.tensor([reference.inner_radius, reference.outer_radius], dtype=torch.float64)
    sigma_r = reference.sigma_r(r)
    assert sigma_r[0].item() == pytest.approx(-reference.inner_pressure, rel=1e-12)
    assert sigma_r[1].item() == pytest.approx(-reference.outer_pressure, abs=1e-6)
    # closed-end solution: uniform axial stress, uniform axial strain
    eps_z = (reference.sigma_z(r) - reference.poisson
             * (reference.sigma_r(r) + reference.sigma_theta(r))) / reference.young
    assert torch.allclose(eps_z, torch.full_like(eps_z, eps_z[0]))
    # plane-strain variant differs only in sigma_z and u_r (the problem is statically
    # determinate in sigma_r/sigma_theta), which is what makes it a discriminating control.
    plane = LameCylinderReference(closed_ends=False)
    assert torch.allclose(plane.sigma_theta(r), reference.sigma_theta(r))
    assert not math.isclose(plane.sigma_z(r)[0].item(), reference.sigma_z(r)[0].item(),
                            rel_tol=.1)


def test_quarter_sector_mesh_topology(reference):
    mesh = quarter_cylinder_mesh(4, 12, 5, inner_radius=reference.inner_radius,
                                 outer_radius=reference.outer_radius, half_length=1.2)
    assert mesh.elements.shape[0] == 4 * 12 * 5
    assert mesh.nodes.shape[1] == 3
    # both cut planes are exact symmetry planes, so they carry exact zero coordinates
    assert torch.all(mesh.nodes[mesh.symmetry["y_zero"], 1] == 0)
    assert torch.all(mesh.nodes[mesh.symmetry["x_zero"], 0] == 0)
    assert torch.all(mesh.nodes[mesh.symmetry["z_mid"], 2] == 1.2)
    radius = torch.linalg.vector_norm(mesh.nodes[:, :2], dim=-1)
    assert float(radius.min()) == pytest.approx(reference.inner_radius)
    assert float(radius.max()) == pytest.approx(reference.outer_radius)


def test_stress_rotation_recovers_cylindrical_components():
    # an axisymmetric tensor with sigma_r = 3, sigma_theta = -5 at a 45 degree point
    xyz = torch.tensor([[1., 1., 0.]], dtype=torch.float64) / math.sqrt(2)
    ex, ey = 3., -5.
    c = s = math.sqrt(.5)
    sxx = c * c * ex + s * s * ey
    syy = s * s * ex + c * c * ey
    sxy = c * s * (ex - ey)
    stress = torch.tensor([[sxx, syy, 7., sxy, 0., 0.]], dtype=torch.float64)
    out = stress_in_cylindrical_frame(stress, xyz)
    assert out["sigma_r"].item() == pytest.approx(ex, rel=1e-12)
    assert out["sigma_theta"].item() == pytest.approx(ey, rel=1e-12)
    assert out["sigma_z"].item() == pytest.approx(7.)


def test_bore_expands_outward_under_internal_pressure(runs, reference):
    """Regression for the traction-sign trap: t = -p*n with n the *outward* normal."""
    fine = runs[0][-1]
    bore = torch.isclose(torch.linalg.vector_norm(fine.nodes[:, :2], dim=-1),
                         fine.nodes.new_tensor(reference.inner_radius))
    mid = torch.isclose(fine.nodes[:, 2], fine.nodes.new_tensor(float(fine.nodes[:, 2].max())))
    picked = bore & mid
    displacement = fine.displacement.reshape(-1, 3)[picked]
    radial = (displacement[:, :2] * fine.nodes[picked][:, :2]).sum(-1)
    assert torch.all(radial > 0), "internal pressure must expand the bore outward"


def test_three_mesh_qualification_passes_below_three_percent(runs, reference):
    solves, short = runs
    verdict = qualify_thick_cylinder(reference, solves, length_run=short)
    assert verdict.status == "qualified", verdict.missing
    for key in ("sigma_r_l2", "sigma_theta_l2", "sigma_z_l2", "u_r_max"):
        assert verdict.errors[key] < .03, (key, verdict.errors[key])
    for key in ("sigma_r_l2", "sigma_theta_l2", "u_r_max"):
        assert verdict.convergence[f"{key}_order"] > .8
    assert verdict.length_sensitivity < .03
    assert verdict.equilibrium["axial_balance"] < .03
    assert verdict.equilibrium["radial_balance"] < .03
    assert verdict.equilibrium["free_residual"] < 1e-8
    # the plane-strain Lamé solution must be rejected by the same metric
    assert verdict.negative_control["sigma_z_l2"] > .1


def test_gate_is_fail_closed(runs, reference):
    solves, _ = runs
    too_few = qualify_thick_cylinder(reference, solves[:2], length_run=solves[-1])
    assert too_few.status == "blocked"
    assert any("three mesh" in item for item in too_few.missing)
    without_length = qualify_thick_cylinder(reference, solves, length_run=None)
    assert without_length.status == "blocked"
    assert any("length" in item for item in without_length.missing)


def test_plane_strain_target_is_rejected(runs):
    """A wrong constitutive target must not pass the 3 % gate."""
    solves, _ = runs
    plane = LameCylinderReference(closed_ends=False)
    errors = field_errors(solves[-1], plane)
    assert errors["sigma_z_l2"] > .03
    assert errors["u_r_max"] > .03
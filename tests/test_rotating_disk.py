"""Verification of the centrifugal body-load path against Timoshenko's rotating disk.

The closed form in :mod:`tensorfem.rotating_disk` is itself audited here from
three independent statements (equilibrium, compatibility, traction-free rims)
before it is used as an oracle, so a bug in the reference cannot hide behind a
matched finite element run.
"""
import math

import pytest
import torch

from tensorfem.rotating_disk import (
    RotatingDiskReference,
    annular_disk_mesh,
    centrifugal_consistent_loads,
    radial_hoop_stress,
    solve_rotating_disk,
)

DTYPE = torch.float64
RHO, OMEGA, E, NU = 7850.0, 100.0, 200e9, 0.3
A, B, T = 0.25, 0.5, 0.02


def _reference(plane="stress", nu=NU, a=A, b=B, omega=OMEGA):
    return RotatingDiskReference(a, b, RHO, omega, E, nu, plane)


@pytest.mark.parametrize("plane", ["stress", "strain"])
@pytest.mark.parametrize("nu", [0.0, 0.3, 0.45])
def test_oracle_satisfies_equilibrium_compatibility_and_free_rims(plane, nu):
    """Equilibrium + compatibility + sigma_r(a)=sigma_r(b)=0 pin the solution uniquely."""
    ref = _reference(plane, nu)
    scale = RHO * OMEGA**2 * B
    r = torch.linspace(A, B, 41, dtype=DTYPE)
    residual = ref.equilibrium_residual(r).abs().max() / scale
    assert residual < 1e-9, residual
    sigma_r, _ = ref.stress(r)
    assert sigma_r[0].abs() / scale < 1e-12 and sigma_r[-1].abs() / scale < 1e-12
    rr = torch.linspace(0.26, 0.49, 2001, dtype=DTYPE)
    u = ref.displacement(rr)
    du = (u[2:] - u[:-2]) / (rr[2:] - rr[:-2])
    eps_r, eps_theta = ref.strain(rr[1:-1])
    assert float((du - eps_r).abs().max() / eps_r.abs().max()) < 1e-5
    assert float((u[1:-1] / rr[1:-1] - eps_theta).abs().max() / eps_theta.abs().max()) < 1e-12


def test_consistent_load_element_resultant_is_exact():
    """Sum of nodal loads on one element equals rho*omega^2*A*x_c: 2x2 Gauss is exact."""
    nodes = torch.tensor([[0.0, 0.0], [0.4, 0.1], [0.3, 0.5], [-0.1, 0.3]], dtype=DTYPE)
    elements = torch.tensor([[0, 1, 2, 3]])
    forces = centrifugal_consistent_loads(nodes, elements, RHO, OMEGA, T)
    x, y = nodes[:, 0], nodes[:, 1]
    x_next, y_next = torch.roll(x, -1), torch.roll(y, -1)
    cross = x * y_next - x_next * y
    area = 0.5 * abs(float(cross.sum()))
    # Exact polygonal centroid (independent of the module's own quadrature).
    centroid = torch.stack(((x + x_next) * cross, (y + y_next) * cross)).sum(1) / (
        3.0 * float(cross.sum())
    )
    for component, axis in ((0, 0), (1, 1)):
        got = float(forces[component::2].sum())
        exact = RHO * OMEGA**2 * area * T * float(centroid[axis])
        assert abs(got - exact) / abs(exact) < 1e-12, (got, exact)


def test_quarter_model_load_resultant_matches_analytic_integral():
    nodes, elements = annular_disk_mesh(A, B, 8, 48, quarter=True, dtype=DTYPE)
    forces = centrifugal_consistent_loads(nodes, elements, RHO, OMEGA, T)
    exact = RHO * OMEGA**2 * T * (B**3 - A**3) / 3.0
    assert abs(float(forces[0::2].sum()) - exact) / exact < 1e-3
    assert abs(float(forces[1::2].sum()) - exact) / exact < 1e-3


def test_radial_hoop_transformation():
    coordinates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=DTYPE)
    stress = torch.tensor([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]], dtype=DTYPE)
    sigma_r, sigma_theta, _ = radial_hoop_stress(coordinates, stress)
    assert sigma_r[0].item() == pytest.approx(1.0)
    assert sigma_theta[0].item() == pytest.approx(2.0)
    assert sigma_r[1].item() == pytest.approx(2.0)
    assert sigma_theta[1].item() == pytest.approx(1.0)
    assert sigma_r[2].item() == pytest.approx(2.0)  # 45 deg: vanishing shear, isotropic


def test_rotating_annulus_matches_timoshenko_solution_and_converges():
    errors = []
    for n_radial, n_angular in ((4, 24), (8, 48), (12, 72)):
        out = solve_rotating_disk(n_radial, n_angular)
        assert out["stress_rel_error"] < 0.03, out["stress_rel_error"]
        assert out["peak_hoop_error"] < 0.03, out["peak_hoop_error"]
        assert out["displacement_rel_error"] < 0.03, out["displacement_rel_error"]
        errors.append(out["stress_rel_error"])
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-3


def test_rotating_long_cylinder_plane_strain():
    for n in (8, 12):
        out = solve_rotating_disk(n, 6 * n, plane="strain")
        assert out["stress_rel_error"] < 0.03, out["stress_rel_error"]
        assert out["displacement_rel_error"] < 0.03, out["displacement_rel_error"]


def test_near_solid_disk_with_graded_mesh():
    out = solve_rotating_disk(12, 72, inner_radius=0.025, grading=1.15)
    assert out["stress_rel_error"] < 0.03
    assert out["peak_hoop_error"] < 0.03
    assert out["displacement_rel_error"] < 0.03


def test_energy_balance_and_global_equilibrium():
    out = solve_rotating_disk(8, 48)
    assert out["energy_balance_error"] < 1e-12
    assert out["equilibrium_residual"] < 1e-12
    assert out["cut_balance_error"] < 3e-3


def test_gate_has_teeth():
    """The 3% gate is not vacuous: two plausible mistakes must blow through it."""
    lumped = solve_rotating_disk(8, 48, lumped=True)
    assert lumped["stress_rel_error"] > 0.1
    mismatch = solve_rotating_disk(8, 48, omega_scale=1.05, reference_omega=OMEGA)
    assert mismatch["stress_rel_error"] > 0.05
    zero = solve_rotating_disk(8, 48, omega=0.0)
    assert zero["max_hoop_stress_fem"] == 0.0
    assert zero["displacement_rel_error"] == 0.0


def test_rejects_unsupported_inputs():
    nodes, elements = annular_disk_mesh(A, B, 2, 4, dtype=DTYPE)
    with pytest.raises(ValueError):
        centrifugal_consistent_loads(nodes, elements[:, :3], RHO, OMEGA, T)
    with pytest.raises(ValueError):
        annular_disk_mesh(A, B, 0, 4)
    with pytest.raises(ValueError):
        annular_disk_mesh(A, B, 2, 4, grading=0.0)
    with pytest.raises(ValueError):
        RotatingDiskReference(0.5, 0.5, RHO, OMEGA, E, NU)
    with pytest.raises(ValueError):
        RotatingDiskReference(A, B, RHO, OMEGA, E, NU, plane="axisymmetric")

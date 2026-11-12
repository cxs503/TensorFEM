"""Tests for CASE-HOLLOW-SPHERE-3D (3-D thick-walled sphere under pressure).

Invariants, not smoke: the analytic reference is checked against the boundary
conditions, the strong-form equilibrium equation, and Clapeyron's theorem; the
pressure load generator is checked for closed-surface self-equilibrium; the case
verdict is checked to *fail* when the FE field is perturbed.
"""
import math

import pytest
import torch

from tensorfem.hollow_sphere_3d import (CASE, TOLERANCE, _mesh, _pressure_loads,
                                        _serendipity_quad, case_parameters,
                                        convergence, lame_displacement,
                                        lame_sphere, probe_metrics, shell_volume,
                                        solve_hollow_sphere, strain_energy)


@pytest.fixture(scope="module")
def hex20_case():
    return solve_hollow_sphere(4, 2, order=2, keep_arrays=True)


def test_case_parameters_reject_non_physical_input():
    with pytest.raises(ValueError):
        case_parameters({"inner_radius_m": 10.0, "outer_radius_m": 8.0,
                         "young_modulus_pa": 1e9, "poisson_ratio": 0.3,
                         "internal_pressure_pa": 1.0})
    with pytest.raises(ValueError):
        case_parameters({"inner_radius_m": 8.0, "outer_radius_m": 10.0,
                         "young_modulus_pa": 1e9, "poisson_ratio": 0.5,
                         "internal_pressure_pa": 1.0})
    with pytest.raises(ValueError):
        solve_hollow_sphere(4, 1)  # one radial layer cannot probe both surfaces


def test_lame_reference_satisfies_boundary_conditions():
    a, b, p = CASE["inner_radius_m"], CASE["outer_radius_m"], CASE["internal_pressure_pa"]
    edge = lame_sphere(torch.tensor([a, b]))
    assert float(edge["radial_stress"][0]) == pytest.approx(-p, rel=1e-14)
    assert abs(float(edge["radial_stress"][1])) < 1e-9 * p


def test_lame_reference_satisfies_strong_form_equilibrium():
    """d sigma_r/dr + 2 (sigma_r - sigma_t)/r = 0 at interior radii."""
    a, b = CASE["inner_radius_m"], CASE["outer_radius_m"]
    r = torch.linspace(a, b, 401, dtype=torch.float64)[1:-1]
    h = 1e-6
    dsigma = ((lame_sphere(r + h)["radial_stress"] -
               lame_sphere(r - h)["radial_stress"]) / (2 * h))
    fields = lame_sphere(r)
    residual = dsigma + 2 * (fields["radial_stress"] - fields["hoop_stress"]) / r
    scale = float(fields["radial_stress"].abs().max()) / a
    assert float(residual.abs().max()) < 1e-6 * scale


def test_strain_energy_matches_clapeyron():
    a = CASE["inner_radius_m"]
    p = CASE["internal_pressure_pa"]
    ur = float(lame_sphere(torch.tensor(a))["radial_displacement"])
    clapeyron = 0.5 * p * ur * 4 * math.pi * a ** 2
    assert strain_energy() == pytest.approx(clapeyron, rel=1e-10)


def test_lame_displacement_is_radial():
    x = torch.tensor([[8.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 10.0]])
    u = lame_displacement(x)
    assert torch.allclose(u, u.norm(dim=1)[:, None] * x / x.norm(dim=1)[:, None])
    assert float((u * x).sum(1).min()) > 0  # outward under internal pressure


def test_serendipity_quad_partition_of_unity():
    for xi, eta in ((0.0, 0.0), (0.3, -0.7), (-0.9, 0.2), (1.0, -1.0)):
        N, dN = _serendipity_quad(torch.tensor(xi, dtype=torch.float64),
                                  torch.tensor(eta, dtype=torch.float64))
        assert float(N.sum()) == pytest.approx(1.0, abs=1e-12)
        assert float(dN.sum(0).abs().max()) < 1e-12
    nodes = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
                          [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                         dtype=torch.float64)
    N, _ = _serendipity_quad(nodes[:, 0], nodes[:, 1])
    assert torch.allclose(N, torch.eye(8, dtype=torch.float64), atol=1e-12)


@pytest.mark.parametrize("order", [1, 2])
def test_pressure_loads_are_self_equilibrated(order):
    case = case_parameters(None)
    nodes, elements, faces = _mesh(4, 2, order, case)
    force, resultants = _pressure_loads(nodes, faces, elements, len(faces), order,
                                        case["internal_pressure_pa"], case)
    assert abs(float(force.reshape(-1, 3).sum(0).abs().max())) < 1e-9 * float(force.abs().max())
    assert resultants["closed_surface_force_resultant"] < 1e-12
    assert resultants["closed_surface_moment_resultant"] < 1e-12
    # Internal pressure pushes the wall radially outwards (sigma_r(a) = -p), so the
    # resultant radial component equals p * (discretised inner area).
    radius = torch.linalg.vector_norm(nodes, dim=1)
    radial_component = float((force.reshape(-1, 3) * nodes / radius[:, None]).sum())
    analytic_area = 4 * math.pi * case["inner_radius_m"] ** 2
    assert radial_component == pytest.approx(
        case["internal_pressure_pa"] * analytic_area, rel=0.08)


def test_mesh_geometry_closes_on_the_analytic_shell_volume(hex20_case):
    """Geometry gate (the Cb/L/B analogue for this case): mesh volume vs exact shell."""
    from tensorfem.hollow_sphere_3d import _element_volumes

    case = case_parameters(None)
    exact = shell_volume(case)
    errors = {}
    for n in (4, 8):
        nodes, elements, _ = _mesh(n, 2, 1, case)
        errors[("hex8", n)] = abs(float(_element_volumes(nodes[elements], 1).sum()) - exact) / exact
    nodes, elements, _ = _mesh(4, 2, 2, case)
    errors[("hex20", 4)] = abs(float(_element_volumes(nodes[elements], 2).sum()) - exact) / exact
    assert errors[("hex8", 4)] < 0.10
    assert errors[("hex8", 8)] < 0.03
    assert errors[("hex8", 8)] < errors[("hex8", 4)]
    # curved-boundary projection makes the quadratic mesh an order of magnitude better
    assert errors[("hex20", 4)] < 5e-3
    assert hex20_case["mesh"]["mesh_volume_m3"] == pytest.approx(exact, rel=5e-3)


def test_hex20_case_meets_five_percent_acceptance(hex20_case):
    metrics = hex20_case["metrics"]
    assert hex20_case["passed"] is True
    assert metrics["worst_relative_error"] < TOLERANCE
    assert metrics["worst_secondary_error"] < TOLERANCE
    assert hex20_case["equilibrium"]["free_residual_relative"] < 1e-8
    assert hex20_case["loads"]["closed_surface_force_resultant"] < 1e-12
    # the whole point of the quadratic path: 192 elements suffice
    assert hex20_case["mesh"]["elements"] == 192
    # strain energy is a second independent global check
    assert hex20_case["energy"]["relative_error"] < 0.01


def test_perturbed_field_is_rejected(hex20_case):
    """A 6% stress perturbation must break the verdict (the gate is not decorative)."""
    case = case_parameters(None)
    nodes, elements, faces = _mesh(4, 2, 2, case)
    arrays = hex20_case["arrays"]
    stress, volume, u = arrays["stress"], arrays["volume"], arrays["displacement"]
    baseline = probe_metrics(nodes, elements, stress, u, volume, len(faces), case, order=2)
    # 10 % displacement bias: the pressure-vessel acceptance must notice it
    perturbed = probe_metrics(nodes, elements, stress, u * 1.10, volume, len(faces), case, order=2)
    assert baseline["passed"] is True
    assert perturbed["worst_relative_error"] > baseline["worst_relative_error"]
    assert perturbed["passed"] is False


def test_refinement_reduces_energy_error(hex20_case):
    """Two resolutions: the energy norm must converge monotonically from above."""
    fine = hex20_case  # n = 4
    coarse = solve_hollow_sphere(2, 2, order=2)
    assert fine["energy"]["relative_error"] < coarse["energy"]["relative_error"]
    assert fine["energy"]["fe_strain_energy_j"] >= strain_energy()  # displacement FE bound
    ladder = convergence([coarse, fine])
    assert ladder["monotone_decreasing"] is True
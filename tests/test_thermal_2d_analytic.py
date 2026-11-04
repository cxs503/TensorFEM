import math

import pytest
import torch

from tensorfem.thermal import ThermalModel, solve_steady_thermal
from tensorfem.thermal_analytic import (
    annular_convergence,
    annular_quad_mesh,
    annular_radial_gradient,
    annular_temperature,
    element_areas,
    field_errors,
    flux_error,
    harmonic_convergence,
    harmonic_gradient,
    harmonic_temperature,
    heat_balance_residual,
    laplacian_residual,
    observed_order,
    solve_dirichlet,
    structured_quad_mesh,
    transient_decay,
    transient_decay_rate,
    transient_temperature,
)

D = torch.float64
GATE = 0.03


def annulus(pts):
    return annular_temperature(torch.linalg.vector_norm(pts, dim=1), 1.0, 2.0, 100.0, 20.0)


def annulus_gradient(pts):
    r = torch.linalg.vector_norm(pts, dim=1)
    magnitude = annular_radial_gradient(r, 1.0, 2.0, 100.0, 20.0)
    return pts / r[:, None] * magnitude[:, None]


# ---------------------------------------------------------------------------
# the references themselves must be harmonic, otherwise every error below is
# meaningless
# ---------------------------------------------------------------------------
def test_closed_form_fields_have_zero_laplacian():
    probe = structured_quad_mesh(17, 13).nodes
    assert laplacian_residual(harmonic_temperature, probe) < 1.0e-5
    inner = probe * 0.5 + 0.75
    assert laplacian_residual(annulus, inner) < 1.0e-5


def test_analytic_gradient_matches_finite_difference():
    pts = structured_quad_mesh(9, 7).nodes * 0.8 + 0.1
    step = 1.0e-6
    for field, gradient in ((harmonic_temperature, harmonic_gradient),
                            (annulus, annulus_gradient)):
        numeric = torch.zeros_like(pts)
        for axis in (0, 1):
            e = torch.zeros_like(pts)
            e[:, axis] = step
            numeric[:, axis] = (field(pts + e) - field(pts - e)) / (2.0 * step)
        error = torch.linalg.vector_norm(numeric - gradient(pts)) / torch.linalg.vector_norm(
            gradient(pts))
        assert float(error) < 1.0e-6


def test_mesh_areas_match_closed_form():
    square = structured_quad_mesh(7, 11, shear=0.4)
    assert abs(float(element_areas(square).sum()) - 1.0) < 1.0e-12
    ring = annular_quad_mesh(2, 16)
    expected = 0.5 * 16 * math.sin(2.0 * math.pi / 16) * (4.0 - 1.0)
    assert abs(float(element_areas(ring).sum()) - expected) < 1.0e-12


# ---------------------------------------------------------------------------
# case A: harmonic field on a Cartesian mesh
# ---------------------------------------------------------------------------
def test_harmonic_cartesian_converges_at_second_order_and_passes_3pct_gate():
    case = harmonic_convergence([8, 16, 32])
    assert case.l2[-1] < GATE
    assert case.l2[0] > case.l2[1] > case.l2[2]
    for order in case.orders:
        assert 1.9 < order < 2.1
    assert case.pointwise[-1] < GATE


def test_dirichlet_data_are_applied_exactly():
    mesh = structured_quad_mesh(12, 12)
    temperature = solve_dirichlet(mesh, harmonic_temperature)
    boundary = mesh.boundary_nodes
    exact = harmonic_temperature(mesh.nodes[boundary])
    assert float(torch.max(torch.abs(temperature[boundary] - exact))) < 1.0e-14


def test_pure_dirichlet_problem_conserves_energy():
    mesh = structured_quad_mesh(10, 12)
    temperature = solve_dirichlet(mesh, harmonic_temperature)
    residual = heat_balance_residual(mesh, harmonic_temperature, temperature)
    assert residual < 1.0e-10


# ---------------------------------------------------------------------------
# case B: same field on a sheared mesh (isoparametric mapping)
# ---------------------------------------------------------------------------
def test_sheared_mesh_keeps_second_order_convergence():
    cart = harmonic_convergence([8, 16, 32])
    sheared = harmonic_convergence([8, 16, 32], shear=0.35)
    assert sheared.l2[-1] < GATE
    for order in sheared.orders:
        assert 1.9 < order < 2.1
    # Shearing must not silently degrade accuracy far beyond the mesh-anisotropy
    # effect observed for Q4 bilinear elements.
    assert sheared.l2[-1] < 1.5 * cart.l2[-1]


def test_shear_actually_distorts_the_elements():
    # For bilinear quads dx/deta = (-x0 - x1 + x2 + x3)/4; it is identically zero
    # on a Cartesian grid and non-zero once the mesh is sheared, so the
    # non-orthogonal isoparametric path is really being exercised.
    for shear, expected_zero in ((0.0, True), (0.5, False)):
        coords = structured_quad_mesh(4, 4, shear=shear).nodes[
            structured_quad_mesh(4, 4, shear=shear).elements.long()]
        dx_deta = 0.25 * (-coords[:, 0, 0] - coords[:, 1, 0]
                          + coords[:, 2, 0] + coords[:, 3, 0])
        if expected_zero:
            assert float(torch.max(torch.abs(dx_deta))) < 1.0e-14
        else:
            assert float(torch.min(torch.abs(dx_deta))) > 1.0e-3


def test_inverted_element_fails_closed():
    mesh = structured_quad_mesh(4, 4)
    broken = type(mesh)(mesh.nodes, mesh.elements[:, [0, 3, 2, 1]])
    with pytest.raises(ValueError):
        field_errors(broken, torch.zeros(len(mesh.nodes), dtype=D), harmonic_temperature)


# ---------------------------------------------------------------------------
# case C: logarithmic radial field in an annulus (curvilinear mesh)
# ---------------------------------------------------------------------------
def test_annular_logarithmic_field_converges_and_passes_3pct_gate():
    case = annular_convergence([16, 32, 64])
    assert case.l2[-1] < GATE
    assert case.l2[0] > case.l2[1] > case.l2[2]
    for order in case.orders[1:]:
        assert 1.85 < order < 2.1
    assert case.pointwise[-1] < GATE


def test_annular_flux_recovery_is_second_order():
    errors = []
    for nt in (16, 32, 64):
        mesh = annular_quad_mesh(max(1, nt // 8), nt)
        temperature = solve_dirichlet(mesh, annulus)
        errors.append(flux_error(mesh, temperature, annulus_gradient)["gradient_l2_relative"])
    assert errors[-1] < GATE
    assert errors[0] > errors[1] > errors[2]
    assert 1.8 < observed_order(errors[0], errors[-1], 4.0) < 2.2


def test_annular_mesh_rejects_too_few_circumferential_cells():
    with pytest.raises(ValueError):
        annular_quad_mesh(2, 3)


# ---------------------------------------------------------------------------
# case D: two-dimensional transient decay
# ---------------------------------------------------------------------------
def test_two_dimensional_transient_decay_rate_and_shape():
    alpha = 1.0
    run = transient_decay(16, alpha=alpha)
    exact_rate = transient_decay_rate(alpha)
    assert abs(run["exact_rate"] - exact_rate) < 1.0e-12
    assert run["fitted_rate"] < 0.0
    assert run["rate_relative_error"] < GATE
    for checkpoint in run["checkpoints"]:
        assert checkpoint["field_l2_relative"] < GATE
        assert checkpoint["amplitude_relative"] < GATE


def test_transient_decay_error_decreases_with_mesh_refinement():
    coarse = transient_decay(8)["rate_relative_error"]
    fine = transient_decay(16)["rate_relative_error"]
    assert fine < coarse
    assert observed_order(coarse, fine, 2.0) > 1.5


def test_initial_condition_matches_the_exact_mode():
    mesh = structured_quad_mesh(12, 12)
    initial = transient_temperature(mesh.nodes, 0.0, 1.0)
    exact = torch.sin(math.pi * mesh.nodes[:, 0]) * torch.sin(math.pi * mesh.nodes[:, 1])
    assert float(torch.max(torch.abs(initial - exact))) < 1.0e-14


# ---------------------------------------------------------------------------
# discrimination: the metrics must react to a wrong field
# ---------------------------------------------------------------------------
def test_error_metric_rejects_scaled_and_shifted_fields():
    mesh = structured_quad_mesh(16, 16)
    temperature = solve_dirichlet(mesh, harmonic_temperature)
    honest = field_errors(mesh, temperature, harmonic_temperature)["l2_relative"]
    assert honest < GATE
    scaled = field_errors(mesh, 1.05 * temperature, harmonic_temperature)["l2_relative"]
    shifted = field_errors(mesh, temperature + 0.01, harmonic_temperature)["l2_relative"]
    assert scaled > GATE
    assert shifted > GATE


def test_coarse_mesh_is_not_accidentally_accurate():
    coarse = harmonic_convergence([4]).l2[0]
    fine = harmonic_convergence([32]).l2[0]
    assert coarse / fine > 20.0


def test_q4_path_requires_two_dimensional_coordinates():
    # The production kernel must reject a plane field declared as LINE2; this
    # guards the claim that the two-dimensional cases really run Q4.
    with pytest.raises(ValueError):
        ThermalModel(nodes=torch.rand(5, 2, dtype=D),
                     elements=torch.tensor([[0, 1], [1, 2]]),
                     conductivity=torch.tensor(1.0, dtype=D),
                     density=torch.tensor(1.0, dtype=D),
                     specific_heat=torch.tensor(1.0, dtype=D),
                     fixed_nodes=torch.tensor([0]), fixed_temperatures=torch.tensor([0.0], dtype=D))
    mesh = structured_quad_mesh(2, 2)
    model = ThermalModel(nodes=mesh.nodes, elements=mesh.elements,
                         conductivity=torch.tensor(1.0, dtype=D),
                         density=torch.tensor(1.0, dtype=D),
                         specific_heat=torch.tensor(1.0, dtype=D),
                         fixed_nodes=mesh.boundary_nodes,
                         fixed_temperatures=torch.zeros(len(mesh.boundary_nodes), dtype=D),
                         element_type="q4")
    assert solve_steady_thermal(model).shape == (len(mesh.nodes),)
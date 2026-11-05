"""Gates for the rotating-disc (Michel) verification case.

These tests lock the analytic reference *and* the finite-element agreement that
``docs/rotating-disc-michel-verification.md`` reports, at the coarsest mesh that
still clears the 3 % gate, so the suite stays fast.  The full ladder, the
grading study and the three negative controls live in
``benchmarks/rotating_disc_reference.py``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.continuum import ContinuumModel, solve_continuum
from tensorfem.lame_reference import (
    cartesian_to_polar_stress,
    sector_q4_mesh,
    symmetry_fixed_dofs,
)
from tensorfem.rotating_disc_reference import (
    bore_hoop_stress,
    centrifugal_nodal_loads,
    disc_coefficients,
    rotating_disc_state,
    sector_body_force_resultant,
    sector_hoop_traction_resultant,
    solid_rotating_disc_state,
)

DENSITY = 7850.0
YOUNG = 210.0e9
POISSON = 0.3
OMEGA = 2.0 * math.pi * 3000.0 / 60.0
OUTER = 0.50
INNER = 0.05
THICKNESS = 0.020
GATE = 0.03


def state(radius, *, inner=INNER, tolerance=0.0):
    return rotating_disc_state(
        radius,
        inner_radius=inner,
        outer_radius=OUTER,
        density=DENSITY,
        omega=OMEGA,
        young=YOUNG,
        poisson=POISSON,
        radius_tolerance=tolerance,
    )


def run(n_theta: int, n_radial: int, *, grading: float = 1.0, load_omega: float | None = None):
    omega = OMEGA if load_omega is None else load_omega
    nodes, cells = sector_q4_mesh(INNER, OUTER, n_theta, n_radial, radial_grading=grading)
    forces = centrifugal_nodal_loads(
        nodes, cells, thickness=THICKNESS, density=DENSITY, omega=omega
    )
    fixed = symmetry_fixed_dofs(nodes, sector_angle=math.pi / 2.0, n_theta=n_theta, n_radial=n_radial)
    model = ContinuumModel(
        nodes, cells, torch.tensor(YOUNG), torch.tensor(POISSON), torch.tensor(THICKNESS),
        forces, fixed, plane="stress",
    )
    result = solve_continuum(model)
    centres = nodes[cells.long()].mean(dim=1)
    radius = torch.linalg.vector_norm(centres, dim=1)
    tolerance = max(INNER, OUTER) * (math.pi / 2.0 / n_theta / 2.0) ** 2
    exact = state(radius, tolerance=tolerance)
    sigma_r, sigma_theta, tau = cartesian_to_polar_stress(result.stress, centres)
    node_radius = torch.linalg.vector_norm(nodes, dim=1)
    u_radial = (
        result.displacement[0::2] * nodes[:, 0] + result.displacement[1::2] * nodes[:, 1]
    ) / node_radius
    return {
        "nodes": nodes,
        "cells": cells,
        "forces": forces,
        "result": result,
        "radius": radius,
        "sigma_r": sigma_r,
        "sigma_theta": sigma_theta,
        "tau": tau,
        "exact": exact,
        "u_radial": u_radial,
        "u_exact": state(node_radius).radial_displacement,
        "u_tangential": (
            -result.displacement[0::2] * nodes[:, 1] + result.displacement[1::2] * nodes[:, 0]
        ) / node_radius,
        "n_theta": n_theta,
        "n_radial": n_radial,
    }


# --------------------------------------------------------------------------- #
# the analytic reference
# --------------------------------------------------------------------------- #
def test_coefficients_leave_both_surfaces_traction_free():
    a_coef, b_coef = disc_coefficients(
        inner_radius=INNER, outer_radius=OUTER, density=DENSITY, omega=OMEGA, poisson=POISSON
    )
    faces = state(torch.tensor([INNER, OUTER], dtype=torch.float64))
    scale = bore_hoop_stress(
        inner_radius=INNER, outer_radius=OUTER, density=DENSITY, omega=OMEGA, poisson=POISSON
    )
    assert float(faces.radial_stress.abs().max()) / scale < 1e-6
    assert a_coef > 0.0 and b_coef > 0.0


def test_closed_form_satisfies_governing_ode_and_compatibility():
    radius = torch.linspace(INNER, OUTER, 2001, dtype=torch.float64)
    field = state(radius)
    a_coef, b_coef = field.coefficients
    scale = (3.0 + POISSON) / 4.0 * DENSITY * OMEGA**2 * OUTER
    d_sigma_r = 2.0 * b_coef / radius**3 - (3.0 + POISSON) / 4.0 * DENSITY * OMEGA**2 * radius
    equilibrium = (
        d_sigma_r + (field.radial_stress - field.hoop_stress) / radius + DENSITY * OMEGA**2 * radius
    )
    assert float(equilibrium.abs().max()) / scale < 1e-12

    eps_theta = (field.hoop_stress - POISSON * field.radial_stress) / YOUNG
    eps_radial = (field.radial_stress - POISSON * field.hoop_stress) / YOUNG
    d_r_eps = (
        (1.0 - POISSON) * a_coef
        - (1.0 + POISSON) * b_coef / radius**2
        - 3.0 * DENSITY * OMEGA**2 * radius**2 * (1.0 - POISSON**2) / 8.0
    ) / YOUNG
    assert float((d_r_eps - eps_radial).abs().max()) * YOUNG / scale < 1e-12


def test_solid_limit_matches_the_regular_michel_solution():
    probe = torch.tensor([0.2 * OUTER, 0.5 * OUTER, 0.9 * OUTER], dtype=torch.float64)
    annular = state(probe, inner=1e-4 * OUTER)
    solid = solid_rotating_disc_state(
        probe, outer_radius=OUTER, density=DENSITY, omega=OMEGA, young=YOUNG, poisson=POISSON
    )
    scale = float(solid.hoop_stress.abs().max())
    assert float((annular.hoop_stress - solid.hoop_stress).abs().max()) / scale < 1e-4
    assert float((annular.radial_stress - solid.radial_stress).abs().max()) / scale < 1e-4
    assert (
        float((annular.radial_displacement - solid.radial_displacement).abs().max())
        / float(solid.radial_displacement.abs().max())
        < 1e-4
    )


def test_central_hole_doubles_the_hoop_stress():
    small = 1e-4 * OUTER
    centre = float(
        solid_rotating_disc_state(
            torch.tensor([0.0]), outer_radius=OUTER, density=DENSITY, omega=OMEGA,
            young=YOUNG, poisson=POISSON,
        ).hoop_stress[0]
    )
    bore = bore_hoop_stress(
        inner_radius=small, outer_radius=OUTER, density=DENSITY, omega=OMEGA, poisson=POISSON
    )
    assert abs(bore / centre - 2.0) < 1e-3


def test_thin_ring_limit_is_rho_omega_squared_b_squared():
    thin_bore = 0.999 * OUTER
    thin = state(torch.linspace(thin_bore, OUTER, 17, dtype=torch.float64), inner=thin_bore)
    target = DENSITY * OMEGA**2 * OUTER**2
    assert float((thin.hoop_stress - target).abs().max() / target) < 5e-3


def test_bore_and_rim_traction_resultants_agree():
    traction = sector_hoop_traction_resultant(
        inner_radius=INNER, outer_radius=OUTER, density=DENSITY, omega=OMEGA,
        young=YOUNG, poisson=POISSON, thickness=THICKNESS,
    )
    body = sector_body_force_resultant(
        inner_radius=INNER, outer_radius=OUTER, density=DENSITY, omega=OMEGA, thickness=THICKNESS
    )
    assert abs(traction - body) / body < 1e-8


def test_reference_rejects_radii_outside_the_disc():
    with pytest.raises(ValueError):
        state(torch.tensor([1.1 * OUTER], dtype=torch.float64))
    with pytest.raises(ValueError):
        state(torch.tensor([0.5 * INNER], dtype=torch.float64))
    with pytest.raises(ValueError):
        rotating_disc_state(
            torch.tensor([0.2], dtype=torch.float64), inner_radius=0.0, outer_radius=OUTER,
            density=DENSITY, omega=OMEGA, young=YOUNG, poisson=POISSON,
        )


# --------------------------------------------------------------------------- #
# the centrifugal load path
# --------------------------------------------------------------------------- #
def test_consistent_load_integrates_exactly_on_the_polygonal_domain():
    nodes, cells = sector_q4_mesh(INNER, OUTER, 10, 12)
    forces = centrifugal_nodal_loads(
        nodes, cells, thickness=THICKNESS, density=DENSITY, omega=OMEGA
    )
    coords = nodes[cells.long()]
    exact = 0.0
    for quad in coords:
        for triangle in ((0, 1, 2), (0, 2, 3)):
            p = quad[list(triangle)]
            area = 0.5 * abs(
                float((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
            )
            exact += area * float(p[:, 0].mean())
    exact *= DENSITY * OMEGA**2 * THICKNESS
    assert abs(float(forces[0::2].sum()) - exact) / exact < 1e-12


def test_lumped_and_consistent_loads_share_the_same_resultant():
    nodes, cells = sector_q4_mesh(INNER, OUTER, 8, 8)
    consistent = centrifugal_nodal_loads(
        nodes, cells, thickness=THICKNESS, density=DENSITY, omega=OMEGA
    )
    lumped = centrifugal_nodal_loads(
        nodes, cells, thickness=THICKNESS, density=DENSITY, omega=OMEGA, consistent=False
    )
    assert torch.all(lumped >= 0.0)
    reference = float(consistent[0::2].sum())
    # equal-split lumping reproduces the 4-node mean of x, not the area centroid,
    # so the resultant agrees only to O(h^2) -- unlike the consistent rule, which
    # is exact on the polygonal domain.
    assert abs(float(lumped[0::2].sum()) - reference) / reference < 5e-3


def test_load_rejects_degenerate_geometry():
    nodes, cells = sector_q4_mesh(INNER, OUTER, 4, 2)
    flipped = cells[:, [0, 3, 2, 1]]
    with pytest.raises(ValueError):
        centrifugal_nodal_loads(nodes, flipped, thickness=THICKNESS, density=DENSITY, omega=OMEGA)


# --------------------------------------------------------------------------- #
# the finite-element agreement
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def graded_case():
    return run(12, 24, grading=1.6)


def test_hoop_stress_field_clears_the_gate(graded_case):
    exact = graded_case["exact"].hoop_stress
    field = graded_case["sigma_theta"]
    l2 = float(torch.linalg.vector_norm(field - exact) / torch.linalg.vector_norm(exact))
    peak = float((field - exact).abs().max()) / float(exact.abs().max())
    assert l2 < GATE
    assert peak < GATE


def test_radial_stress_and_displacement_clear_the_gate(graded_case):
    scale = float(graded_case["exact"].hoop_stress.abs().max())
    assert float((graded_case["sigma_r"] - graded_case["exact"].radial_stress).abs().max()) / scale < GATE
    u = graded_case["u_radial"]
    u_exact = graded_case["u_exact"]
    assert float((u - u_exact).abs().max() / u_exact.abs().max()) < GATE
    assert float(u.min()) > 0.0  # radial displacement is strictly outward


def test_symmetry_cuts_are_exact(graded_case):
    assert float(graded_case["u_tangential"].abs().max() / graded_case["u_radial"].abs().max()) < 1e-10
    scale = float(graded_case["exact"].hoop_stress.abs().max())
    assert float(graded_case["tau"].abs().max()) / scale < 1e-6


def test_bore_hoop_stress_extrapolation(graded_case):
    nodes = graded_case["nodes"]
    radius = graded_case["radius"]
    hoop = graded_case["sigma_theta"]
    ray = torch.arange(graded_case["n_radial"], dtype=torch.long) * graded_case["n_theta"]
    order = torch.argsort(radius[ray])
    rr, hh = radius[ray][order], hoop[ray][order]
    bore_exact = bore_hoop_stress(
        inner_radius=INNER, outer_radius=OUTER, density=DENSITY, omega=OMEGA, poisson=POISSON
    )
    x0, x1 = float(rr[0]) - INNER, float(rr[1]) - INNER
    y0, y1 = float(hh[0]), float(hh[1])
    linear = y0 + (y1 - y0) * (INNER - float(rr[0])) / (float(rr[1]) - float(rr[0]))
    x2, y2 = float(rr[2]) - INNER, float(hh[2])
    quadratic = (
        y0 * x1 * x2 / ((x0 - x1) * (x0 - x2))
        + y1 * x0 * x2 / ((x1 - x0) * (x1 - x2))
        + y2 * x0 * x1 / ((x2 - x0) * (x2 - x1))
    )
    # the two-point linear value is what a post-processor does by default and is
    # the worse of the two; the gated quantity in the benchmark is the quadratic
    assert abs(linear - bore_exact) / bore_exact < 0.02
    assert abs(quadratic - bore_exact) / bore_exact < 0.01
    assert bore_exact > 0.0


def test_symmetry_cut_reaction_matches_the_membrane_force(graded_case):
    n_theta, n_radial = graded_case["n_theta"], graded_case["n_radial"]
    ux_cut = torch.tensor(
        [2 * (i * (n_theta + 1) + n_theta) for i in range(n_radial + 1)], dtype=torch.long
    )
    reaction = -float(graded_case["result"].reaction[ux_cut].sum())
    analytic = sector_hoop_traction_resultant(
        inner_radius=INNER, outer_radius=OUTER, density=DENSITY, omega=OMEGA,
        young=YOUNG, poisson=POISSON, thickness=THICKNESS,
    )
    assert abs(reaction - analytic) / analytic < GATE
    assert abs(reaction - float(graded_case["forces"][0::2].sum())) / abs(reaction) < 1e-10


def test_twenty_percent_omega_error_is_detected():
    wrong = run(10, 16, load_omega=1.2 * OMEGA)
    exact = wrong["exact"].hoop_stress
    field = wrong["sigma_theta"]
    l2 = float(torch.linalg.vector_norm(field - exact) / torch.linalg.vector_norm(exact))
    assert l2 > GATE


def test_plane_strain_is_not_accepted_as_plane_stress():
    nodes, cells = sector_q4_mesh(INNER, OUTER, 10, 16)
    forces = centrifugal_nodal_loads(
        nodes, cells, thickness=THICKNESS, density=DENSITY, omega=OMEGA
    )
    fixed = symmetry_fixed_dofs(nodes, sector_angle=math.pi / 2.0, n_theta=10, n_radial=16)
    model = ContinuumModel(
        nodes, cells, torch.tensor(YOUNG), torch.tensor(POISSON), torch.tensor(THICKNESS),
        forces, fixed, plane="strain",
    )
    result = solve_continuum(model)
    centres = nodes[cells.long()].mean(dim=1)
    radius = torch.linalg.vector_norm(centres, dim=1)
    tolerance = max(INNER, OUTER) * (math.pi / 2.0 / 10.0 / 2.0) ** 2
    exact = state(radius, tolerance=tolerance)
    _, hoop, _ = cartesian_to_polar_stress(result.stress, centres)
    l2 = float(torch.linalg.vector_norm(hoop - exact.hoop_stress) / torch.linalg.vector_norm(exact.hoop_stress))
    assert l2 > GATE  # 1/(1-nu^2) on the stiffness is not a plane-stress model
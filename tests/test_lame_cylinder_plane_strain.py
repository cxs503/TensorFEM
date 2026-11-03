"""Verification of the plane-strain constitutive path via the Lame thick cylinder.

The reference is the closed-form Lame solution, checked here from three
independent directions (traction boundary conditions, pointwise equilibrium,
and strain compatibility with the constitutive law) before it is used to score
the finite element results.
"""
import math

import pytest
import torch

from tensorfem.continuum import ContinuumModel, solve_continuum
from tensorfem.lame_reference import (
    cartesian_to_polar_stress,
    circular_pressure_loads,
    lame_coefficients,
    lame_state,
    radial_component,
    sector_q4_mesh,
    symmetry_fixed_dofs,
)

DTYPE = torch.float64
INNER, OUTER = 0.5, 0.8
PRESSURE = 100.0e6
YOUNG, POISSON = 210.0e9, 0.3
GATE = 0.03


def _solve(n_theta, n_radial, *, plane="strain", grading=1.0, pressure=PRESSURE):
    nodes, cells = sector_q4_mesh(INNER, OUTER, n_theta, n_radial, radial_grading=grading)
    forces = circular_pressure_loads(n_theta, n_radial, INNER, pressure, inner=True)
    fixed = symmetry_fixed_dofs(nodes, n_theta=n_theta, n_radial=n_radial)
    model = ContinuumModel(
        nodes, cells, torch.tensor(YOUNG), torch.tensor(POISSON), torch.tensor(1.0),
        forces, fixed, plane=plane,
    )
    result = solve_continuum(model)
    displacement = result.displacement.reshape(-1, 2)
    radius = torch.linalg.vector_norm(nodes, dim=1)
    u_radial = radial_component(displacement, nodes)
    stride = n_theta + 1
    diag = torch.tensor([i * stride for i in range(n_radial + 1)])
    reference = lame_state(
        radius[diag], inner_radius=INNER, outer_radius=OUTER, inner_pressure=pressure,
        young_modulus=YOUNG, poisson_ratio=POISSON, plane=plane,
    )
    centres = nodes[cells].mean(dim=1)
    sigma_r, sigma_theta, tau = cartesian_to_polar_stress(result.stress, centres)
    stress_reference = lame_state(
        torch.linalg.vector_norm(centres, dim=1), inner_radius=INNER, outer_radius=OUTER,
        inner_pressure=pressure, young_modulus=YOUNG, poisson_ratio=POISSON, plane=plane,
    )
    return {
        "nodes": nodes,
        "cells": cells,
        "result": result,
        "radius": radius[diag],
        "u_radial": u_radial[diag],
        "u_reference": reference.radial_displacement,
        "displacement": displacement,
        "sigma_r": sigma_r,
        "sigma_theta": sigma_theta,
        "tau": tau,
        "sigma_r_reference": stress_reference.radial_stress,
        "sigma_theta_reference": stress_reference.hoop_stress,
    }


def _l2(field, reference):
    return (torch.linalg.vector_norm(field - reference) / torch.linalg.vector_norm(reference)).item()


def test_lame_coefficients_satisfy_traction_boundary_conditions():
    a_coef, b_coef = lame_coefficients(INNER, OUTER, PRESSURE, 0.0)
    sigma_r_bore = (a_coef - b_coef / INNER**2).item()
    sigma_r_outer = (a_coef - b_coef / OUTER**2).item()
    assert sigma_r_bore == pytest.approx(-PRESSURE, rel=1e-13)
    assert abs(sigma_r_outer) < 1e-6


def test_analytic_field_is_in_equilibrium_and_compatible():
    radii = torch.linspace(INNER, OUTER, 41, dtype=DTYPE)
    state = lame_state(
        radii, inner_radius=INNER, outer_radius=OUTER, inner_pressure=PRESSURE,
        young_modulus=YOUNG, poisson_ratio=POISSON, plane="strain",
    )
    h = 1e-9
    ahead = lame_state(radii + h, inner_radius=INNER, outer_radius=OUTER, inner_pressure=PRESSURE,
                       young_modulus=YOUNG, poisson_ratio=POISSON, plane="strain")
    behind = lame_state(radii - h, inner_radius=INNER, outer_radius=OUTER, inner_pressure=PRESSURE,
                        young_modulus=YOUNG, poisson_ratio=POISSON, plane="strain")
    d_sigma_r = (ahead.radial_stress - behind.radial_stress) / (2 * h)
    residual = d_sigma_r + (state.radial_stress - state.hoop_stress) / radii
    assert residual.abs().max().item() < 1e-3 * (PRESSURE / INNER)

    # Strain compatibility: eps_r = du/dr, eps_theta = u/r must reproduce the
    # in-plane stresses through the plane-strain (and plane-stress) constitutive law.
    eps_r = (ahead.radial_displacement - behind.radial_displacement) / (2 * h)
    eps_theta = state.radial_displacement / radii
    nu = POISSON
    c = YOUNG / ((1 + nu) * (1 - 2 * nu))
    sigma_r_strain = c * ((1 - nu) * eps_r + nu * eps_theta)
    assert torch.allclose(sigma_r_strain, state.radial_stress, rtol=1e-6, atol=1e-3 * PRESSURE)
    stress_state = lame_state(radii, inner_radius=INNER, outer_radius=OUTER, inner_pressure=PRESSURE,
                              young_modulus=YOUNG, poisson_ratio=POISSON, plane="stress")
    h_plane = h
    ahead_s = lame_state(radii + h_plane, inner_radius=INNER, outer_radius=OUTER, inner_pressure=PRESSURE,
                         young_modulus=YOUNG, poisson_ratio=POISSON, plane="stress")
    behind_s = lame_state(radii - h_plane, inner_radius=INNER, outer_radius=OUTER, inner_pressure=PRESSURE,
                          young_modulus=YOUNG, poisson_ratio=POISSON, plane="stress")
    eps_r_s = (ahead_s.radial_displacement - behind_s.radial_displacement) / (2 * h_plane)
    eps_theta_s = stress_state.radial_displacement / radii
    sigma_r_stress = YOUNG / (1 - nu**2) * (eps_r_s + nu * eps_theta_s)
    assert torch.allclose(sigma_r_stress, stress_state.radial_stress, rtol=1e-6, atol=1e-3 * PRESSURE)
    # Lame stress field does not depend on the plane assumption.
    assert torch.allclose(stress_state.hoop_stress, state.hoop_stress, rtol=1e-12)


def test_pressure_loads_are_self_equilibrated_and_correct():
    n_theta, n_radial = 24, 8
    forces = circular_pressure_loads(n_theta, n_radial, INNER, PRESSURE, inner=True)
    vector = forces.reshape(-1, 2).sum(dim=0)
    expected = 2.0 * PRESSURE * INNER * math.sin(math.pi / 4.0)
    assert vector.norm().item() == pytest.approx(expected, rel=1e-12)
    assert vector[0].item() == pytest.approx(vector[1].item(), rel=1e-12)  # along the bisector
    assert vector[0].item() > 0.0  # internal pressure pushes outward
    outer = circular_pressure_loads(n_theta, n_radial, OUTER, PRESSURE, inner=False)
    outer_vector = outer.reshape(-1, 2).sum(dim=0)
    expected_outer = 2.0 * PRESSURE * OUTER * math.sin(math.pi / 4.0)
    assert outer_vector.norm().item() == pytest.approx(expected_outer, rel=1e-12)
    assert outer_vector[0].item() < 0.0  # external pressure pushes inward


def test_q4_plane_strain_matches_lame_within_three_percent():
    case = _solve(48, 16)
    assert ((case["u_radial"] - case["u_reference"]).abs() / case["u_reference"].abs()).max().item() < GATE
    assert _l2(case["u_radial"], case["u_reference"]) < GATE
    assert _l2(case["sigma_r"], case["sigma_r_reference"]) < GATE
    assert _l2(case["sigma_theta"], case["sigma_theta_reference"]) < GATE
    assert ((case["sigma_r"] - case["sigma_r_reference"]).abs()
            / case["sigma_r_reference"].abs()).max().item() < GATE
    assert ((case["sigma_theta"] - case["sigma_theta_reference"]).abs()
            / case["sigma_theta_reference"].abs()).max().item() < GATE


def test_mesh_convergence_is_second_order():
    errors = []
    for n_theta, n_radial in ((12, 4), (24, 8), (48, 16)):
        case = _solve(n_theta, n_radial)
        errors.append(_l2(case["u_radial"], case["u_reference"]))
    assert errors[1] < errors[0] and errors[2] < errors[1]
    order = math.log(errors[1] / errors[2]) / math.log(2.0)
    assert order > 1.8, (errors, order)


def test_plane_strain_and_plane_stress_are_distinct_and_both_correct():
    strain = _solve(24, 8, plane="strain")
    stress = _solve(24, 8, plane="stress")
    assert _l2(strain["u_radial"], strain["u_reference"]) < GATE
    assert _l2(stress["u_radial"], stress["u_reference"]) < GATE
    gap = (strain["u_radial"] - stress["u_radial"]).abs().max().item() / strain["u_radial"].abs().max().item()
    assert gap > 1e-2, gap  # the switch must not be a silent no-op
    # Plane strain is the stiffer response at the bore.
    assert strain["u_radial"].abs().max().item() < stress["u_radial"].abs().max().item()


def test_discrete_solution_stays_axisymmetric():
    case = _solve(24, 8)
    assert case["tau"].abs().max().item() < 1e-8 * case["sigma_r_reference"].abs().max().item()
    nodes, radius = case["nodes"], torch.linalg.vector_norm(case["nodes"], dim=1)
    hoop = torch.stack((-nodes[:, 1], nodes[:, 0]), dim=1) / radius[:, None]
    u_hoop = torch.einsum("ij,ij->i", case["displacement"], hoop)
    assert u_hoop.abs().max().item() < 1e-8 * case["displacement"].abs().max().item()


def test_zero_pressure_gives_zero_displacement_and_balanced_reactions():
    case = _solve(24, 8, pressure=0.0)
    assert case["result"].displacement.abs().max().item() < 1e-18
    loaded = _solve(24, 8)
    reaction = loaded["result"].reaction.reshape(-1, 2).sum(dim=0)
    expected = 2.0 * PRESSURE * INNER * math.sin(math.pi / 4.0)
    assert reaction.norm().item() == pytest.approx(expected, rel=1e-10)


def test_radial_grading_keeps_the_bore_nodes_on_the_circle():
    nodes, cells = sector_q4_mesh(INNER, OUTER, 12, 6, radial_grading=2.0)
    radius = torch.linalg.vector_norm(nodes, dim=1)
    stride = 13
    assert torch.allclose(radius[torch.arange(stride)], torch.full((stride,), INNER, dtype=DTYPE))
    last_ring = 6 * stride + torch.arange(stride)
    assert torch.allclose(radius[last_ring], torch.full((stride,), OUTER, dtype=DTYPE))
    case = _solve(24, 8, grading=2.0)
    assert _l2(case["sigma_r"], case["sigma_r_reference"]) < GATE


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        lame_coefficients(OUTER, INNER, PRESSURE)
    with pytest.raises(ValueError):
        lame_state(torch.tensor([0.0], dtype=DTYPE), inner_radius=INNER, outer_radius=OUTER,
                   inner_pressure=PRESSURE, young_modulus=YOUNG, poisson_ratio=POISSON)
    with pytest.raises(ValueError):
        lame_state(torch.tensor([0.6], dtype=DTYPE), inner_radius=INNER, outer_radius=OUTER,
                   inner_pressure=PRESSURE, young_modulus=YOUNG, poisson_ratio=POISSON, plane="3d")
    with pytest.raises(ValueError):
        sector_q4_mesh(INNER, OUTER, 1, 4)
    with pytest.raises(ValueError):
        symmetry_fixed_dofs(torch.zeros(65, 2), n_theta=12, n_radial=4, sector_angle=math.pi / 3)
    with pytest.raises(ValueError):
        circular_pressure_loads(8, 4, INNER, -1.0, inner=True)

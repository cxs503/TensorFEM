"""Tests for the curved-bar (Winkler-Bach) verification case.

The analytic reference is validated independently of any finite element trace:
equilibrium, traction-free surfaces, the prescribed moment and the
strain-displacement relation must all close at round-off.  The FE checks then
run the cheapest ladder rungs so the file stays fast; the full study lives in
``benchmarks/curved_beam_pure_bending.py``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.continuum import ContinuumModel, solve_continuum
from tensorfem.curved_beam_reference import (
    curved_bar_coefficients,
    curved_bar_end_loads,
    curved_bar_gauge_dof,
    curved_bar_state,
    curved_bar_symmetry_dofs,
    eccentricity,
    neutral_radius,
    strain_residual,
    winkler_bach_hoop_stress,
)
from tensorfem.lame_reference import cartesian_to_polar_stress, sector_q4_mesh

INNER, OUTER, WIDTH = 1.0, 2.0, 1.0
MOMENT = 1.0e6
YOUNG, POISSON = 210.0e9, 0.3
SECTOR = math.pi / 4.0


def reference(radius, **kwargs):
    return curved_bar_state(
        radius, r_inner=INNER, r_outer=OUTER, width=WIDTH, moment=MOMENT,
        young=YOUNG, poisson=POISSON, plane="stress", **kwargs,
    )


def solve(n_theta: int, n_radial: int, *, pin: str = "exact"):
    nodes, cells = sector_q4_mesh(INNER, OUTER, n_theta, n_radial, sector_angle=SECTOR)
    forces = curved_bar_end_loads(
        nodes, n_theta, n_radial, r_inner=INNER, r_outer=OUTER, width=WIDTH,
        moment=MOMENT, sector_angle=SECTOR,
    )
    symmetry = curved_bar_symmetry_dofs(nodes, n_theta, n_radial)
    gauge = curved_bar_gauge_dof(nodes, n_theta, n_radial)
    gauge_radius = float(torch.linalg.vector_norm(nodes[gauge // 2]))
    value = (
        float(reference(torch.tensor([gauge_radius])).radial_displacement[0])
        if pin == "exact"
        else 0.0
    )
    fixed = torch.cat((symmetry, torch.tensor([gauge])))
    values = torch.cat((torch.zeros(len(symmetry)), torch.tensor([value])))
    model = ContinuumModel(
        nodes, cells, torch.tensor(YOUNG), torch.tensor(POISSON), torch.tensor(WIDTH),
        forces, fixed, plane="stress", prescribed_values=values,
    )
    result = solve_continuum(model)
    centres = nodes[cells.long()].mean(dim=1)
    sigma_r, sigma_theta, tau_rtheta = cartesian_to_polar_stress(result.stress, centres)
    return nodes, cells, result, centres, sigma_r, sigma_theta, tau_rtheta


def test_neutral_axis_shift_matches_closed_form():
    """r_n = h/ln(r_o/r_i) and e = r_bar - r_n for the r_o/r_i = 2 rectangle."""
    r_n = neutral_radius(INNER, OUTER)
    assert r_n == pytest.approx(1.0 / math.log(2.0), rel=1e-14)
    assert r_n == pytest.approx(1.4426950408889634, rel=1e-12)
    e = eccentricity(INNER, OUTER)
    assert e == pytest.approx(1.5 - 1.0 / math.log(2.0), rel=1e-14)
    # the shift is a few percent of the section height, which is why it matters
    assert 0.05 < e / (OUTER - INNER) < 0.06


def test_reference_closes_equilibrium_and_boundary_conditions():
    """Exact solution: radial equilibrium, free surfaces, moment and net force."""
    radius = torch.linspace(INNER, OUTER, 501, dtype=torch.float64)
    state = reference(radius)
    a_coef, b_coef, _ = (float(x) for x in state.coefficients)
    dsig = -2.0 * a_coef / radius**3 + 2.0 * b_coef / radius
    equilibrium = dsig + (state.radial_stress - state.hoop_stress) / radius
    scale = float(state.hoop_stress.abs().max()) / INNER
    assert float(equilibrium.abs().max()) / scale < 1e-12

    # ASSERTION-REBASELINE (m2618, flagged for owner): sigma_r at the free
    # faces cancels analytically, but the cancellation is between terms of
    # magnitude ~1e7, so the computed value lands within ~1 ulp of zero
    # (measured 1.8626e-9 = 1 ulp of float64 at 7.8e6, CI py3.11 x86; exact
    # 0.0 on py3.12 x86 and loong64).  Judge it against the same
    # relative-to-scale tolerance as the equilibrium residual above instead
    # of float equality, which is rounding-build luck.
    faces = reference(torch.tensor([INNER, OUTER], dtype=torch.float64))
    assert float(faces.radial_stress.abs().max()) / scale < 1e-12

    fine = torch.linspace(INNER, OUTER, 200001, dtype=torch.float64)
    fine_state = reference(fine)
    moment = WIDTH * float(
        torch.trapezoid(fine_state.hoop_stress * (0.5 * (INNER + OUTER) - fine), fine)
    )
    force = WIDTH * float(torch.trapezoid(fine_state.hoop_stress, fine))
    assert abs(moment - MOMENT) / MOMENT < 1e-6
    assert abs(force) / (MOMENT / (OUTER - INNER)) < 1e-9

    # hoop tension at the bore, compression outside, for the positive convention
    assert float(reference(torch.tensor([INNER])).hoop_stress[0]) > 0.0
    assert float(reference(torch.tensor([OUTER])).hoop_stress[0]) < 0.0


def test_displacement_field_is_compatible_with_the_stress_field():
    radius = torch.linspace(INNER, OUTER, 25, dtype=torch.float64)
    residual = strain_residual(
        radius, r_inner=INNER, r_outer=OUTER, width=WIDTH, moment=MOMENT,
        young=YOUNG, poisson=POISSON, plane="stress",
    )
    assert residual["eps_r_relative"] < 1e-7
    assert residual["eps_theta_relative"] < 1e-12


def test_winkler_bach_is_an_approximation_not_the_exact_solution():
    """The textbook closed form neglects sigma_r; for r_o/r_i = 2 that is ~0.7 %."""
    radius = torch.linspace(INNER, OUTER, 201, dtype=torch.float64)
    exact = reference(radius)
    web = winkler_bach_hoop_stress(
        radius, r_inner=INNER, r_outer=OUTER, width=WIDTH, moment=MOMENT
    )
    scale = float(exact.hoop_stress.abs().max())
    deviation = float((exact.hoop_stress - web).abs().max()) / scale
    radial_share = float(exact.radial_stress.abs().max()) / scale
    # small enough to be engineering-useful, large enough to be worth reporting
    assert 1e-3 < deviation < 0.03
    assert radial_share > 0.1
    assert deviation == pytest.approx(0.00701365898496714, rel=1e-6)


def test_airy_coefficients_satisfy_the_moment_condition():
    coefficients = curved_bar_coefficients(INNER, OUTER, WIDTH, MOMENT)
    a_coef, b_coef, c_coef = (float(x) for x in coefficients)

    def sigma(radius: float) -> float:
        return -a_coef / radius**2 + 2.0 * b_coef * math.log(radius) + 3.0 * b_coef + 2.0 * c_coef

    for face in (INNER, OUTER):
        radial = a_coef / face**2 + 2.0 * b_coef * math.log(face) + b_coef + 2.0 * c_coef
        assert abs(radial) < 1e-6 * abs(sigma(INNER))
    radius = torch.linspace(INNER, OUTER, 200001, dtype=torch.float64)
    integrand = torch.tensor([sigma(float(r)) for r in radius], dtype=torch.float64)
    moment = WIDTH * float(torch.trapezoid(integrand * (0.5 * (INNER + OUTER) - radius), radius))
    assert moment == pytest.approx(MOMENT, rel=1e-6)


def test_end_loads_are_a_pure_couple():
    n_theta, n_radial = 8, 16
    nodes, _ = sector_q4_mesh(INNER, OUTER, n_theta, n_radial, sector_angle=SECTOR)
    forces = curved_bar_end_loads(
        nodes, n_theta, n_radial, r_inner=INNER, r_outer=OUTER, width=WIDTH,
        moment=MOMENT, sector_angle=SECTOR,
    ).reshape(-1, 2)
    # zero net force, and a couple equal to -M about the polar origin (the sign
    # follows from M = w*int sigma_theta (r_bar - r) dr = -M_z)
    assert abs(float(forces.sum())) / MOMENT < 1e-9
    loaded = (forces.abs().sum(dim=1) > 0).nonzero().reshape(-1)
    moment_z = float((nodes[loaded, 0] * forces[loaded, 1] - nodes[loaded, 1] * forces[loaded, 0]).sum())
    assert moment_z == pytest.approx(-MOMENT, rel=1e-6)
    # every loaded node sits on the loaded radial face
    angles = torch.atan2(nodes[loaded, 1], nodes[loaded, 0])
    assert torch.allclose(angles, torch.full_like(angles, SECTOR), atol=1e-12)


def test_symmetry_constraints_only_touch_the_symmetry_ray():
    n_theta, n_radial = 8, 16
    nodes, _ = sector_q4_mesh(INNER, OUTER, n_theta, n_radial, sector_angle=SECTOR)
    fixed = curved_bar_symmetry_dofs(nodes, n_theta, n_radial)
    assert len(fixed) == n_radial + 1
    # all constraints are u_y on the theta = 0 ray (node ids i * (n_theta + 1))
    assert all(int(dof) % 2 == 1 for dof in fixed)
    ray_nodes = [int(dof) // 2 for dof in fixed]
    assert ray_nodes == [i * (n_theta + 1) for i in range(n_radial + 1)]
    assert torch.all(nodes[ray_nodes, 1] == 0)
    gauge = curved_bar_gauge_dof(nodes, n_theta, n_radial)
    assert gauge % 2 == 0 and gauge not in set(int(d) for d in fixed)
    assert float(nodes[gauge // 2, 1]) == 0.0


def test_coarse_mesh_reproduces_the_exact_solution():
    n_theta, n_radial = 16, 32
    nodes, cells, result, centres, sigma_r, sigma_theta, tau = solve(n_theta, n_radial)
    radius = torch.linalg.vector_norm(centres, dim=1)
    exact = reference(radius)
    hoop_l2 = float(
        torch.linalg.vector_norm(sigma_theta - exact.hoop_stress) / torch.linalg.vector_norm(exact.hoop_stress)
    )
    radial_l2 = float(
        torch.linalg.vector_norm(sigma_r - exact.radial_stress) / torch.linalg.vector_norm(exact.radial_stress)
    )
    scale = float(exact.hoop_stress.abs().max())
    assert hoop_l2 < 0.03
    assert radial_l2 < 0.03
    assert float(tau.abs().max()) / scale < 0.01

    ray_nodes = torch.arange(0, n_radial + 1) * (n_theta + 1)
    u_x = result.displacement[torch.tensor([2 * int(i) for i in ray_nodes])]
    u_exact = reference(torch.linalg.vector_norm(nodes[ray_nodes], dim=1)).radial_displacement
    assert float((u_x - u_exact).abs().max() / u_exact.abs().max()) < 0.005


def test_rigid_body_gauge_does_not_change_the_stress_field():
    _, _, _, centres_ref, _, sigma_theta_ref, _ = solve(8, 16, pin="exact")
    _, _, _, _, _, sigma_theta_zero, _ = solve(8, 16, pin="zero")
    scale = float(sigma_theta_ref.abs().max())
    assert float((sigma_theta_ref - sigma_theta_zero).abs().max()) / scale < 1e-10


def test_reference_moment_sign_flips_the_fibre_stresses():
    positive = reference(torch.tensor([INNER, OUTER]))
    negative = curved_bar_state(
        torch.tensor([INNER, OUTER]), r_inner=INNER, r_outer=OUTER, width=WIDTH,
        moment=-MOMENT, young=YOUNG, poisson=POISSON, plane="stress",
    )
    assert torch.allclose(negative.hoop_stress, -positive.hoop_stress)


def test_coarse_mesh_error_decreases_with_refinement():
    """Fail-closed guard: the study must not silently lose its convergence."""
    def hoop_error(n_theta: int, n_radial: int) -> float:
        _, _, _, centres, _, sigma_theta, _ = solve(n_theta, n_radial)
        radius = torch.linalg.vector_norm(centres, dim=1)
        exact = reference(radius)
        return float(
            torch.linalg.vector_norm(sigma_theta - exact.hoop_stress)
            / torch.linalg.vector_norm(exact.hoop_stress)
        )

    coarse = hoop_error(4, 8)
    fine = hoop_error(8, 16)
    assert coarse > 0.03  # the coarse rung genuinely fails the gate
    assert fine < coarse / 2.0
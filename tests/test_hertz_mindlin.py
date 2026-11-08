"""Verification tests for Hertz / Cattaneo-Mindlin contact (P22-N02).

Every assertion below is a physics gate, not a smoke test: the closed-form Hertz
and Cattaneo references are computed independently of the solver (plain algebra
in this file), and the solver only ever sees the geometry, the material, the
load and the unilateral constraint.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.hertz_mindlin import (
    ElasticSolver, block_mesh, dirichlet_dofs, half_plane_influence,
    hertz_cylinder_reference, hertz_sphere_reference, plane_strain_modulus,
    plane_strain_stiffness, segment_grid, solve_cattaneo_mindlin_2d,
    solve_hertz_2d, solve_hertz_sphere_3d, solve_indentation_fe,
    stick_half_width_2d, surface_nodes, _disc_settlement, _log_segment_integral,
)

YOUNG = 210.0e9
POISSON = 0.3
RADIUS = 0.05
CYL_LOAD = 3.625e6           # N/m
FRICTION = 0.3


# --------------------------------------------------------------------------- #
# reference algebra (independent of the solver)
# --------------------------------------------------------------------------- #
def test_hertz_cylinder_reference_algebra():
    e_star = YOUNG / (1.0 - POISSON ** 2)
    ref = hertz_cylinder_reference(load=CYL_LOAD, radius=RADIUS, effective_modulus=e_star)
    b = math.sqrt(4.0 * CYL_LOAD * RADIUS / (math.pi * e_star))
    assert ref.half_width == pytest.approx(b, rel=1e-14)
    assert ref.peak_pressure == pytest.approx(2.0 * CYL_LOAD / (math.pi * b), rel=1e-14)
    # Hertz pressure integrates to the load
    x = np.linspace(-b, b, 200001)
    assert np.trapezoid(ref.pressure(x), x) == pytest.approx(CYL_LOAD, rel=1e-6)


def test_hertz_sphere_reference_algebra():
    e_star = YOUNG / (1.0 - POISSON ** 2)
    load = 5.0e5
    ref = hertz_sphere_reference(load=load, radius=RADIUS, effective_modulus=e_star)
    a = (3.0 * load * RADIUS / (4.0 * e_star)) ** (1.0 / 3.0)
    assert ref.contact_radius == pytest.approx(a, rel=1e-13)
    assert ref.peak_pressure == pytest.approx(3.0 * load / (2.0 * math.pi * a * a), rel=1e-13)
    assert ref.approach == pytest.approx(a * a / RADIUS, rel=1e-13)
    # load-approach consistency: P(delta) = 4/3 E* sqrt(R) delta^1.5
    assert ref.load == pytest.approx(4.0 / 3.0 * e_star * math.sqrt(RADIUS) * ref.approach ** 1.5,
                                     rel=1e-13)


def test_cattaneo_reference_algebra():
    b = 1.0e-3
    c = stick_half_width_2d(b, 0.4 * FRICTION * CYL_LOAD, CYL_LOAD, FRICTION)
    assert c == pytest.approx(b * math.sqrt(0.6), rel=1e-14)
    assert stick_half_width_2d(b, 0.0, CYL_LOAD, FRICTION) == pytest.approx(b, rel=1e-14)
    assert stick_half_width_2d(b, FRICTION * CYL_LOAD, CYL_LOAD, FRICTION) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
def test_log_segment_integral_matches_quadrature():
    edges = segment_grid(1.0, 8)
    centres = 0.5 * (edges[:-1] + edges[1:])
    table = _log_segment_integral(centres, edges[:-1], edges[1:])
    s = np.linspace(0.0, 1.0, 4001)
    for i in (0, 3, 7):
        for j in (0, 4, 6):
            a, b = edges[j], edges[j + 1]
            x = centres[i]
            if a < x < b:
                continue
            values = np.log(np.abs(x - (a + (b - a) * s)))
            assert table[i, j] == pytest.approx(np.trapezoid(values, a + (b - a) * s), rel=1e-6)


def test_disc_settlement_centre_and_continuity():
    # Love's closed form: centre settlement of a uniformly loaded disc
    # u(0) = 2 (1-nu^2) p a / E  (unit pressure here)
    young, nu, a = 210.0e9, 0.3, 1.0e-3
    inside = _disc_settlement(np.array([0.0]), np.array([a]), young, nu)[0, 0]
    assert inside == pytest.approx(2.0 * (1.0 - nu ** 2) * a / young, rel=1e-12)
    # continuity across the disc edge
    radii = np.array([a - 1e-12, a + 1e-12])
    values = _disc_settlement(radii, np.array([a]), young, nu)[:, 0]
    assert values[0] == pytest.approx(values[1], rel=1e-9)


def test_half_plane_influence_is_positive_definite():
    edges = segment_grid(3.0e-3, 60)
    centres = 0.5 * (edges[:-1] + edges[1:])
    S = half_plane_influence(centres, edges, plane_strain_modulus(YOUNG, POISSON), None)
    assert np.allclose(S, S.T, rtol=1e-12, atol=0.0)
    assert np.min(np.linalg.eigvalsh(S)) > 0.0


# --------------------------------------------------------------------------- #
# E1 / E2: Hertz contact radius and peak pressure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rings", [80, 200])
def test_hertz_3d_contact_radius_and_peak_pressure(rings):
    e_star = plane_strain_modulus(YOUNG, POISSON)
    load = 5.0e5
    sol = solve_hertz_sphere_3d(radius=RADIUS, young=YOUNG, poisson=POISSON,
                                load=load, rings=rings)
    ref = hertz_sphere_reference(load=sol.load, radius=RADIUS, effective_modulus=e_star)
    assert sol.converged
    assert abs(sol.half_width / ref.contact_radius - 1.0) <= 0.02          # E1
    assert abs(sol.peak_pressure / ref.peak_pressure - 1.0) <= 0.05        # E2
    assert sol.load == pytest.approx(load, rel=1e-10)                      # exact load control


def test_hertz_2d_cylinder_half_width_and_pressure():
    e_star = plane_strain_modulus(YOUNG, POISSON)
    sol = solve_hertz_2d(load=CYL_LOAD, radius=RADIUS, young=YOUNG, poisson=POISSON,
                         segments=400)
    ref = hertz_cylinder_reference(load=sol.load, radius=RADIUS, effective_modulus=e_star)
    assert sol.converged
    assert abs(sol.half_width / ref.half_width - 1.0) <= 0.02
    assert abs(sol.peak_pressure / ref.peak_pressure - 1.0) <= 0.05
    assert np.all(sol.pressure >= 0.0)


# --------------------------------------------------------------------------- #
# E3: load-approach law
# --------------------------------------------------------------------------- #
def test_hertz_3d_load_approach_power_law():
    e_star = plane_strain_modulus(YOUNG, POISSON)
    loads = [1.0e5, 2.0e5, 4.0e5, 8.0e5]
    approaches, computed, analytic = [], [], []
    for load in loads:
        sol = solve_hertz_sphere_3d(radius=RADIUS, young=YOUNG, poisson=POISSON,
                                    load=load, rings=200)
        ref = hertz_sphere_reference(approach=sol.indentation, radius=RADIUS,
                                     effective_modulus=e_star)
        approaches.append(sol.indentation)
        computed.append(sol.load)
        analytic.append(ref.load)
    computed = np.array(computed)
    analytic = np.array(analytic)
    r2 = 1.0 - np.sum((computed - analytic) ** 2) / np.sum((analytic - analytic.mean()) ** 2)
    assert r2 >= 0.99                                                      # E3
    exponent = np.polyfit(np.log(approaches), np.log(computed), 1)[0]
    assert exponent == pytest.approx(1.5, abs=1e-3)                        # Hertz 3/2 law


# --------------------------------------------------------------------------- #
# E4: Cattaneo-Mindlin stick/slip boundary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fraction", [0.25, 0.7])
def test_cattaneo_mindlin_stick_boundary(fraction):
    tangential = fraction * FRICTION * CYL_LOAD
    sol = solve_cattaneo_mindlin_2d(normal_load=CYL_LOAD, tangential_load=tangential,
                                    friction=FRICTION, radius=RADIUS, young=YOUNG,
                                    poisson=POISSON, segments=800)
    c_ref = stick_half_width_2d(sol.contact_half_width, tangential, CYL_LOAD, FRICTION)
    assert sol.converged
    assert sol.tangential_load == pytest.approx(tangential, rel=1e-8)
    assert abs(sol.stick_half_width / c_ref - 1.0) <= 0.05                 # E4
    # traction inside the Coulomb cone, and exactly at the cone in the slip strips
    cone = FRICTION * sol.pressure
    assert np.all(sol.traction[sol.stick] <= cone[sol.stick] + 1e-6 * np.max(cone))
    slip = sol.active & ~sol.stick
    if slip.any():
        assert np.allclose(sol.traction[slip], cone[slip], rtol=1e-10)
    # monotone: the stick zone shrinks as the tangential load grows
    tighter = solve_cattaneo_mindlin_2d(
        normal_load=CYL_LOAD, tangential_load=min(0.95 * FRICTION * CYL_LOAD,
                                                1.4 * tangential),
        friction=FRICTION, radius=RADIUS, young=YOUNG, poisson=POISSON, segments=800)
    assert tighter.stick_half_width < sol.stick_half_width


def test_cattaneo_rejects_outside_friction_cone():
    with pytest.raises(ValueError):
        solve_cattaneo_mindlin_2d(normal_load=CYL_LOAD,
                                  tangential_load=1.05 * FRICTION * CYL_LOAD,
                                  friction=FRICTION, radius=RADIUS, young=YOUNG,
                                  poisson=POISSON, segments=200)


# --------------------------------------------------------------------------- #
# E5 / E6: complementarity residual and the no-contact negative control
# --------------------------------------------------------------------------- #
def test_complementarity_residual_small():
    sol2 = solve_hertz_2d(load=CYL_LOAD, radius=RADIUS, young=YOUNG, poisson=POISSON,
                          segments=800)
    sol3 = solve_hertz_sphere_3d(radius=RADIUS, young=YOUNG, poisson=POISSON,
                                 load=5.0e5, rings=200)
    assert sol2.complementarity_residual <= 1e-8                             # E5
    assert sol3.complementarity_residual <= 1e-8                             # E5
    # the two unilateral conditions hold pointwise on the discrete solution
    assert np.all(sol2.gap >= -1e-15)
    assert np.all(sol2.pressure >= 0.0)
    assert np.all(sol2.gap[sol2.active] <= 1e-15)
    assert np.all(sol2.pressure[~sol2.active] == 0.0)


def test_no_contact_gives_exactly_zero_force():
    sol2 = solve_hertz_2d(load=CYL_LOAD, radius=RADIUS, young=YOUNG, poisson=POISSON,
                          segments=200, indentation=-1.0e-6)
    sol3 = solve_hertz_sphere_3d(radius=RADIUS, young=YOUNG, poisson=POISSON,
                                 indentation=-1.0e-6, rings=100)
    assert sol2.load == 0.0                                                  # E6
    assert sol3.load == 0.0                                                  # E6
    assert not sol2.active.any() and not sol3.active.any()
    assert np.all(sol2.gap > 0.0) and np.all(sol3.gap > 0.0)


# --------------------------------------------------------------------------- #
# finite-element companion: patch test + independent contact cross-check
# --------------------------------------------------------------------------- #
def test_plane_strain_fe_patch_test():
    """Uniaxial-strain patch test: u_y(top) = -q H / (lambda + 2 mu)."""
    young, nu = YOUNG, POISSON
    lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = young / (2.0 * (1.0 + nu))
    nx, ny, width, depth, traction = 12, 10, 1.0, 1.0, 1.0e6
    nodes, elems = block_mesh(nx, ny, width, depth, grading=1.0)
    K = plane_strain_stiffness(nodes, elems, young, nu)
    solver = ElasticSolver(K, dirichlet_dofs(nodes, ny))
    surf = surface_nodes(nodes, ny)
    loads = np.zeros(len(solver.free))
    x = nodes[surf, 0]
    order = np.argsort(x)
    for k in range(len(surf) - 1):
        a, b = surf[order[k]], surf[order[k + 1]]
        length = x[order[k + 1]] - x[order[k]]
        for node in (a, b):
            dof = solver.remap[2 * node + 1]
            if dof >= 0:
                loads[dof] -= 0.5 * traction * length
    u = solver.solve(loads)
    exact = -traction * depth / (lam + 2.0 * mu)
    assert np.allclose(u[2 * surf + 1], exact, rtol=1e-10, atol=0.0)


def test_fe_contact_cross_check_half_width():
    """Independent Q4 node-to-node contact solve of the same Hertz problem."""
    e_star = plane_strain_modulus(YOUNG, POISSON)
    ref = hertz_cylinder_reference(load=CYL_LOAD, radius=RADIUS, effective_modulus=e_star)
    nx, ny = 60, 40
    nodes, elems = block_mesh(nx, ny, 30.0 * ref.half_width, 20.0 * ref.half_width)
    K = plane_strain_stiffness(nodes, elems, YOUNG, POISSON)
    surf = surface_nodes(nodes, ny)
    solver = ElasticSolver(K, dirichlet_dofs(nodes, ny))
    def fe_load(depth):
        return solve_indentation_fe(nodes, elems, K, surf, RADIUS, depth,
                                    solver=solver)["load"]

    lo, hi = 0.0, 1.0e-6
    while fe_load(hi) < CYL_LOAD:
        hi *= 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fe_load(mid) < CYL_LOAD:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    state = solve_indentation_fe(nodes, elems, K, surf, RADIUS, 0.5 * (lo + hi),
                                 solver=solver)
    assert state["converged"]
    assert state["load"] == pytest.approx(CYL_LOAD, rel=1e-4)
    assert state["contact_extent"] == pytest.approx(ref.half_width, rel=0.20)
    assert state["complementarity_residual"] <= 1e-8
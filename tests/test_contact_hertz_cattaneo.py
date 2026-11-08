"""Verification tests for Hertz / Cattaneo--Mindlin / complementarity kernels.

Fast mesh (h/a = 1/6) so the suite stays in seconds; the production ladder with
three mesh levels and GCI lives in
``benchmarks/contact_hertz_cattaneo_study.py``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.cattaneo_hertz import (
    axisymmetric_compliance,
    cattaneo_displacement,
    cattaneo_stick_radius,
    cattaneo_stiffness_small_q,
    cattaneo_traction,
    complementarity_residual,
    hertz_line,
    hertz_line_approach_quadrature,
    hertz_point,
    hertz_point_from_indentation,
    hertz_traction,
    hertz_work,
    plane_strain_compliance,
    recover_hertz_pressure,
    solve_cattaneo_tangential,
    solve_normal_alm,
    solve_normal_penalty,
)

DTYPE = torch.float64
YOUNG = 210.0e9
POISSON = 0.3
RADIUS = 0.05
E_STAR = YOUNG / (1.0 - POISSON ** 2)
LOAD = 100.0


@pytest.fixture(scope="module")
def point_compliance():
    return axisymmetric_compliance(n_in=12, nz_in=12, poisson=POISSON, extent=2.0,
                                   total=20.0, growth=1.3)


@pytest.fixture(scope="module")
def line_compliance():
    return plane_strain_compliance(n_in=12, nz_in=12, poisson=POISSON, extent=2.0,
                                   total=20.0, growth=1.3)


def _indent(comp, load=LOAD):
    oracle = hertz_point(load, RADIUS, E_STAR)
    young_fe = (1.0 - POISSON ** 2) * E_STAR
    radius_hat = RADIUS / oracle["a"]
    load_hat = load / (young_fe * oracle["a"] ** 2)
    delta = 1.0 / radius_hat
    sol = None
    for _ in range(30):
        offset = comp.coords ** 2 / (2.0 * radius_hat) - delta
        sol = solve_normal_alm(comp.normal, offset, comp.measures, tolerance=1e-12)
        ratio = float(sol.load) / load_hat
        if abs(ratio - 1.0) < 1e-11:
            break
        delta *= (1.0 / ratio) ** (2.0 / 3.0)
    return oracle, sol, radius_hat, delta, young_fe


# --------------------------------------------------------------------------
# closed-form oracles
# --------------------------------------------------------------------------
def test_hertz_point_inverse_roundtrip():
    hz = hertz_point(LOAD, RADIUS, E_STAR)
    back = hertz_point_from_indentation(hz["delta"], RADIUS, E_STAR)
    assert back["load"] == pytest.approx(LOAD, rel=1e-12)
    assert back["a"] == pytest.approx(hz["a"], rel=1e-12)
    # p0 = 3F/(2 pi a^2) and delta = a^2/R must hold identically
    assert hz["p0"] == pytest.approx(3.0 * LOAD / (2.0 * math.pi * hz["a"] ** 2), rel=1e-12)
    assert hz["delta"] == pytest.approx(hz["a"] ** 2 / RADIUS, rel=1e-12)


def test_hertz_point_rejects_bad_input():
    with pytest.raises(ValueError):
        hertz_point(0.0, RADIUS, E_STAR)
    with pytest.raises(ValueError):
        hertz_point_from_indentation(-1e-6, RADIUS, E_STAR)


def test_hertz_line_relations():
    load_per_length = LOAD / 1.0e-2
    hz = hertz_line(load_per_length, RADIUS, E_STAR)
    assert hz["b"] == pytest.approx(
        math.sqrt(4.0 * load_per_length * RADIUS / (math.pi * E_STAR)), rel=1e-12)
    assert hz["p0"] == pytest.approx(2.0 * load_per_length / (math.pi * hz["b"]), rel=1e-12)
    # the closed-form approach is a log-reference convention; the Flamant
    # quadrature at reference = R must agree to a few percent
    quad = hertz_line_approach_quadrature(load_per_length, RADIUS, E_STAR, reference=RADIUS)
    assert abs(quad / hz["delta"] - 1.0) < 0.06


def test_hertz_work_limits():
    hz = hertz_point(LOAD, RADIUS, E_STAR)
    # U = int_0^delta F d(delta) = (2/5) F delta for the Hertz relation F ~ delta^1.5
    assert hertz_work(LOAD, hz["delta"]) == pytest.approx(0.4 * LOAD * hz["delta"])


def test_cattaneo_limits():
    a = 2.5e-4
    shear = YOUNG / (2.0 * (1.0 + POISSON))
    # Q -> 0 must reproduce the Mindlin tangential stiffness 8 G a/(2-nu)
    tiny = 1e-9 * LOAD
    u = cattaneo_displacement(tiny, LOAD, 0.3, a, shear, POISSON)
    stiffness = cattaneo_stiffness_small_q(LOAD, a, shear, POISSON)
    assert tiny / u == pytest.approx(stiffness, rel=1e-6)
    # full sliding: c -> 0
    assert cattaneo_stick_radius(0.3 * LOAD, LOAD, 0.3, a) == pytest.approx(0.0, abs=1e-15)
    # stick radius monotone in Q
    radii = [cattaneo_stick_radius(q, LOAD, 0.3, a) for q in (0.0, 0.1 * LOAD, 0.2 * LOAD)]
    assert radii[0] > radii[1] > radii[2]
    with pytest.raises(ValueError):
        cattaneo_stick_radius(1.1 * 0.3 * LOAD, LOAD, 0.3, a)


def test_cattaneo_traction_bounded_by_coulomb():
    a = 1.0
    c = 0.6
    r = torch.linspace(0.0, 1.0, 201, dtype=DTYPE)
    p0 = 3.0 * LOAD / (2.0 * math.pi * a * a)
    q = cattaneo_traction(r, a, c, p0, 0.3)
    limit = 0.3 * hertz_traction(r, a, p0)
    assert torch.all(q <= limit + 1e-12 * float(limit.max()))
    assert float(q.max()) > 0.0


# --------------------------------------------------------------------------
# FE kernels
# --------------------------------------------------------------------------
def test_recover_hertz_pressure_from_analytic_field():
    coords = torch.linspace(0.02, 1.0, 60, dtype=DTYPE)
    pressure = hertz_traction(coords, 1.0, 1.0)
    a, p0 = recover_hertz_pressure(pressure, coords)
    assert a == pytest.approx(1.0, rel=2e-2)
    assert p0 == pytest.approx(1.0, rel=1e-3)


def test_alm_satisfies_complementarity(point_compliance):
    oracle, sol, radius_hat, delta, young_fe = _indent(point_compliance)
    assert sol.converged
    assert sol.residual <= 1e-8
    # exact non-penetration: no node overlaps the indenter
    assert float(sol.gap.min()) > -1e-9 * delta
    assert float(sol.nodal_force.min()) >= 0.0
    # normal force balance
    assert float(sol.load) / (LOAD / (young_fe * oracle["a"] ** 2)) == pytest.approx(
        1.0, rel=1e-9)


def test_penalty_and_alm_agree_within_engineering_tolerance(point_compliance):
    oracle, alm, radius_hat, delta, young_fe = _indent(point_compliance)
    offset = point_compliance.coords ** 2 / (2.0 * radius_hat) - delta
    h_scale = float(torch.median(torch.diagonal(point_compliance.normal)))
    penalty = solve_normal_penalty(point_compliance.normal, offset, 1.0e3 / h_scale,
                                   point_compliance.measures)
    a_alm, p0_alm = recover_hertz_pressure(alm.pressure, point_compliance.coords)
    a_pen, p0_pen = recover_hertz_pressure(penalty.pressure, point_compliance.coords)
    assert a_pen == pytest.approx(a_alm, rel=5e-3)
    assert p0_pen == pytest.approx(p0_alm, rel=5e-3)


def test_penalty_residual_halves_with_doubling(point_compliance):
    oracle = hertz_point(LOAD, RADIUS, E_STAR)
    radius_hat = RADIUS / oracle["a"]
    offset = point_compliance.coords ** 2 / (2.0 * radius_hat) - 1.0 / radius_hat
    h_scale = float(torch.median(torch.diagonal(point_compliance.normal)))
    residuals = []
    for factor in (1.0e2, 2.0e2, 4.0e2, 8.0e2):
        sol = solve_normal_penalty(point_compliance.normal, offset, factor / h_scale,
                                   point_compliance.measures)
        residuals.append(sol.residual)
    for coarse, fine in zip(residuals, residuals[1:]):
        # exact halving up to the discrete active-set round-off (2 % band)
        assert fine <= 0.51 * coarse
        assert fine >= 0.40 * coarse


def test_penalty_condition_number_grows_with_stiffness(point_compliance):
    oracle = hertz_point(LOAD, RADIUS, E_STAR)
    radius_hat = RADIUS / oracle["a"]
    offset = point_compliance.coords ** 2 / (2.0 * radius_hat) - 1.0 / radius_hat
    h_scale = float(torch.median(torch.diagonal(point_compliance.normal)))
    low = solve_normal_penalty(point_compliance.normal, offset, 1.0e1 / h_scale,
                               point_compliance.measures)
    high = solve_normal_penalty(point_compliance.normal, offset, 1.0e4 / h_scale,
                                point_compliance.measures)
    assert high.condition_number > low.condition_number
    assert high.residual < low.residual


def test_no_contact_gives_exactly_zero_force(point_compliance):
    radius_hat = RADIUS / hertz_point(LOAD, RADIUS, E_STAR)["a"]
    offset = point_compliance.coords ** 2 / (2.0 * radius_hat) + 0.05
    alm = solve_normal_alm(point_compliance.normal, offset, point_compliance.measures)
    penalty = solve_normal_penalty(point_compliance.normal, offset, 1.0e3,
                                   point_compliance.measures)
    assert float(alm.load) == 0.0
    assert float(penalty.load) == 0.0
    assert bool(torch.all(alm.gap > 0.0))


def test_zero_tangential_shift_gives_zero_traction(point_compliance):
    oracle, sol, radius_hat, delta, young_fe = _indent(point_compliance)
    path = solve_cattaneo_tangential(point_compliance.tangential, sol.nodal_force,
                                     point_compliance.measures,
                                     point_compliance.coords, 0.3,
                                     torch.zeros(2, dtype=DTYPE))
    assert float(path.tangential_load.max()) == 0.0


def test_cattaneo_partial_slip_structure(point_compliance):
    oracle, sol, radius_hat, delta, young_fe = _indent(point_compliance)
    shear = YOUNG / (2.0 * (1.0 + POISSON))
    a_fe = float(torch.tensor(1.0)) * oracle["a"]
    u_peak = cattaneo_displacement(0.9 * 0.3 * LOAD, LOAD, 0.3, a_fe, shear, POISSON)
    shifts = torch.linspace(0.0, 1.2 * u_peak / oracle["a"], 41, dtype=DTYPE)
    path = solve_cattaneo_tangential(point_compliance.tangential, sol.nodal_force,
                                     point_compliance.measures, point_compliance.coords,
                                     0.3, shifts)
    loads = path.tangential_load
    assert float(loads[0]) == 0.0
    assert bool(torch.all(loads[1:] >= loads[:-1] - 1e-18))       # monotone loading
    radii = path.stick_radius[1:]
    assert bool(torch.all(radii[1:] <= radii[:-1] + 1e-12))       # slip annulus grows
    contact = sol.nodal_force > 0.0
    mixed = any(bool((path.stick[i] & contact).any()) and
                bool((~path.stick[i] & contact).any()) for i in range(1, len(shifts)))
    assert mixed                       # a partial-slip sample exists
    # full sliding must saturate exactly at the Coulomb limit mu N
    # (loads are nodal forces in the dimensionless unit-modulus problem)
    assert float(path.tangential_load.max()) == pytest.approx(
        0.3 * float(sol.load), rel=1e-9)
    # Coulomb bound holds node by node
    assert bool(torch.all(path.traction[-1] <= 0.3 * sol.nodal_force + 1e-12))


def test_complementarity_residual_definition():
    p = torch.tensor([1.0, 0.0], dtype=DTYPE)
    g = torch.tensor([0.0, 2.0], dtype=DTYPE)
    assert complementarity_residual(p, g, 1.0, 1.0) == 0.0
    assert complementarity_residual(p, torch.tensor([1.0, 0.0], dtype=DTYPE),
                                    1.0, 1.0) == pytest.approx(1.0)


def test_line_contact_kernel_runs(line_compliance):
    """The plane-strain kernel must produce a symmetric compliance and reflect."""
    H = line_compliance.normal
    assert torch.allclose(H, H.T, atol=1e-12)
    assert bool(torch.all(torch.diagonal(H) > 0.0))
    oracle = hertz_line(LOAD / 1.0e-2, RADIUS, E_STAR)
    radius_hat = RADIUS / oracle["b"]
    young_fe = (1.0 - POISSON ** 2) * E_STAR          # unit-modulus plane strain
    # the plane-strain model is a symmetry half, so it carries half of P'
    load_hat = 0.5 * (LOAD / 1.0e-2) / (young_fe * oracle["b"])
    delta = 1.0 / radius_hat
    sol = None
    for _ in range(30):
        offset = line_compliance.coords ** 2 / (2.0 * radius_hat) - delta
        sol = solve_normal_alm(line_compliance.normal, offset, line_compliance.measures,
                               tolerance=1e-12)
        ratio = float(sol.load) / load_hat
        if abs(ratio - 1.0) < 1e-10:
            break
        delta *= 1.0 / ratio
    b_fe, p0_fe = recover_hertz_pressure(sol.pressure, line_compliance.coords)
    assert b_fe == pytest.approx(1.0, rel=0.05)
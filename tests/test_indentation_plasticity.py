"""Tests for the nonlinear ice-structure contact / large-deflection chain.

Covers :mod:`tensorfem.indentation_plasticity` (J2 return mapping, algorithmic
tangent, axisymmetric Q4 contact solver, whole-cycle energy balance, cyclic
shakedown, Johnson cavity model, Armstrong-Frederick ratcheting) and
:mod:`tensorfem.beam_large_deflection` (corotational beam vs Euler elastica).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem import indentation_plasticity as ind
from tensorfem.beam_large_deflection import (
    _elastica_by_quadrature,
    corotational_tangent_error,
    elastica_cantilever,
    solve_corotational_beam,
)
from tensorfem.hertz_axisymmetric import hertz_reference


# ---------------------------------------------------------------------------
# constitutive level
# ---------------------------------------------------------------------------
def _uniaxial(strain_axial, material):
    """Uniaxial stress state: lateral strain iterated so sigma_rr = sigma_tt = 0."""
    lateral = -material.poisson * strain_axial
    out = None
    for _ in range(80):
        strain = np.array([[lateral, lateral, strain_axial, 0.0]])
        out = ind.j2_return(strain, np.zeros((1, 4)), np.zeros(1), material)
        error = 0.5 * (out["stress"][0, 0] + out["stress"][0, 1])
        lateral -= error / material.young
    return out, out["stress"][0]


def test_uniaxial_elastoplastic_curve():
    """von Mises stress follows Y + H eps_p with the exact tangency of the flow rule."""
    material = ind.J2Material(young=1000.0, poisson=0.3, yield_stress=10.0,
                              hardening=50.0)
    for strain in (0.02, 0.05):
        out, stress = _uniaxial(strain, material)
        eqp = float(out["eqp"][0])
        assert eqp > 0.0
        assert abs(stress[0]) < 1e-4 and abs(stress[1]) < 1e-4
        assert out["equivalent_stress"][0] == pytest.approx(
            material.yield_stress + material.hardening * eqp, rel=1e-8)
    # well inside the elastic range nothing is plastic
    out, _ = _uniaxial(0.005, material)
    assert float(out["eqp"][0]) == 0.0


def test_plastic_work_increment_is_exact_for_linear_hardening():
    """dW_p = Y0 dlam + H/2 ((eps+dl)^2 - eps^2) integrated along the yield path."""
    material = ind.J2Material(young=1000.0, poisson=0.3, yield_stress=10.0,
                              hardening=50.0)
    state = {"plastic_strain": np.zeros((1, 4)), "eqp": np.zeros(1)}
    total = 0.0
    strain = np.zeros((1, 4))
    for _ in range(50):
        strain = strain + np.array([[0.0, 0.0, 1e-3, 0.0]])
        out = ind.j2_return(strain, state["plastic_strain"], state["eqp"],
                            material)
        total += float(out["plastic_work"][0])
        state = {"plastic_strain": out["plastic_strain"], "eqp": out["eqp"]}
    eqp = float(state["eqp"][0])
    expected = material.yield_stress * eqp + 0.5 * material.hardening * eqp ** 2
    assert total == pytest.approx(expected, rel=1e-10)


def test_algorithmic_tangent_matches_finite_difference():
    """The consistent tangent of the return map is audited against central differences."""
    material = ind.J2Material(young=1000.0, poisson=0.3, yield_stress=5.0,
                              hardening=80.0)
    strain = np.array([[0.004, -0.001, 0.012, 1e-3]])
    out = ind.j2_return(strain, np.zeros((1, 4)), np.zeros(1), material)
    assert float(out["eqp"][0]) > 0.0            # the state must be plastic
    tangent = out["tangent"][0]
    h = 1e-9
    numeric = np.zeros((4, 4))
    for j in range(4):
        step = np.zeros((1, 4))
        step[0, j] = h
        plus = ind.j2_return(strain + step, np.zeros((1, 4)), np.zeros(1),
                             material)["stress"][0]
        minus = ind.j2_return(strain - step, np.zeros((1, 4)), np.zeros(1),
                              material)["stress"][0]
        numeric[:, j] = (plus - minus) / (2 * h)
    scale = np.max(np.abs(numeric))
    assert np.max(np.abs(tangent - numeric)) / scale < 1e-6
    # the tangent is self-adjoint with respect to the weighted (tensor-shear)
    # double contraction, i.e. W C is symmetric (the assembly converts the
    # shear column to the engineering convention, which restores plain symmetry)
    weighted = ind._W[:, None] * tangent
    assert np.allclose(weighted, weighted.T)
    # proportional flow must not stiffen: d sigma ~ 0 for d eps along the flow
    flow = 1.5 * ind.von_mises(out["stress"][0])[1] / out["equivalent_stress"][0]
    assert np.linalg.norm(tangent @ flow) < 0.05 * scale * np.linalg.norm(flow)


def test_elastic_tangent_is_isotropic_and_matches_uniaxial():
    material = ind.J2Material(young=210000.0, poisson=0.3, yield_stress=250.0)
    ce = material.elastic_tangent()
    strain = np.array([1e-3, 0.0, 0.0, 0.0])
    stress = ce @ strain
    # uniaxial strain: sigma_rr = (lambda + 2 mu) eps
    lam = material.young * material.poisson / ((1 + material.poisson) * (1 - 2 * material.poisson))
    mu = material.shear
    assert stress[0] == pytest.approx((lam + 2 * mu) * 1e-3, rel=1e-12)
    # tensor-shear convention: tau = 2 G e_rz
    assert ce[3, 3] == pytest.approx(2 * mu, rel=1e-12)


# ---------------------------------------------------------------------------
# discretisation / solver level
# ---------------------------------------------------------------------------
def test_rigid_translation_produces_no_strain():
    """A uniform translation field must give zero strain and zero internal force."""
    discret = ind.axisymmetric_discretisation(nr=4, nz=4, width=4.0, depth=4.0,
                                              grading=2.0)
    material = ind.J2Material(young=1000.0, poisson=0.3, yield_stress=10.0)
    solver = ind.IndentationSolver(discret, material, radius=1.0)
    u = np.zeros(discret.n_dof)
    u[1::2] = 3e-3                      # uniform axial translation
    assert np.max(np.abs(solver.strains(u))) < 1e-15
    force, stiffness, _ = solver.assemble(u)
    assert np.linalg.norm(force) < 1e-10
    assert stiffness.shape == (discret.n_dof, discret.n_dof)


def test_elastic_axisymmetric_energy_consistency():
    """0.5 u^T K u equals 0.5 * integral of stress:strain over the volume."""
    discret = ind.axisymmetric_discretisation(nr=4, nz=4, width=4.0, depth=4.0,
                                              grading=2.0)
    material = ind.J2Material(young=1000.0, poisson=0.3, yield_stress=1e9)
    solver = ind.IndentationSolver(discret, material, radius=1.0)
    rng = np.random.default_rng(0)
    u = 1e-4 * rng.standard_normal(discret.n_dof)
    _, stiffness, _ = solver.assemble(u)
    quadratic = 0.5 * u @ (stiffness @ u)
    stored, _ = solver.volume_energies() if False else (None, None)
    strains = solver.strains(u)
    stress = strains @ material.elastic_tangent().T
    volume = float(np.sum(0.5 * np.einsum("ni,ni->n", ind._W * stress, strains)
                          * discret.weight.reshape(-1)))
    assert quadratic == pytest.approx(volume, rel=1e-10)


def test_elastic_hertz_contact_matches_closed_form():
    """Small-strain elastic limit: contact radius and force vs the Hertz oracle."""
    material = ind.J2Material(young=1000.0, poisson=0.3, yield_stress=1e9)
    result = ind.run_indentation_cycles(
        nr=16, nz=16, width=6.0, depth=6.0, grading=2.2, radius=1.0,
        peak_indentation=0.005, load_steps=2, unload_steps=1, cycles=1,
        material=material, penalty_factor=100.0)
    step = result.cycles[0].steps[-1]
    force, radius, peak = hertz_reference(0.005, 1.0, material.young,
                                          material.poisson)
    # Hertz-consistent radius (p^2 fit) on a coarse graded mesh
    # nr=16 ladder bottom; the convergence of this estimate is quantified in
    # benchmarks/indentation_cycle_study.py (3-mesh GCI)
    assert abs(step.fitted_radius / radius - 1) < 0.15
    assert abs(step.load / force - 1) < 0.20
    # the raw contact edge from two-point pressure extrapolation is a much
    # noisier estimator (graded mesh, penalty contact) -- only bounded here
    assert abs(step.contact_radius / radius - 1) < 0.30


def test_plastic_cycle_energy_balance_and_residual_depth():
    """Whole-cycle energy balance and a positive residual indentation depth."""
    material = ind.J2Material(young=1000.0, poisson=0.3, yield_stress=18.2)
    result = ind.run_indentation_cycles(
        nr=14, nz=14, width=6.0, depth=6.0, grading=2.2, radius=1.0,
        peak_indentation=0.02, load_steps=16, unload_steps=6, cycles=2,
        material=material, penalty_factor=100.0)
    cycle = result.cycles[0]
    assert cycle.plastic_work > 0.0
    assert cycle.residual_depth > 0.0
    assert abs(cycle.energy_closure) < 0.05
    # unloading must be elastic: plastic flow during the release is a bug
    assert cycle.release_plastic_flow < 1e-8
    assert cycle.converged
    # the residual depth must be smaller than the peak indentation
    assert cycle.residual_depth < 0.02


def test_repeated_cycles_shake_down():
    """Displacement-controlled cycles on an isotropically hardening solid shake down."""
    material = ind.J2Material(young=1000.0, poisson=0.3, yield_stress=18.2,
                              hardening=40.0)
    result = ind.run_indentation_cycles(
        nr=12, nz=12, width=6.0, depth=6.0, grading=2.2, radius=1.0,
        peak_indentation=0.01, load_steps=10, unload_steps=4, cycles=4,
        material=material, penalty_factor=100.0)
    depths = result.residual_depths
    assert depths[0] > 0.0
    # analytic (shakedown) prediction: no further accumulation after cycle 1
    for later in depths[1:]:
        assert abs(later - depths[0]) / depths[0] < 1e-3


# ---------------------------------------------------------------------------
# Johnson model
# ---------------------------------------------------------------------------
def test_johnson_model_limits():
    material = ind.J2Material(young=1000.0, poisson=0.3, yield_stress=18.2)
    model = ind.johnson_model(material, 1.0)
    a_y = model.elastic_limit_radius
    # elastic branch = Hertz mean pressure p_m = 4 E* a / (3 pi R)
    a_probe = 0.5 * a_y
    hertz_pm = 4.0 * material.effective_modulus * a_probe / (3.0 * math.pi)
    assert model.mean_pressure(a_probe) == pytest.approx(hertz_pm, rel=1e-12)
    # continuity at first yield: p_m = 1.07 Y
    assert model.mean_pressure(a_y) == pytest.approx(
        model.elastic_limit_pressure, rel=1e-12)
    # monotone increasing and capped by the fully plastic constraint factor
    radii = np.linspace(0.0, 1.0, 200)
    pressures = np.array([model.mean_pressure(a) for a in radii])
    assert np.all(np.diff(pressures) >= -1e-12)
    assert pressures.max() <= model.fully_plastic_factor * material.yield_stress + 1e-9
    # load and residual depth are positive and the residual depth < peak
    assert model.load(0.01) > 0.0
    assert 0.0 < model.residual_depth(0.01) < 0.01


# ---------------------------------------------------------------------------
# Armstrong-Frederick ratcheting
# ---------------------------------------------------------------------------
def test_armstrong_frederick_closed_form_matches_numerical_integration():
    params = dict(yield_stress=200.0, hardening_modulus=50000.0, gamma=200.0,
                  stress_max=400.0, stress_min=-50.0)
    closed = ind.armstrong_frederick_ratchet_rate(**params)
    numeric = ind.integrate_armstrong_frederick_cycles(cycles=12, steps=1200,
                                                       **params)
    assert closed > 0.0
    assert abs(numeric[-1] / closed - 1) < 0.02
    # exactly 2Y of stress range -> elastic shakedown, zero ratcheting
    assert ind.armstrong_frederick_ratchet_rate(**{**params, "stress_max": 350.0}) == 0.0
    # a stress range beyond the saturation back-stress -> unbounded ratcheting
    assert ind.armstrong_frederick_ratchet_rate(
        **{**params, "stress_max": 700.0}) == float("inf")


# ---------------------------------------------------------------------------
# large-deflection beam
# ---------------------------------------------------------------------------
def test_elastica_small_load_reduces_to_euler_bernoulli():
    alpha = 0.02
    exact = elastica_cantilever(load_parameter=alpha)
    assert exact.tip_y == pytest.approx(alpha / 3.0, rel=2e-3)
    assert abs(exact.tip_x - 1.0) < 1e-3
    assert exact.tip_angle == pytest.approx(alpha / 2.0, rel=5e-2)


def test_elastica_matches_independent_quadrature():
    for alpha in (1.0, 3.0, 10.0):
        exact = elastica_cantilever(load_parameter=alpha)
        quad = _elastica_by_quadrature(load_parameter=alpha)
        # the quadrature oracle is first-order accurate at the endpoint
        # square-root singularity, so the agreement band is 1 %
        assert quad.tip_y == pytest.approx(exact.tip_y, rel=2e-2)
        assert quad.tip_x == pytest.approx(exact.tip_x, rel=2e-2)


def test_corotational_beam_matches_elastica():
    for alpha, n_el in ((1.0, 12), (10.0, 20)):
        exact = elastica_cantilever(load_parameter=alpha)
        fe = solve_corotational_beam(n_elements=n_el, tip_force=alpha,
                                     load_steps=8, tolerance=1e-9)
        assert fe.converged
        assert abs(fe.tip_y / exact.tip_y - 1) < 0.02
        assert abs(fe.tip_x / exact.tip_x - 1) < 0.02


def test_corotational_tangent_finite_difference_audit():
    audit = corotational_tangent_error(n_elements=3)
    assert audit["max_relative_error"] < 1e-7


def test_beam_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        elastica_cantilever(load_parameter=-1.0)
    with pytest.raises(ValueError):
        solve_corotational_beam(n_elements=0)
    with pytest.raises(ValueError):
        ind.J2Material(young=-1.0, poisson=0.3, yield_stress=1.0)
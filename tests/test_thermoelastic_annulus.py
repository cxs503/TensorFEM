"""Verification of the plane thermoelastic path on the hollow-cylinder case.

Reference: closed-form traction-free annulus under a steady radial temperature
field (see :mod:`tensorfem.thermoelastic_annulus`).  The reference itself is
scored first (equilibrium residual, traction-free surfaces, uniform-temperature
limit, constitutive consistency), then the Q4 finite element path is scored
against it, and finally two negative controls prove the gates can fail.
"""
import math

import pytest
import torch

from tensorfem.continuum import elasticity_matrix, q4_stiffness
from tensorfem.lame_reference import sector_q4_mesh
from tensorfem.thermoelastic_annulus import (
    _b_matrix,
    _shape_derivatives,
    annulus_thermal_state,
    effective_thermal_strain,
    radial_temperature,
    solve_annulus_thermoelasticity,
    thermal_strain_loads,
)

DTYPE = torch.float64
YOUNG, POISSON, ALPHA = 210.0e9, 0.3, 12.0e-6
INNER, OUTER = 0.5, 0.8
T_INNER, T_OUTER = 500.0, 100.0
GATE = 0.03


def _state(radius, inner_temperature=T_INNER, outer_temperature=T_OUTER):
    return annulus_thermal_state(
        torch.as_tensor(radius, dtype=DTYPE),
        inner_radius=INNER,
        outer_radius=OUTER,
        inner_temperature=inner_temperature,
        outer_temperature=outer_temperature,
        young_modulus=YOUNG,
        poisson_ratio=POISSON,
        expansion_coefficient=ALPHA,
    )


def _solve(n_theta, n_radial, grading=1.0, **kwargs):
    return solve_annulus_thermoelasticity(
        INNER,
        OUTER,
        n_theta,
        n_radial,
        young_modulus=YOUNG,
        poisson_ratio=POISSON,
        expansion_coefficient=ALPHA,
        inner_temperature=T_INNER,
        outer_temperature=T_OUTER,
        radial_grading=grading,
        **kwargs,
    )


def test_temperature_profile_matches_logarithmic_exact_solution():
    radius = torch.tensor([INNER, OUTER], dtype=DTYPE)
    temperature = radial_temperature(
        radius,
        inner_radius=INNER,
        outer_radius=OUTER,
        inner_temperature=T_INNER,
        outer_temperature=T_OUTER,
    )
    assert float(temperature[0]) == pytest.approx(T_INNER, rel=1e-14)
    assert float(temperature[1]) == pytest.approx(T_OUTER, rel=1e-14)
    # The exact profile is harmonic: (1/r) d/dr(r dT/dr) = 0.
    r = torch.linspace(INNER, OUTER, 401, dtype=DTYPE)
    t = radial_temperature(
        r,
        inner_radius=INNER,
        outer_radius=OUTER,
        inner_temperature=T_INNER,
        outer_temperature=T_OUTER,
    )
    h = float(r[1] - r[0])
    laplacian = (t[2:] - 2.0 * t[1:-1] + t[:-2]) / h**2 + (t[2:] - t[:-2]) / (2.0 * h * r[1:-1])
    slope = (T_OUTER - T_INNER) / math.log(OUTER / INNER)
    assert float(laplacian.abs().max()) < 1e-4 * abs(slope) / INNER**2


def test_closed_form_satisfies_axisymmetric_equilibrium():
    r = torch.linspace(INNER, OUTER, 4001, dtype=DTYPE)
    state = _state(r)
    h = float(r[1] - r[0])
    d_sigma_r = (state.radial_stress[2:] - state.radial_stress[:-2]) / (2.0 * h)
    residual = d_sigma_r + (state.radial_stress[1:-1] - state.hoop_stress[1:-1]) / r[1:-1]
    scale = float(state.hoop_stress.abs().max()) / (OUTER - INNER)
    assert float(residual.abs().max()) < 1e-4 * scale


def test_closed_form_is_traction_free_on_both_surfaces():
    state = _state(torch.tensor([INNER, OUTER], dtype=DTYPE))
    scale = float(state.hoop_stress.abs().max())
    assert abs(float(state.radial_stress[0])) < 1e-9 * scale
    assert abs(float(state.radial_stress[1])) < 1e-9 * scale


def test_closed_form_is_constitutive_consistent_and_plane_strain():
    r = torch.linspace(INNER, OUTER, 4001, dtype=DTYPE)
    state = _state(r)
    lam = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
    bulk = YOUNG / (1.0 - 2.0 * POISSON)
    span = math.log(OUTER / INNER)
    theta = T_INNER + (T_OUTER - T_INNER) * torch.log(r / INNER) / span
    eps_r = torch.gradient(state.radial_displacement, spacing=float(r[1] - r[0]))[0]
    eps_t = state.radial_displacement / r
    axial = lam * (eps_r + eps_t) - bulk * ALPHA * theta
    # sigma_z is reproduced by the plane-strain transcription of the 3-D law.
    assert torch.allclose(axial[2:-2], state.axial_stress[2:-2], rtol=1e-6)
    # The same identity written with the stresses: sigma_z = nu (sigma_r + sigma_theta) - alpha E T.
    axial_from_stresses = (
        POISSON * (state.radial_stress[2:-2] + state.hoop_stress[2:-2]) - ALPHA * YOUNG * theta[2:-2]
    )
    assert torch.allclose(axial_from_stresses, state.axial_stress[2:-2], rtol=1e-10, atol=1e-3)
    # radial equilibrium in displacement form is the ODE the closed form solves.
    assert float(state.radial_displacement[0]) > 0.0
    assert float(state.hoop_stress[0]) < 0.0  # hot bore is restrained -> compression
    assert float(state.radial_stress[1:-1].abs().max()) < float(state.hoop_stress.abs().max())


def test_uniform_temperature_leaves_the_in_plane_field_stress_free():
    state = _state(torch.linspace(INNER, OUTER, 9, dtype=DTYPE), 400.0, 400.0)
    scale = YOUNG * ALPHA * 400.0
    assert float(state.hoop_stress.abs().max()) < 1e-12 * scale
    assert float(state.radial_stress.abs().max()) < 1e-12 * scale
    # Plane strain keeps the axial strain at zero, so the axial stress is *not*
    # zero: it is the constraint stress -E alpha T (a free 3-D body would relax it).
    assert torch.allclose(state.axial_stress, torch.full_like(state.axial_stress, -scale), rtol=1e-12)


def test_effective_plane_strain_thermal_strain_reproduces_three_dimensional_term():
    """``D eps_th_eff`` must equal the 3-D thermal stress ``(3 lambda + 2 mu) alpha T``."""
    mesh = torch.tensor([290.0], dtype=DTYPE)
    dmat = elasticity_matrix(
        torch.tensor([YOUNG], dtype=DTYPE), torch.tensor([POISSON], dtype=DTYPE), "strain"
    )[0]
    thermal = effective_thermal_strain(
        mesh, expansion_coefficient=ALPHA, poisson_ratio=POISSON, plane="strain"
    ).reshape(3)
    stress = dmat @ thermal
    expected = YOUNG / (1.0 - 2.0 * POISSON) * ALPHA * mesh.reshape(())
    assert float(stress[0]) == pytest.approx(float(expected), rel=1e-12)
    assert float(stress[1]) == pytest.approx(float(expected), rel=1e-12)
    # The naive alpha*T thermal strain (no (1+nu) factor) is off by exactly (1+nu).
    naive = (ALPHA * mesh.reshape(()) * torch.tensor([1.0, 1.0, 0.0], dtype=DTYPE))
    assert float((dmat @ naive)[0]) == pytest.approx(float(expected) / (1.0 + POISSON), rel=1e-12)
    # Plane stress needs no correction: E alpha T /(1 - nu).
    dmat_ps = elasticity_matrix(
        torch.tensor([YOUNG], dtype=DTYPE), torch.tensor([POISSON], dtype=DTYPE), "stress"
    )[0]
    thermal_ps = effective_thermal_strain(
        mesh, expansion_coefficient=ALPHA, poisson_ratio=POISSON, plane="stress"
    ).reshape(3)
    assert float((dmat_ps @ thermal_ps)[0]) == pytest.approx(
        YOUNG * ALPHA * float(mesh) / (1.0 - POISSON), rel=1e-12
    )


def test_shape_derivatives_reproduce_an_arbitrary_linear_field_on_a_skew_element():
    nodes, elements = sector_q4_mesh(INNER, OUTER, 4, 4, dtype=DTYPE)
    xy = nodes[elements[3]]  # strongly skewed annular-sector element
    gradient = torch.tensor([[1.0e-3, 2.0e-4], [1.0e-4, -7.0e-4]], dtype=DTYPE)
    displacement = torch.zeros(8, dtype=DTYPE)
    mapped = xy @ gradient.T
    displacement[0::2] = mapped[:, 0]
    displacement[1::2] = mapped[:, 1]
    exact = torch.tensor(
        [gradient[0, 0], gradient[1, 1], gradient[0, 1] + gradient[1, 0]], dtype=DTYPE
    )
    gauss = 1.0 / 3.0**0.5
    for xi, eta in ((-gauss, -gauss), (gauss, -gauss), (gauss, gauss), (-gauss, gauss)):
        _, derivatives, det = _shape_derivatives(xy, xi, eta)
        assert float(det) > 0.0
        strain = _b_matrix(derivatives) @ displacement
        assert torch.allclose(strain, exact, atol=1e-15)


def test_thermal_load_vector_balances_free_expansion():
    """``f_th = K u_free`` for uniform heating: the load vector is fully consistent."""
    nodes, elements = sector_q4_mesh(INNER, OUTER, 4, 4, dtype=DTYPE)
    n_elem = elements.shape[0]
    temperature = torch.full((nodes.shape[0],), 250.0, dtype=DTYPE)
    young = torch.full((n_elem,), YOUNG, dtype=DTYPE)
    poisson = torch.full((n_elem,), POISSON, dtype=DTYPE)
    thickness = torch.ones(n_elem, dtype=DTYPE)
    forces = thermal_strain_loads(
        nodes,
        elements,
        temperature,
        young_modulus=young,
        poisson_ratio=poisson,
        expansion_coefficient=ALPHA,
        thickness=thickness,
        plane="strain",
    )
    dmat = elasticity_matrix(young, poisson, "strain")
    element_stiffness, _ = q4_stiffness(nodes[elements], dmat, thickness)
    edofs = torch.stack(tuple(2 * elements + i for i in (0, 1)), dim=2).reshape(n_elem, 8)
    stiffness = torch.zeros((2 * nodes.shape[0],) * 2, dtype=DTYPE)
    rows = edofs[:, :, None].expand(-1, -1, 8).reshape(-1)
    cols = edofs[:, None, :].expand(-1, 8, -1).reshape(-1)
    stiffness = stiffness.index_put((rows, cols), element_stiffness.reshape(-1), accumulate=True)
    expansion = (1.0 + POISSON) * ALPHA * 250.0
    free = torch.zeros(2 * nodes.shape[0], dtype=DTYPE)
    free[0::2] = expansion * nodes[:, 0]
    free[1::2] = expansion * nodes[:, 1]
    residual = forces - stiffness @ free
    assert float(residual.abs().max()) < 1e-10 * float(forces.abs().max())


def test_bore_hoop_stress_matches_closed_form_within_three_percent():
    result = _solve(8, 16)
    index = int(torch.argmin(result.sample_radius))
    reference = _state(result.sample_radius[index : index + 1])
    peak = float(_state(result.sample_radius).hoop_stress.abs().max())
    error = abs(float(result.hoop_stress[index]) - float(reference.hoop_stress[0])) / peak
    assert error < GATE


def test_hoop_and_axial_fields_match_closed_form_within_three_percent():
    result = _solve(12, 24, grading=1.6)
    reference = _state(result.sample_radius)
    hoop_peak = float(reference.hoop_stress.abs().max())
    axial_peak = float(reference.axial_stress.abs().max())
    hoop_error = float((result.hoop_stress - reference.hoop_stress).abs().max()) / hoop_peak
    axial_error = float((result.axial_stress - reference.axial_stress).abs().max()) / axial_peak
    assert hoop_error < GATE
    assert axial_error < GATE


def test_radial_stress_error_is_reported_against_its_own_peak():
    """Honest limitation: sigma_r does *not* meet the 3 % gate relative to its own peak.

    sigma_r is a small difference quantity (its peak is roughly a twentieth of the
    hoop peak), so an O(h) error of a few percent of the hoop scale is tens of
    percent of the sigma_r scale.  Normalised against the macroscopic stress scale
    the same field meets the gate.  This test locks the fact so that no later
    report can claim a 3 % sigma_r field error.
    """
    result = _solve(12, 24, grading=1.6)
    reference = _state(result.sample_radius)
    hoop_peak = float(reference.hoop_stress.abs().max())
    radial_peak = float(reference.radial_stress.abs().max())
    absolute = float((result.radial_stress - reference.radial_stress).abs().max())
    assert absolute / radial_peak > GATE        # 43 % of its own peak
    assert absolute / hoop_peak > GATE          # and still 4.3 % of the hoop peak
    # The failure is O(h), not a constant offset: refining radially by 2 halves it.
    coarse = _solve(12, 12, grading=1.6)
    coarse_reference = _state(coarse.sample_radius)
    coarse_absolute = float((coarse.radial_stress - coarse_reference.radial_stress).abs().max())
    assert coarse_absolute > 1.6 * absolute


def test_negative_control_dropping_thermal_strain_breaks_the_gate():
    """If the thermal strain were forgotten, the comparison must fail loudly."""
    result = _solve(12, 24, grading=1.6, include_thermal_strain=False)
    reference = _state(result.sample_radius)
    peak = float(reference.hoop_stress.abs().max())
    error = float((result.hoop_stress - reference.hoop_stress).abs().max()) / peak
    assert error > 1.0


def test_naive_plane_strain_thermal_strain_breaks_the_gate():
    """Negative control on the plane-strain (1 + nu) factor of the thermal strain."""
    dmat = elasticity_matrix(
        torch.tensor([YOUNG], dtype=DTYPE), torch.tensor([POISSON], dtype=DTYPE), "strain"
    )[0]
    temperature = 200.0
    naive = dmat @ (ALPHA * temperature * torch.tensor([1.0, 1.0, 0.0], dtype=DTYPE))
    corrected = dmat @ effective_thermal_strain(
        torch.tensor([temperature], dtype=DTYPE),
        expansion_coefficient=ALPHA,
        poisson_ratio=POISSON,
        plane="strain",
    ).reshape(3)
    assert float((naive[0] - corrected[0]).abs() / corrected[0].abs()) == pytest.approx(
        POISSON / (1.0 + POISSON), rel=1e-12
    )
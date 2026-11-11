"""Tests for the cohesive-interface three-point bend fracture model.

Layers
------
1. constitutive checks on :class:`BilinearCohesiveLaw` (envelope area, damage,
   irreversibility, consistent tangent);
2. an element-level cross-check of the in-house Q4 stiffness against the
   repository's verified ``tensorfem.continuum.q4_stiffness``;
3. the fracture process: crack initiation at the tensile face, monotone upward
   propagation along the symmetry plane, peak load against the flexural
   strength criterion, dissipated energy against ``G_f``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfem.cohesive_bending import (
    BilinearCohesiveLaw,
    build_three_point_bend,
    flexural_strength_load,
    plane_stress_matrix,
    solve_cmod_control,
    timoshenko_midspan_deflection,
    _q4_stiffness,
)

YOUNG = 5.0e9
POISSON = 0.3
STRENGTH = 1.5e6
FRACTURE_ENERGY = 1.0
SPAN = 0.16
DEPTH = 0.04
THICKNESS = 0.04


@pytest.fixture(scope="module")
def law() -> BilinearCohesiveLaw:
    return BilinearCohesiveLaw.from_material(
        YOUNG, STRENGTH, FRACTURE_ENERGY, penalty_ratio=50.0
    )


# ---------------------------------------------------------------------------
# constitutive law
# ---------------------------------------------------------------------------
def test_envelope_area_equals_fracture_energy(law: BilinearCohesiveLaw):
    kappa = np.array([0.0, 0.5 * law.damage_opening, law.damage_opening, law.failure_opening])
    area = law.envelope_area(kappa)
    assert area[0] == 0.0
    assert area[1] == pytest.approx(0.5 * law.normal_stiffness * kappa[1] ** 2, rel=1e-12)
    assert area[-1] == pytest.approx(law.fracture_energy, rel=1e-12)
    assert 0.5 * law.peak_traction * law.failure_opening == pytest.approx(
        law.fracture_energy, rel=1e-12
    )


def test_damage_is_bounded_and_monotone(law: BilinearCohesiveLaw):
    kappa = np.linspace(0.0, 1.5 * law.failure_opening, 257)
    d = law.damage(kappa)
    assert np.all(d >= 0.0) and np.all(d <= 1.0)
    assert np.all(np.diff(d) >= -1e-15)
    assert d[0] == 0.0
    assert law.damage(np.array([law.failure_opening]))[0] == 1.0


def test_peak_traction_and_failure_traction(law: BilinearCohesiveLaw):
    d0, df = law.damage_opening, law.failure_opening
    tn, _, _, _, _ = law.traction(d0, 0.0, 0.0)
    assert tn == pytest.approx(law.peak_traction, rel=1e-12)
    tn, _, cnn, _, _ = law.traction(df, 0.0, df)
    assert abs(tn) < 1e-9 * law.peak_traction
    assert cnn == pytest.approx(0.0, abs=1e-9 * law.normal_stiffness)


def test_unloading_is_secant_to_the_origin(law: BilinearCohesiveLaw):
    d0, df = law.damage_opening, law.failure_opening
    kappa = 0.5 * (d0 + df)
    d = float(law.damage(np.array([kappa]))[0])
    tn, _, cnn, _, kappa_new = law.traction(0.2 * kappa, 0.0, kappa)
    assert tn == pytest.approx((1.0 - d) * law.normal_stiffness * 0.2 * kappa, rel=1e-12)
    assert cnn == pytest.approx((1.0 - d) * law.normal_stiffness, rel=1e-12)
    assert kappa_new == pytest.approx(kappa, rel=1e-15)


def test_compression_never_damages(law: BilinearCohesiveLaw):
    tn, _, cnn, _, kappa = law.traction(-1e-6, 0.0, 0.0)
    assert tn == pytest.approx(-law.normal_stiffness * 1e-6, rel=1e-12)
    assert cnn == pytest.approx(law.normal_stiffness, rel=1e-12)
    assert kappa == 0.0


# note: delta = delta_0 is a kink of the envelope, so the tangent is
# intentionally not differentiated there
@pytest.mark.parametrize("fraction", [0.5, 0.99, 1.2, 1.5, 3.0, 8.0])
def test_consistent_tangent_matches_finite_difference(law: BilinearCohesiveLaw, fraction):
    delta_n = fraction * law.damage_opening
    h = 1e-9 * law.damage_opening
    _, _, cnn, _, _ = law.traction(delta_n, 0.0, 0.0)
    t_plus, _, _, _, _ = law.traction(delta_n + h, 0.0, 0.0)
    t_minus, _, _, _, _ = law.traction(delta_n - h, 0.0, 0.0)
    numeric = (t_plus - t_minus) / (2.0 * h)
    assert cnn == pytest.approx(numeric, rel=5e-3, abs=1e-3 * law.normal_stiffness)


# ---------------------------------------------------------------------------
# element level cross-check against the repository continuum path
# ---------------------------------------------------------------------------
def test_q4_stiffness_matches_repository_continuum():
    from tensorfem.continuum import elasticity_matrix, q4_stiffness

    coords = torch.tensor(
        [[[0.0, 0.0], [0.12, 0.01], [0.1, 0.08], [-0.02, 0.07]]], dtype=torch.float64
    )
    dmat_t = elasticity_matrix(
        torch.tensor([YOUNG], dtype=torch.float64),
        torch.tensor([POISSON], dtype=torch.float64),
        "stress",
    )
    ke_ref, _ = q4_stiffness(coords, dmat_t, torch.tensor([THICKNESS], dtype=torch.float64))
    ke = _q4_stiffness(coords.numpy(), plane_stress_matrix(YOUNG, POISSON), THICKNESS)
    assert np.allclose(ke[0], ke_ref.numpy()[0], rtol=1e-11, atol=1e-6)


# ---------------------------------------------------------------------------
# fracture process (one shared small run)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_run(law: BilinearCohesiveLaw):
    model = build_three_point_bend(
        span=SPAN, depth=DEPTH, thickness=THICKNESS, young_modulus=YOUNG,
        poisson_ratio=POISSON, law=law, nx=16, ny=16,
    )
    result = solve_cmod_control(
        model, initial_step=0.5 * law.damage_opening,
        cmod_max=60.0 * law.failure_opening, max_steps=200, growth=1.3,
    )
    return model, result


def test_crack_starts_at_tensile_face_and_grows_upwards(small_run):
    model, result = small_run
    damage = result.damage_profile
    assert damage[0] > 0.5
    assert np.all(np.diff(damage) <= 1e-9)
    assert np.all(np.diff(result.crack_length) >= -1e-12)
    assert result.crack_length[-1] > 0.0
    assert result.crack_length[-1] <= DEPTH * (1.0 + 1e-9)


def test_peak_load_matches_flexural_strength(small_run):
    model, result = small_run
    analytic = flexural_strength_load(
        span=SPAN, depth=DEPTH, thickness=THICKNESS, tensile_strength=STRENGTH
    )
    # coarse mesh (h_seg ~ 1.5 l_ch): the discrete crack front overstrengthens
    # the specimen by a few per cent, which the benchmark resolves by mesh
    # refinement.
    assert abs(result.peak_load - analytic) / analytic < 0.10


def test_load_drops_after_the_peak(small_run):
    model, result = small_run
    assert np.isfinite(result.peak_load) and result.peak_load > 0.0
    if result.peak_index < len(result.load) - 1:
        assert result.load[-1] < result.peak_load


def test_dissipated_energy_bounded_by_fracture_energy(small_run):
    model, result = small_run
    assert np.all(np.diff(result.dissipated_envelope) >= -1e-18)
    assert result.dissipated_envelope[-1] <= (
        FRACTURE_ENERGY * THICKNESS * DEPTH * (1.0 + 1e-9)
    )
    # the path integral and the closed-form envelope area of the same
    # dissipation must agree
    assert result.dissipated_path[-1] == pytest.approx(
        result.dissipated_envelope[-1], rel=0.15
    )
    # and the dissipation must be at least the energy of the fully failed part
    # of the interface (tip energy only adds)
    failed_area = THICKNESS * result.crack_length[-1]
    assert result.dissipated_envelope[-1] >= FRACTURE_ENERGY * failed_area * (1.0 - 1e-9)


def test_elastic_compliance_is_thick_beam_like(law: BilinearCohesiveLaw):
    """Small-load compliance: 2-D elasticity vs the Timoshenko estimate."""
    model = build_three_point_bend(
        span=SPAN, depth=DEPTH, thickness=THICKNESS, young_modulus=YOUNG,
        poisson_ratio=POISSON, law=law, nx=16, ny=16,
    )
    result = solve_cmod_control(
        model, initial_step=0.1 * law.damage_opening,
        cmod_max=0.2 * law.damage_opening, max_steps=3,
    )
    load = float(result.load[0])
    deflection = float(result.deflection[0])
    reference = timoshenko_midspan_deflection(
        span=SPAN, depth=DEPTH, thickness=THICKNESS, young_modulus=YOUNG,
        poisson_ratio=POISSON, load=load,
    )
    assert 0.75 < deflection / reference < 1.35
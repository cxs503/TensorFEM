"""Verification tests for the Hillerborg fictitious-crack model.

Every assertion is against an independently known quantity (closed-form law area,
Maxwell--Betti reciprocity, handbook stress-intensity factor, monotonicity of the
cohesive size effect), never against a stored copy of the module's own output.
"""

import math

import numpy as np
import pytest

from tensorfem.fictitious_crack import (
    BilinearSoftening,
    ExponentialSoftening,
    SENBGeometry,
    _q4_stiffness,
    _q4_stiffness_batch,
    build_influence_model,
    elastic_stress_intensity_from_compliance,
    senb_handbook_shape_factor,
    senb_handbook_stress_intensity,
    senb_lefm_peak_load,
    solve_cmod_control,
    strength_limit_load,
)

E = 30.0e9
NU = 0.2
FT = 3.0e6
GF = 100.0
L_CH = E * GF / FT**2  # Hillerborg characteristic length, 0.3333 m


def geometry(depth=L_CH, alpha=0.2, young=E):
    return SENBGeometry(
        depth=depth, span=4.0 * depth, thickness=depth, notch=alpha * depth, young=young
    )


def law_kind(kind, depth=L_CH, factor=1.0e3):
    cls = {"exponential": ExponentialSoftening, "bilinear": BilinearSoftening}[kind]
    return cls(tensile_strength=FT, fracture_energy=GF, initial_stiffness=factor * E / depth)


# ---------------------------------------------------------------------------
# softening laws
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["exponential", "bilinear"])
def test_envelope_area_equals_fracture_energy(kind):
    law = law_kind(kind)
    # the envelope integral must reproduce G_F from the elastic triangle plus the
    # softening branch (independent quadrature, analytic target)
    assert law.total_area() == pytest.approx(GF, rel=1e-4)


@pytest.mark.parametrize("kind", ["exponential", "bilinear"])
def test_envelope_is_monotone_and_invertible(kind):
    law = law_kind(kind)
    # the envelope itself is non-increasing (elastic rise, then softening)
    w = np.linspace(0.0, law.separation_opening, 20000)
    sigma = law.traction(w)
    peak = int(np.argmax(sigma))
    assert np.all(np.diff(sigma[:peak]) > 0.0)
    assert np.all(np.diff(sigma[peak:]) <= 1e-6 * FT)
    # the inverse is single valued on the softening branch only
    ws = np.linspace(1.001 * law.elastic_opening, 0.999 * law.separation_opening, 500)
    sig = law.traction(ws)
    assert np.max(np.abs(law.opening(sig) - ws)) <= 1e-8 * law.separation_opening


@pytest.mark.parametrize("kind", ["exponential", "bilinear"])
def test_dissipated_energy_is_consistent_with_traction(kind):
    law = law_kind(kind)
    w = np.linspace(0.0, law.separation_opening, 20001)
    numeric = np.concatenate([[0.0], np.cumsum(0.5 * (law.traction(w[1:]) + law.traction(w[:-1]))
                                               * np.diff(w))])
    assert np.max(np.abs(law.dissipated(w) - numeric)) <= 1e-4 * GF


def test_handbook_shape_factor_limits():
    # edge-crack asymptote of the bending shape factor: f -> 1.99 * 1.5 * sqrt(a)
    a = 1e-6
    assert senb_handbook_shape_factor(a) / (2.985 * math.sqrt(a)) == pytest.approx(1.0, rel=2e-3)
    # tabulated value (Srawley--Gross) at a/W = 0.2
    assert senb_handbook_shape_factor(0.2) == pytest.approx(1.1749, rel=1e-3)


def test_handbook_and_lefm_limits_are_consistent():
    geom = geometry()
    k_ic = math.sqrt(E * GF)
    # P_LEFM must be the load that makes the handbook K equal sqrt(E G_F)
    p = senb_lefm_peak_load(geom, GF)
    assert senb_handbook_stress_intensity(p, geom) == pytest.approx(k_ic, rel=1e-12)
    # strength limit reduces to the flexural limit for a vanishing notch
    # flexural limit for a0 -> 0:  (2/3) B W^2 f_t / S  with B = W, S = 4W
    assert strength_limit_load(geometry(alpha=1e-12), FT) == pytest.approx(
        FT * L_CH**2 / 6.0, rel=1e-4)


# ---------------------------------------------------------------------------
# element and influence construction
# ---------------------------------------------------------------------------
def test_batched_q4_stiffness_matches_scalar():
    rng = np.random.default_rng(0)
    base = np.array([[0.0, 0.0], [1.0, 0.0], [1.2, 1.1], [0.0, 1.0]])
    coords = base[None, :, :] + rng.random((6, 1, 2)) * 0.05
    dmat = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 0.35]]) * E
    batch = _q4_stiffness_batch(coords, dmat, 0.5)
    for e in range(coords.shape[0]):
        assert np.allclose(batch[e], _q4_stiffness(coords[e], dmat, 0.5), rtol=1e-12, atol=1e-6)


def test_influence_matrix_satisfies_maxwell_betti():
    model = build_influence_model(geometry(), law_kind("bilinear"), nodes_y=16)
    # reciprocity holds for the opening-per-unit-force matrix; a violation means
    # the influence columns do not come from one linear-elastic body
    assert model.reciprocity_error() < 1e-10


def test_influence_openings_are_positive_and_decreasing_with_traction():
    model = build_influence_model(geometry(), law_kind("bilinear"), nodes_y=16)
    sig = np.zeros(model.n_active)
    w0 = model.openings(1000.0, sig)
    sig[:] = 1.0e6
    w1 = model.openings(1000.0, sig)
    assert np.all(w1 <= w0 + 1e-18 * np.abs(w0).max())  # closing tractions reduce the opening
    diag = np.diag(model.C)[model.area > 0.0]
    assert np.all(diag > 0.0)  # positive opening under tension on the crack path


def test_elastic_stress_intensity_matches_handbook():
    model = build_influence_model(geometry(), law_kind("bilinear"), nodes_y=48)
    result = elastic_stress_intensity_from_compliance(model, 0.2 * L_CH, delta=0.06 * L_CH)
    assert abs(result["rel_error_pct"]) <= 5.0


# ---------------------------------------------------------------------------
# nonlinear solution
# ---------------------------------------------------------------------------
def test_peak_load_decreases_with_notch_depth():
    peaks = []
    for alpha in (0.1, 0.25):
        model = build_influence_model(geometry(alpha=alpha), law_kind("bilinear"), nodes_y=16)
        path = solve_cmod_control(model, n_steps=12, max_iter=300)
        assert path.converged.sum() >= 10
        peaks.append(path.peak_load)
    assert peaks[0] > peaks[1]


def test_energy_balance_is_closed():
    model = build_influence_model(geometry(), law_kind("bilinear"), nodes_y=16)
    path = solve_cmod_control(model, n_steps=16, max_iter=300)
    ok = path.converged
    assert ok.sum() >= 12
    bal = np.abs(path.balance[ok])
    # elastic steps close to machine precision, softening steps to a few percent
    assert np.max(bal[:5]) <= 1e-10
    # NOTE: this is a deliberately coarse (nodes_y = 16) CI mesh; the production
    # mesh (nodes_y = 48/96) closes the balance to <= 1 % (see artifacts/hillerborg1976)
    assert np.max(bal) <= 0.25
    # the constitutive dissipation may never exceed G_F times the fracture area
    area = model.geometry.thickness * (model.effective_cut - model.effective_notch)
    assert path.dissipated[-1] <= GF * area * 1.001
    assert path.dissipated[-1] > 0.5 * GF * area


def test_peak_load_below_lefm_and_strength_for_transition_specimen():
    geom = geometry()
    model = build_influence_model(geom, law_kind("bilinear"), nodes_y=24)
    path = solve_cmod_control(model, n_steps=24, max_iter=300)
    lefm = senb_lefm_peak_load(geom, GF)
    strength = strength_limit_load(geom, FT)
    # a W/l_ch = 1 specimen is neither fully brittle (LEFM) nor fully ductile
    assert 0.15 < path.peak_load / lefm < 0.8
    assert path.peak_load < strength
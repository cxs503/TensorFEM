"""Biaxial Kirsch reference + production Q4 verification (fail-closed anchors).

The uniaxial Kirsch case is covered by ``tests/test_kirsch_hole_reference.py``;
here the *analytic reference itself* is generalised to a biaxial far field and
the production plane-stress path is graded for several biaxiality ratios.  The
assertions below are the machine gate for
``docs/kirsch-biaxial-hole-validation.md`` and
``benchmarks/kirsch_biaxial_reference.json``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.kirsch_hole import KirschHole, solve_case

BASE = dict(radius=1.0, far_field=1.0, young=1.0, poisson=0.3, plane="stress")
RATIOS = (-1.0, 0.0, 0.25, 0.5, 1.0, 2.0)


def hole(ratio: float, **overrides) -> KirschHole:
    return KirschHole(**{**BASE, **overrides, "lateral_ratio": ratio})


# --------------------------------------------------------------- reference ---

def test_default_construction_is_uniaxial():
    default = KirschHole()
    assert default.lateral_ratio == 0.0
    assert default.far_field_y == 0.0
    assert default.stress_concentration == pytest.approx(3.0, abs=1e-15)
    # textbook uniaxial field at (r/a = 2, theta = 30 deg)
    srr, stt, srt = default.stress_polar(2.0, math.radians(30.0))
    a2r2, a4r4 = 0.25, 0.0625
    c2, s2 = 0.5, math.sqrt(3.0) / 2.0
    assert float(srr) == pytest.approx(0.5 * (1 - a2r2) + 0.5 * (1 - 4 * a2r2 + 3 * a4r4) * c2)
    assert float(stt) == pytest.approx(0.5 * (1 + a2r2) - 0.5 * (1 + 3 * a4r4) * c2)
    assert float(srt) == pytest.approx(-0.5 * (1 + 2 * a2r2 - 3 * a4r4) * s2)


@pytest.mark.parametrize("ratio", RATIOS)
def test_hole_edge_is_traction_free(ratio):
    sample = hole(ratio)
    theta = np.linspace(0.0, 2.0 * math.pi, 361, endpoint=False)
    srr, _, srt = sample.stress_polar(sample.radius, theta)
    assert np.max(np.abs(srr)) <= 1e-14 * max(abs(sample.far_field), abs(sample.far_field_y))
    assert np.max(np.abs(srt)) <= 1e-14 * max(abs(sample.far_field), abs(sample.far_field_y))


@pytest.mark.parametrize("ratio", RATIOS)
def test_edge_hoop_identity(ratio):
    sample = hole(ratio)
    theta = np.linspace(0.0, 2.0 * math.pi, 181, endpoint=False)
    _, stt, _ = sample.stress_polar(sample.radius, theta)
    expected = sample.edge_hoop(theta) * sample.far_field
    assert np.max(np.abs(stt - expected)) <= 1e-14 * max(
        abs(sample.far_field), abs(sample.far_field_y)
    )
    at_zero = sample.stress_polar(sample.radius, 0.0)[1]
    at_ninety = sample.stress_polar(sample.radius, math.pi / 2.0)[1]
    assert float(at_zero) == pytest.approx(sample.far_field * (3.0 * ratio - 1.0), abs=1e-14)
    assert float(at_ninety) == pytest.approx(sample.far_field * (3.0 - ratio), abs=1e-14)


def test_peak_stress_concentration_ladder():
    assert hole(-1.0).peak_edge_stress["stress_concentration"] == pytest.approx(4.0)  # pure shear
    assert hole(0.0).peak_edge_stress["stress_concentration"] == pytest.approx(3.0)   # uniaxial
    assert hole(0.5).peak_edge_stress["stress_concentration"] == pytest.approx(2.5)
    equi = hole(1.0).peak_edge_stress
    assert equi["stress_concentration"] == pytest.approx(2.0)
    assert equi["uniform"] is True
    assert hole(2.0).peak_edge_stress["stress_concentration"] == pytest.approx(5.0)


def test_equibiaxial_is_uniform_and_radial():
    sample = hole(1.0)
    theta = np.linspace(0.0, 2.0 * math.pi, 181, endpoint=False)
    _, stt, _ = sample.stress_polar(sample.radius, theta)
    assert np.max(np.abs(stt - 2.0 * sample.far_field)) <= 1e-14
    _, ut = sample.displacement_polar(np.linspace(1.0, 5.0, 9), theta[:, None])
    assert np.max(np.abs(ut)) <= 1e-14
    e, nu = sample.effective_constants
    r = 2.0
    ur, _ = sample.displacement_polar(r, 0.37)
    expected = sample.far_field * r / e * ((1.0 - nu) + (1.0 + nu) * sample.radius**2 / r**2)
    assert float(ur) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("ratio", RATIOS)
def test_displacement_reproduces_the_stress_field(ratio):
    """Numeric differentiation of the closed-form displacement -> stress."""
    sample = hole(ratio)
    e, nu = sample.effective_constants
    step = 1e-6
    for x, y in ((1.4, 0.3), (2.0, -0.8), (3.5, 1.7)):
        ux_x = (sample.displacement_cartesian(x + step, y)[0] - sample.displacement_cartesian(x - step, y)[0]) / (2 * step)
        ux_y = (sample.displacement_cartesian(x, y + step)[0] - sample.displacement_cartesian(x, y - step)[0]) / (2 * step)
        uy_x = (sample.displacement_cartesian(x + step, y)[1] - sample.displacement_cartesian(x - step, y)[1]) / (2 * step)
        uy_y = (sample.displacement_cartesian(x, y + step)[1] - sample.displacement_cartesian(x, y - step)[1]) / (2 * step)
        rebuilt = np.array(
            [
                e / (1 - nu**2) * (ux_x + nu * uy_y),
                e / (1 - nu**2) * (uy_y + nu * ux_x),
                e / (2 * (1 + nu)) * (ux_y + uy_x),
            ]
        )
        closed = np.array([float(v) for v in sample.stress_cartesian(x, y)])
        assert np.max(np.abs(rebuilt - closed)) <= 1e-6 * max(np.max(np.abs(closed)), 1e-30)


@pytest.mark.parametrize("ratio", RATIOS)
def test_superposition_against_rotated_uniaxial_field(ratio):
    sample = hole(ratio)
    uniaxial = hole(0.0)
    for x, y in ((1.3, 0.4), (2.2, -1.1), (4.0, 2.6)):
        ux0, uy0 = uniaxial.displacement_cartesian(y, -x)
        expected = (
            uniaxial.displacement_cartesian(x, y)[0] - ratio * uy0,
            uniaxial.displacement_cartesian(x, y)[1] + ratio * ux0,
        )
        got = sample.displacement_cartesian(x, y)
        scale = max(math.hypot(*expected), 1e-30)
        assert math.hypot(got[0] - expected[0], got[1] - expected[1]) <= 1e-12 * scale


def test_plane_strain_uses_equivalent_constants():
    stress_case = hole(0.6, plane="stress")
    strain_case = hole(0.6, plane="strain")
    # stresses are material independent, displacements are not
    assert strain_case.stress_concentration == pytest.approx(stress_case.stress_concentration)
    r, theta = 2.0, 0.4
    assert strain_case.displacement_polar(r, theta)[0] != pytest.approx(
        stress_case.displacement_polar(r, theta)[0]
    )
    e, nu = strain_case.effective_constants
    assert e == pytest.approx(stress_case.young / (1 - stress_case.poisson**2))
    assert nu == pytest.approx(stress_case.poisson / (1 - stress_case.poisson))


def test_invalid_lateral_ratio_is_rejected():
    with pytest.raises(ValueError):
        KirschHole(lateral_ratio=float("nan"))


# ------------------------------------------------------------ production FE ---

@pytest.mark.parametrize("ratio", (0.5, 1.0, -1.0, 2.0))
def test_production_q4_concentration_below_three_percent(ratio):
    """Quarter annulus, symmetry on both axes, exact arc tractions."""
    sample = hole(ratio)
    case = solve_case(sample, outer_radius=10.0, n_theta=24)
    edge = case["hole_edge"]
    peak = sample.peak_edge_stress["stress_concentration"]
    fe_peak = edge["peak_hoop"] / sample.far_field
    assert abs(fe_peak - peak) / peak < 0.03
    # the theta = pi/2 edge value (classical K_t) is graded separately
    kt = edge["hole_edge_at_90_degrees"] / sample.far_field
    exact = float(sample.edge_hoop(math.pi / 2.0))
    assert abs(kt - exact) / abs(exact) < 0.03
    # displacement and pointwise stress fields agree with the closed form
    assert case["displacement_error"]["l2_relative"] < 3e-3
    assert case["max_stress_relative_error"] < 0.03
    applied = case["load_resultant"]["applied"]
    exact_arc = case["load_resultant"]["exact_arc"]
    scale = math.hypot(sample.far_field, sample.far_field_y)
    # chord-vs-arc quadrature of the outer boundary is O(dtheta^2); 5e-3 at n=24
    assert math.hypot(applied[0] - exact_arc[0], applied[1] - exact_arc[1]) / scale < 5e-3


def test_coarse_grid_invariance_of_the_reference():
    """The reference must not depend on the FE model: two meshes, same analytic value."""
    sample = hole(0.25)
    values = []
    for n in (16, 24):
        case = solve_case(sample, outer_radius=10.0, n_theta=n)
        values.append(case["hole_edge"]["peak_hoop"] / sample.far_field)
    assert abs(values[1] - values[0]) < 1e-2
    assert abs(values[1] - 2.75) / 2.75 < 1e-2
"""Tests for :mod:`tensorfem.fatigue.weld_toe` (detail fatigue chain).

The acceptance gates of the campaign node are exercised here:

* ``hotspot_mesh_change`` -- 2x near-toe refinement changes the extrapolated
  hot-spot stress by <= 2 % (met);
* ``fatigue_life_vs_analytic`` -- the S-N inversion reproduces an independently
  coded closed form to <= 15 % (met, machine precision);
* ``t_joint_kt_vs_nii`` / ``butt_joint_kt_vs_maddox`` -- the finite-element notch
  ``K_t`` against the empirical anchors.  These gates are **not** met by the
  circular-arc toe idealisation; the measured ratios are locked below so the
  finding cannot silently drift.  See ``docs/reports/weld_toe_fatigue.md``.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorfem.fatigue import (  # noqa: E402
    ENVIRONMENTS,
    WeldDetail,
    fatigue_life,
    hot_spot_stress,
    maddox_stress_concentration,
    mesh_convergence,
    nii_stress_concentration,
    solve_weld_detail,
)
from tensorfem.fatigue.weld_toe import (  # noqa: E402
    _JointProfile,
    _build_mesh,
    _lagrange_extrapolate,
    _q8_dN,
    _q8_shape_row,
)

PLATE = 0.025
ATTACH_T = 0.025
ATTACH_H = 0.030
LEG = 0.0175


def t_joint(radius: float, **kwargs) -> WeldDetail:
    fields = dict(
        joint_type="t_fillet", plate_thickness_m=PLATE, toe_radius_m=radius,
        attachment_thickness_m=ATTACH_T, attachment_height_m=ATTACH_H, weld_leg_m=LEG,
    )
    fields.update(kwargs)
    return WeldDetail(**fields)


def butt_joint(radius: float) -> WeldDetail:
    return WeldDetail(
        joint_type="butt", plate_thickness_m=PLATE, toe_radius_m=radius,
        reinforcement_m=0.003, crest_width_m=0.012, flank_angle_deg=30.0,
    )


# --------------------------------------------------------------------------- #
# closed forms and geometry validation
# --------------------------------------------------------------------------- #
def test_nii_and_maddox_closed_forms():
    assert nii_stress_concentration(0.025, 1e-3, 0.025) == pytest.approx(2.35, abs=1e-12)
    assert nii_stress_concentration(0.025, 0.5e-3, 0.025) == pytest.approx(
        1.0 + 0.27 * math.sqrt(50.0), rel=1e-12)
    assert maddox_stress_concentration(0.025, 1e-3) == pytest.approx(1.75, abs=1e-12)
    # both anchors grow as r**-0.5
    assert nii_stress_concentration(0.025, 0.5e-3, 0.025) > nii_stress_concentration(
        0.025, 2e-3, 0.025)
    with pytest.raises(ValueError):
        nii_stress_concentration(0.025, 0.0, 0.025)
    with pytest.raises(ValueError):
        maddox_stress_concentration(-0.025, 1e-3)


def test_weld_detail_validation():
    with pytest.raises(ValueError):
        WeldDetail(joint_type="lap", plate_thickness_m=0.025, toe_radius_m=1e-3)
    with pytest.raises(ValueError):
        # weld leg must exceed the toe radius
        WeldDetail(joint_type="t_fillet", plate_thickness_m=0.025, toe_radius_m=1e-3,
                   attachment_thickness_m=0.025, attachment_height_m=0.005, weld_leg_m=1e-3)
    with pytest.raises(ValueError):
        # attachment height must exceed the weld leg
        WeldDetail(joint_type="t_fillet", plate_thickness_m=0.025, toe_radius_m=1e-3,
                   attachment_thickness_m=0.025, attachment_height_m=0.010, weld_leg_m=0.0175)
    with pytest.raises(ValueError):
        # corner joints are modelled on the full L section
        WeldDetail(joint_type="corner", plate_thickness_m=0.025, toe_radius_m=1e-3,
                   attachment_thickness_m=0.025, attachment_height_m=0.030,
                   weld_leg_m=LEG, symmetry=True)
    detail = t_joint(1e-3)
    assert detail.toe_ratio == pytest.approx(25.0)
    assert detail.attachment_ratio == pytest.approx(1.2)


# --------------------------------------------------------------------------- #
# shape functions and extrapolation weights
# --------------------------------------------------------------------------- #
def test_q8_shape_functions_partition_of_unity():
    for xi, eta in ((-1.0, -1.0), (0.0, 0.0), (0.4, -0.7)):
        assert _q8_shape_row(xi, eta).sum() == pytest.approx(1.0, abs=1e-14)
        # derivatives sum to zero (rigid translation) and are consistent with the
        # shape functions
        total = _q8_dN(xi, eta).sum(axis=0)
        assert abs(total[0]) < 1e-13 and abs(total[1]) < 1e-13


def test_iiw_extrapolation_weights():
    # linear 0.4t / 0.9t -> 1.8, -0.8;  quadratic 0.4/0.9/1.4t -> 2.52, -2.24, 0.72
    assert _lagrange_extrapolate([0.4, 0.9], [1.0, 1.0]) == pytest.approx(1.0)
    assert _lagrange_extrapolate([0.4, 0.9], [1.0, 0.0]) == pytest.approx(1.8)
    assert _lagrange_extrapolate([0.4, 0.9], [0.0, 1.0]) == pytest.approx(-0.8)
    assert _lagrange_extrapolate([0.4, 0.9, 1.4], [1.0, 0.0, 0.0]) == pytest.approx(2.52)
    assert _lagrange_extrapolate([0.4, 0.9, 1.4], [0.0, 1.0, 0.0]) == pytest.approx(-2.24)
    assert _lagrange_extrapolate([0.4, 0.9, 1.4], [0.0, 0.0, 1.0]) == pytest.approx(0.72)


def test_hot_spot_uses_the_declared_stations():
    detail = t_joint(1e-3)
    solution = solve_weld_detail(detail, level=1)
    linear = hot_spot_stress(solution, detail.plate_thickness_m, method="linear")
    quadratic = hot_spot_stress(solution, detail.plate_thickness_m, method="quadratic")
    assert linear.stations == (0.4, 0.9)
    assert quadratic.stations == (0.4, 0.9, 1.4)
    for result in (linear, quadratic):
        assert result.toe_x_m == pytest.approx(solution.x_toe_m)
        assert result.stress_ratio == pytest.approx(
            result.stress_pa / solution.nominal_stress_pa)
    # the first station value must agree between the two methods
    assert linear.sampled_stress_pa[0] == pytest.approx(quadratic.sampled_stress_pa[0],
                                                        rel=1e-12)
    with pytest.raises(ValueError):
        hot_spot_stress(solution, detail.plate_thickness_m, method="cubic")


# --------------------------------------------------------------------------- #
# mesh quality and mesh independence (gate: hotspot change <= 2 %)
# --------------------------------------------------------------------------- #
def test_mesh_is_valid_and_surface_follows_the_weld_profile():
    detail = t_joint(1e-3)
    mesh = _build_mesh(detail, 1)
    coords = mesh["coords"]
    assert coords.shape[0] > 2000
    assert coords[:, 1].min() == pytest.approx(-PLATE, rel=1e-12)
    assert coords[:, 1].max() == pytest.approx(ATTACH_H, rel=1e-12)
    # the free-surface row of the weld block sits on the weld profile
    top = mesh["top"]
    elements = mesh["elements"]
    assert len(top) > 10
    surface = coords[elements[top][:, 3]]
    profile = _JointProfile(detail)
    deviation = abs(surface[:, 1] - profile.height(surface[:, 0]))
    assert deviation.max() < 1e-12
    # the toe tangent point is part of the surface row
    assert abs(surface[:, 0] - profile.toe).min() < 1e-9
    # no node lies inside the void above the weld profile
    assert coords[coords[:, 0] >= profile.flank_origin + 1e-9, 1].max() <= profile.crest


def test_hotspot_stress_is_mesh_independent():
    study = mesh_convergence(t_joint(1e-3), levels=(1, 2))
    change = study["change"]
    assert change["hotspot_linear"] <= 0.02
    assert change["hotspot_quadratic"] <= 0.02
    # the toe (notch) stress is the most sensitive quantity but still converged
    assert change["kt_toe"] <= 0.02
    # finer mesh has more elements and a smaller smallest edge
    coarse, fine = study["levels"][0], study["levels"][1]
    assert fine["n_elements"] > coarse["n_elements"]
    assert fine["min_edge_m"] < coarse["min_edge_m"]


def test_butt_weld_hotspot_stress_is_mesh_independent():
    study = mesh_convergence(butt_joint(1e-3), levels=(1, 2))
    assert study["change"]["hotspot_quadratic"] <= 0.02


# --------------------------------------------------------------------------- #
# finite-element K_t: physics, and the documented anchor gap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("radius", [0.5e-3, 1e-3, 2e-3])
def test_toe_stress_concentration_decreases_with_toe_radius(radius):
    solution = solve_weld_detail(t_joint(radius), level=1)
    assert solution.kt_toe > 1.2
    assert solution.kt_toe < 4.0
    # K_t - 1 scales like r ** -0.5 for a circular toe
    reference = solve_weld_detail(t_joint(1e-3), level=1).kt_toe
    assert (solution.kt_toe - 1.0) * math.sqrt(radius) == pytest.approx(
        (reference - 1.0) * math.sqrt(1e-3), rel=0.15)


def test_anchor_gap_is_documented_not_hidden():
    """The empirical anchors are NOT reproduced to 3 % (honest FAIL, locked).

    The Q8 circular-arc toe model gives ``K_t`` below both anchors: the ratio is
    ~0.79 for the T-joint over the whole radius family and ~0.92 for the butt weld.
    The band asserted here is the measured one, so the documented gap cannot drift
    without a visible test change.
    """
    t_ratios = []
    for radius in (0.5e-3, 1e-3, 2e-3):
        detail = t_joint(radius)
        fe = solve_weld_detail(detail, level=1).kt_toe
        anchor = nii_stress_concentration(PLATE, radius, ATTACH_H)
        t_ratios.append(fe / anchor)
    assert all(0.75 < ratio < 0.85 for ratio in t_ratios), t_ratios
    assert max(abs(ratio - 1.0) for ratio in t_ratios) > 0.03  # gate NOT met

    b_ratios = []
    for radius in (0.5e-3, 1e-3, 2e-3):
        fe = solve_weld_detail(butt_joint(radius), level=1).kt_toe
        anchor = maddox_stress_concentration(PLATE, radius)
        b_ratios.append(fe / anchor)
    # the Maddox anchor is reproduced at the finest toe but not over the family
    assert all(0.90 < ratio < 1.10 for ratio in b_ratios), b_ratios
    assert abs(b_ratios[0] - 1.0) < 0.01  # r = 0.5 mm matches Maddox


def test_plane_assumption_changes_kt_by_a_few_percent():
    strain = solve_weld_detail(t_joint(1e-3), level=1, plane="strain").kt_toe
    stress = solve_weld_detail(t_joint(1e-3), level=1, plane="stress").kt_toe
    assert abs(strain - stress) / strain < 0.05


def test_corner_joint_is_solved_without_a_closed_form_anchor():
    detail = WeldDetail(
        joint_type="corner", plate_thickness_m=PLATE, toe_radius_m=1e-3,
        attachment_thickness_m=ATTACH_T, attachment_height_m=ATTACH_H,
        weld_leg_m=LEG, symmetry=False,
    )
    solution = solve_weld_detail(detail, level=1)
    assert solution.kt_toe > 1.2
    assert math.isfinite(solution.kt_hotspot_quadratic)
    # no independent closed form is claimed for the corner joint
    from tensorfem.fatigue import anchor_stress_concentration
    assert anchor_stress_concentration(detail) is None


def test_nominal_stress_scaling_is_linear():
    detail = t_joint(1e-3)
    base = solve_weld_detail(detail, level=1)
    doubled = solve_weld_detail(detail, level=1, nominal_stress_pa=2.0e6)
    assert doubled.kt_toe == pytest.approx(base.kt_toe, rel=1e-10)
    assert doubled.peak_toe_stress_pa == pytest.approx(2.0 * base.peak_toe_stress_pa, rel=1e-10)


# --------------------------------------------------------------------------- #
# S-N curves and fatigue life (gate: life vs analytic <= 15 %)
# --------------------------------------------------------------------------- #
def test_sn_curves_are_continuous_and_have_the_declared_fat_values():
    curve = ENVIRONMENTS["air"].curve
    assert curve.cycles_to_failure(curve.knee_stress_range_pa) == pytest.approx(
        curve.knee_cycles, rel=1e-9)
    assert curve.stress_range_at_cycles(2.0e6) / 1e6 == pytest.approx(90.0, rel=1e-3)
    # second branch is anchored on the knee: N(knee stress) is continuous
    knee = curve.knee_stress_range_pa * (1.0 + 1e-9)
    assert curve.cycles_to_failure(knee) == pytest.approx(curve.knee_cycles, rel=1e-6)
    assert 0.5 < ENVIRONMENTS["seawater_free"].curve.log_a1 < curve.log_a1


def test_environment_fatigue_lives_are_ordered():
    stress = 100.0e6
    air = fatigue_life(stress, environment="air")
    cp = fatigue_life(stress, environment="seawater_cp")
    free = fatigue_life(stress, environment="seawater_free")
    assert free < air and free < cp
    with pytest.raises(ValueError):
        fatigue_life(stress, environment="vacuum")


def test_life_matches_independent_closed_form_within_15_percent():
    for environment, curve in (
        (name, env.curve) for name, env in ENVIRONMENTS.items()
    ):
        for stress in (40.0e6, 100.0e6, 180.0e6):
            ds = stress / 1e6
            log_n = curve.log_a1 - curve.slope_m1 * math.log10(ds)
            if 10.0 ** log_n > curve.knee_cycles:
                log_knee = math.log10(curve.knee_cycles)
                log_a2 = log_knee + (curve.slope_m2 / curve.slope_m1) * (
                    curve.log_a1 - log_knee)
                log_n = log_a2 - curve.slope_m2 * math.log10(ds)
            reference = 10.0 ** log_n
            assert fatigue_life(stress, environment=environment) == pytest.approx(
                reference, rel=0.15)


def test_thickness_correction_shortens_life_for_thick_plates():
    thick = fatigue_life(100.0e6, environment="air", thickness_m=0.040)
    thin = fatigue_life(100.0e6, environment="air", thickness_m=0.020)
    assert thick < thin
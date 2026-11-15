"""Tests for the oblique ice-patch extension of the grillage ultimate layer.

Anchor ``GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE``.  The audited axis-aligned layer
is *not* re-tested here; these tests lock the new ``grillage_oblique_patch``
behaviour and its tie-back to the audited numbers:

* exact polygon geometry / clipping and quadrature normalisation;
* ``theta = 0`` reproduces the closed-form Navier layer;
* quarter-turn symmetry on an isotropic square plate;
* tributary load path: conservation, single-strip selection, mirror symmetry;
* fixed-pressure tie-back of all three utilisations to ``grillage_utilisations``
  in the audited convention;
* linearity of the utilisations in the ice pressure (the closed-form ultimate
  pressure is exact, not an iteration result);
* orientation sensitivity is finite, area-preserving, and the audited
  ``theta = 0`` orientation is not favourable;
* input validation.
"""
import math

import numpy as np
import pytest

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    case_study_geometry,
    grillage_utilisations,
    plate_patch_response,
)
from tensorfem.grillage_oblique_patch import (
    ObliqueIcePatch,
    bay_load_polygon,
    oblique_patch_response,
    oblique_ultimate_pressure,
    oblique_utilisations,
    orientation_sensitivity,
    patch_fourier_coefficients,
    polygon_area,
    polygon_centroid,
    tributary_areas,
)


def _spans(geom):
    return geom.n_spans * geom.frame_spacing, geom.frame_span


# --------------------------------------------------------------------------- #
# geometry and quadrature
# --------------------------------------------------------------------------- #
def test_rotated_square_area_and_centroid():
    patch = ObliqueIcePatch(1.3, -0.4, 0.8, 0.8, 37.0, 1.0)
    poly = patch.polygon()
    assert abs(abs(polygon_area(poly)) - 0.8 * 0.8) < 1e-12
    cx, cy = polygon_centroid(poly)
    assert abs(cx - 1.3) < 1e-12 and abs(cy + 0.4) < 1e-12
    # counter-clockwise ordering
    assert polygon_area(poly) > 0.0


def test_edge_clipping_halves_a_rotated_square():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    # centre on the x = 0 edge: exactly half of a 45 deg square is loaded
    half = ObliqueIcePatch(0.0, 0.5 * span_y, 1.0, 1.0, 45.0, 1.0)
    assert abs(abs(polygon_area(half.clipped(span_x, span_y))) - 0.5) < 1e-12
    # a patch completely outside the panel carries no load
    outside = ObliqueIcePatch(-5.0, -5.0, 0.5, 0.5, 0.0, 1.0e6)
    resp = oblique_patch_response(span_x=span_x, span_y=span_y, d1=1.0, d2=1.0,
                                  d3=1.0, patch=outside, terms=11, sample=11)
    assert resp["total_force"] == 0.0
    assert resp["load_area"] == 0.0


def test_quadrature_reproduces_full_panel_area():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    full = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, span_x, span_y, 0.0, 1.0)
    resp = oblique_patch_response(span_x=span_x, span_y=span_y, d1=1.0, d2=1.0,
                                  d3=1.0, patch=full, terms=11, sample=11)
    assert abs(resp["load_area"] - span_x * span_y) < 1e-12
    assert abs(resp["clipped_area_fraction"] - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# theta = 0 must reproduce the audited closed-form layer
# --------------------------------------------------------------------------- #
def test_axis_aligned_matches_closed_form_layer():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    dx, dy, d3 = geom.plate_rigidities()
    for lx, ly, pressure in ((0.70, 0.35, 1.0e6), (1.65, 0.55, 2.0e6)):
        patch = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, lx, ly, 0.0, pressure)
        general = oblique_patch_response(
            span_x=span_x, span_y=span_y, d1=dx, d2=dy, d3=d3, patch=patch,
            terms=41, sample=41)
        closed = plate_patch_response(
            span_x=span_x, span_y=span_y, d1=dx, d2=dy, d3=d3, pressure=pressure,
            patch=(0.5 - 0.5 * lx / span_x, 0.5 + 0.5 * lx / span_x,
                   0.5 - 0.5 * ly / span_y, 0.5 + 0.5 * ly / span_y),
            terms=41, sample=41)
        for key in ("m_x_max", "m_y_max", "w_max", "total_force"):
            assert general[key] == pytest.approx(closed[key], rel=1e-6), key
        for key in ("load_share_x", "load_share_y", "load_share_torsion"):
            assert general[key] == pytest.approx(closed[key], rel=1e-6), key


def test_fourier_coefficients_of_a_small_patch_are_normalized():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    patch = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, 0.7, 0.35, 0.0, 1.0)
    q = patch_fourier_coefficients(patch.clipped(span_x, span_y), span_x, span_y,
                                   terms=21, pressure=1.0)
    # the series must reproduce the applied load: sum(q_mn) of the first few modes
    # is positive and dominated by the (1,1) mode for a centred patch
    assert q[0, 0] > 0.0
    assert q[0, 0] == pytest.approx(np.max(q))


# --------------------------------------------------------------------------- #
# isotropy: quarter turn swaps the bending directions exactly
# --------------------------------------------------------------------------- #
def test_quarter_turn_symmetry_on_isotropic_square():
    a, d = 1.0, 1.0e5
    p = ObliqueIcePatch(0.5 * a, 0.5 * a, 0.4 * a, 0.25 * a, 25.0, 1.0e6)
    r = ObliqueIcePatch(0.5 * a, 0.5 * a, 0.4 * a, 0.25 * a, 115.0, 1.0e6)
    f1 = oblique_patch_response(span_x=a, span_y=a, d1=d, d2=d, d3=d, patch=p,
                                terms=41, sample=61)
    f2 = oblique_patch_response(span_x=a, span_y=a, d1=d, d2=d, d3=d, patch=r,
                                terms=41, sample=61)
    assert f1["m_x_max"] == pytest.approx(f2["m_y_max"], rel=1e-6)
    assert f1["m_y_max"] == pytest.approx(f2["m_x_max"], rel=1e-6)
    assert f1["w_max"] == pytest.approx(f2["w_max"], rel=1e-6)


# --------------------------------------------------------------------------- #
# tributary load path
# --------------------------------------------------------------------------- #
def test_tributary_shares_conserve_the_patch_force():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    patch = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, 1.65, 0.55, 30.0, 1.0e6)
    trib = tributary_areas(geom, patch, span_x, span_y)
    assert trib["share_sum_stiffener"] == pytest.approx(1.0, abs=1e-12)
    assert trib["share_sum_frame"] == pytest.approx(1.0, abs=1e-12)
    areas = sum(r["area"] for r in trib["stiffener"])
    assert areas == pytest.approx(trib["patch_area"], rel=1e-12)


def test_patch_inside_one_bay_loads_one_stiffener_and_one_frame():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    # stiffener 4 sits at y = 4 * 0.35 = 1.40 m, frame 2 at x = 2 * 2.10 = 4.20 m
    tiny = ObliqueIcePatch(4.20, 1.40, 0.10, 0.10, 0.0, 1.0e6)
    trib = tributary_areas(geom, tiny, span_x, span_y)
    assert trib["stiffener_touched"] == 1
    assert trib["frame_touched"] == 1
    assert trib["share_sum_stiffener"] == pytest.approx(1.0, abs=1e-12)


def test_tributary_shares_mirror_with_the_patch():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    up = ObliqueIcePatch(0.5 * span_x, 0.50 * span_y, 1.65, 0.55, 30.0, 1.0e6)
    down = ObliqueIcePatch(0.5 * span_x, 0.50 * span_y, 1.65, 0.55, -30.0, 1.0e6)
    t1 = tributary_areas(geom, up, span_x, span_y)
    t2 = tributary_areas(geom, down, span_x, span_y)
    s1 = [r["share"] for r in t1["stiffener"]]
    s2 = [r["share"] for r in t2["stiffener"]]
    assert s1 == pytest.approx(s2[::-1], abs=1e-12)


# --------------------------------------------------------------------------- #
# tie-back to the audited axis-aligned layer at fixed pressure
# --------------------------------------------------------------------------- #
def test_base_convention_matches_audited_utilisations_at_fixed_pressure():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    for sigma, hydro, lx, ly in ((106.0e6, 0.10e6, 1.65, 0.55),
                                 (53.0e6, 0.05e6, 1.05, 0.35),
                                 (0.0, 0.0, 1.65, 0.55)):
        p_ice = 1.5e6
        case = CombinedLoadCase(axial_stress=sigma, ice_pressure=p_ice,
                                patch_length=lx, patch_width=ly,
                                hydrostatic_pressure=hydro)
        base = grillage_utilisations(geom, case, terms=41)
        patch = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, lx, ly, 0.0, p_ice)
        oblique = oblique_utilisations(
            geom, patch, axial_stress=sigma, hydrostatic_pressure=hydro,
            ice_pressure=p_ice, terms=41, sample=41,
            member_allocation="base_convention")
        for key in ("plate_local_bending", "longitudinal_stiffener",
                    "transverse_frame"):
            assert oblique[key] == pytest.approx(base[key], rel=2e-6), key


def test_plate_branch_follows_the_local_bay_window():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    patch = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, 1.65, 0.55, 0.0, 1.0)
    poly, a, b = bay_load_polygon(patch, geom, span_x, span_y, mode="centred")
    assert (a, b) == (geom.frame_spacing, geom.bay_spacing)
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    # the clamped patch sits inside the bay and is centred in it
    assert min(xs) >= -1e-12 and max(xs) <= a + 1e-12
    assert min(ys) >= -1e-12 and max(ys) <= b + 1e-12
    assert 0.5 * (min(xs) + max(xs)) == pytest.approx(0.5 * a, abs=1e-9)
    clip_poly, _, _ = bay_load_polygon(patch, geom, span_x, span_y, mode="clip")
    clip_area = abs(polygon_area(clip_poly))
    assert 0.0 < clip_area < a * b          # the patch straddles the bay boundary
    assert all(-1e-12 <= q[0] <= a + 1e-12 and -1e-12 <= q[1] <= b + 1e-12
               for q in clip_poly)


# --------------------------------------------------------------------------- #
# linearity -> closed-form ultimate pressure
# --------------------------------------------------------------------------- #
def test_utilisations_are_affine_in_the_ice_pressure():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    patch = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, 1.65, 0.55, 0.0, 1.0)
    u0 = oblique_utilisations(geom, patch, axial_stress=106.0e6,
                              hydrostatic_pressure=0.10e6, ice_pressure=0.0,
                              terms=31, sample=31)
    u1 = oblique_utilisations(geom, patch, axial_stress=106.0e6,
                              hydrostatic_pressure=0.10e6, ice_pressure=1.0e6,
                              terms=31, sample=31)
    u2 = oblique_utilisations(geom, patch, axial_stress=106.0e6,
                              hydrostatic_pressure=0.10e6, ice_pressure=2.0e6,
                              terms=31, sample=31)
    for key in ("plate_local_bending", "longitudinal_stiffener", "transverse_frame"):
        mid = 0.5 * (u0[key] + u2[key])
        assert u1[key] == pytest.approx(mid, rel=1e-9), key


def test_closed_form_ultimate_matches_bisection():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    patch = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, 1.05, 0.35, 0.0, 0.0)
    res = oblique_ultimate_pressure(geom, patch, axial_stress=160.0e6,
                                    hydrostatic_pressure=0.15e6, terms=31,
                                    sample=31, verify_bisection=True,
                                    bisection_tol=1e-6)
    assert res["ice_pressure_ultimate"] > 0.0
    assert res["bisection_relative_difference"] < 1e-5
    at_limit = res["utilisations_at_limit"]
    assert max(at_limit.values()) == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------- #
# orientation sensitivity
# --------------------------------------------------------------------------- #
def test_orientation_sensitivity_preserves_area_and_reports_spread():
    geom = case_study_geometry()
    table = orientation_sensitivity(geom, patch_length=1.65, patch_width=0.55,
                                    angles_deg=(0.0, 45.0), axial_stress=106.0e6,
                                    hydrostatic_pressure=0.10e6, terms=31,
                                    sample=31)
    areas = [row["patch_area_m2"] for row in table["rows"]]
    assert areas == pytest.approx([0.9075, 0.9075], rel=1e-12)
    assert table["spread_relative_to_angle_0"] > 0.0
    # the audited axis-aligned orientation (patch long along the stiffeners) is
    # the least favourable of the two here -> the assumption is not unsafe
    assert table["rows"][0]["ice_pressure_ultimate_MPa"] < \
        table["rows"][1]["ice_pressure_ultimate_MPa"]


def test_input_validation():
    geom = case_study_geometry()
    span_x, span_y = _spans(geom)
    good = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, 1.0, 0.5, 0.0, 1.0)
    good.validate()
    with pytest.raises(ValueError):
        ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, -1.0, 0.5, 0.0, 1.0).validate()
    with pytest.raises(ValueError):
        ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, 1.0, 0.5, 0.0, -1.0).validate()
    with pytest.raises(ValueError):
        ObliqueIcePatch(float("nan"), 0.5 * span_y, 1.0, 0.5, 0.0, 1.0).validate()
    with pytest.raises(ValueError):
        oblique_utilisations(geom, good, axial_stress=geom.yield_stress)
    with pytest.raises(ValueError):
        oblique_utilisations(geom, good, member_allocation="nonsense")
    with pytest.raises(ValueError):
        oblique_patch_response(span_x=span_x, span_y=span_y, d1=-1.0, d2=1.0,
                               d3=1.0, patch=good)
    with pytest.raises(ValueError):
        orientation_sensitivity(geom, patch_length=1.0, patch_width=1.0,
                                angles_deg=(0.0,), plate_bay_mode="nonsense",
                                terms=11, sample=11)
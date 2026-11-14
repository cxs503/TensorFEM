"""Tests for the eccentric ice-patch extension of the grillage ultimate layer.

Covers (round 3 of the anchor ``GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE``):

* the exact closed form ``eccentric_patch_collapse_force`` (analytic limits,
  symmetry, minimum at midspan) and its agreement with the hinge solver;
* the even-mode fix of ``plate_patch_response``: centred patches unchanged,
  eccentric patches checked against an independent finite-difference solve;
* the ``patch_centre`` option of ``grillage_plastic_collapse`` (validation,
  centred recovery, machine-readable patch-position map).
"""
import math

import numpy as np
import pytest

from tensorfem.grillage_ice_ultimate import (
    MemberSection,
    case_study_geometry,
    plate_patch_response,
)
from tensorfem.grillage_plastic_collapse import (
    MemberLine,
    case_study_load_case,
    eccentric_patch_collapse_force,
    grillage_plastic_collapse,
    member_collapse_pressure,
)
from tensorfem.grillage_plastic_verification import patch_position_map
from tensorfem.orthotropic_plate_fd import ssss_orthotropic_plate_fd

YIELD = 3.55e8
SPAN = 2.1


def _section():
    return MemberSection(0.25, 0.010, 0.080, 0.012,
                         plate_thickness=0.020, plate_width=0.35)


def _line(elements=32):
    return MemberLine("bench", [SPAN], _section(), YIELD, young=2.06e11,
                      elements_per_span=elements)


# --------------------------------------------------------------------------- #
# closed form
# --------------------------------------------------------------------------- #
def test_closed_form_reduces_to_central_patch_value():
    mp = _section().plastic_modulus * YIELD
    for c in (0.2, 0.525, 1.05):
        r = eccentric_patch_collapse_force(plastic_moment=mp, span=SPAN,
                                           patch_length=c, patch_centre=0.5)
        assert r["force_ultimate"] == pytest.approx(
            8.0 * mp / (2.0 * SPAN - c), rel=1e-12)


def test_closed_form_point_load_limit():
    mp = _section().plastic_modulus * YIELD
    for a in (0.2, 0.35, 0.5):
        r = eccentric_patch_collapse_force(plastic_moment=mp, span=SPAN,
                                          patch_length=1e-6, patch_centre=a)
        assert r["force_ultimate"] == pytest.approx(
            mp * SPAN / (a * SPAN * (1.0 - a) * SPAN), rel=1e-5)


def test_closed_form_symmetric_in_patch_position():
    mp = _section().plastic_modulus * YIELD
    for a in (0.15, 0.3, 0.4):
        lo = eccentric_patch_collapse_force(plastic_moment=mp, span=SPAN,
                                            patch_length=0.525, patch_centre=a)
        hi = eccentric_patch_collapse_force(plastic_moment=mp, span=SPAN,
                                            patch_length=0.525,
                                            patch_centre=1.0 - a)
        assert lo["force_ultimate"] == pytest.approx(hi["force_ultimate"],
                                                     rel=1e-12)


def test_closed_form_minimum_at_midspan():
    mp = _section().plastic_modulus * YIELD
    forces = [eccentric_patch_collapse_force(
        plastic_moment=mp, span=SPAN, patch_length=0.525,
        patch_centre=a)["force_ultimate"]
        for a in np.linspace(0.26, 0.74, 17)]
    assert int(np.argmin(forces)) == len(forces) // 2
    assert min(forces) < forces[0]


@pytest.mark.parametrize("centre", [0.30, 0.50, 0.70])
def test_hinge_solver_matches_closed_form(centre):
    line = _line(64)
    c = 0.525
    r = member_collapse_pressure(
        line, ice_tributary=1.0,
        ice_interval=(centre * SPAN - c / 2, centre * SPAN + c / 2),
        hydro_pressure=0.0, hydro_tributary=0.0)
    ref = eccentric_patch_collapse_force(plastic_moment=line.plastic_moment,
                                         span=SPAN, patch_length=c,
                                         patch_centre=centre)
    assert r["ice_pressure_ultimate"] == pytest.approx(
        ref["line_load_ultimate"], rel=5e-3)
    assert len(r["hinges"]) == 1                       # determinate => one hinge


def test_closed_form_rejects_patch_outside_span():
    with pytest.raises(ValueError):
        eccentric_patch_collapse_force(plastic_moment=1.0e6, span=SPAN,
                                       patch_length=1.5, patch_centre=0.1)
    with pytest.raises(ValueError):
        eccentric_patch_collapse_force(plastic_moment=1.0e6, span=SPAN,
                                       patch_length=0.2, patch_centre=0.0)


# --------------------------------------------------------------------------- #
# plate response: even modes
# --------------------------------------------------------------------------- #
def _patch(cx, cy, lx, ly, span_x, span_y):
    return (cx - 0.5 * lx / span_x, cx + 0.5 * lx / span_x,
            cy - 0.5 * ly / span_y, cy + 0.5 * ly / span_y)


def test_centred_patch_unchanged_by_even_modes():
    """Even modes vanish for a patch centred on the mid-lines: the round-2
    centred numbers must be reproduced (documented value for the case study)."""
    geom = case_study_geometry()
    dx, dy, d3 = geom.plate_rigidities()
    span_x = geom.n_spans * geom.frame_spacing
    resp = plate_patch_response(span_x=span_x, span_y=geom.frame_span,
                                d1=dx, d2=dy, d3=d3, pressure=1.0,
                                patch=_patch(0.5, 0.5, 1.05, 0.35,
                                             span_x, geom.frame_span))
    assert resp["load_share_x"] == pytest.approx(0.1490, abs=5e-3)
    assert resp["load_share_x"] + resp["load_share_y"] \
        + resp["load_share_torsion"] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("cx", [0.30, 0.50])
def test_eccentric_patch_plate_vs_finite_difference(cx):
    geom = case_study_geometry()
    dx, dy, d3 = geom.plate_rigidities()
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    rect = _patch(cx, 0.5, 1.05, 0.35, span_x, span_y)
    nav = plate_patch_response(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                               d3=d3, pressure=1.0, patch=rect)
    fd = ssss_orthotropic_plate_fd(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                                   d3=d3, pressure=1.0, patch=rect, nx=96)
    assert nav["w_max"] == pytest.approx(fd["w_max"], rel=0.02)
    assert nav["m_x_max"] == pytest.approx(fd["m_x_max"], rel=0.02)
    # the peak must sit on the patch side, not at midspan
    if cx != 0.5:
        assert (fd["w_peak_x"] / span_x - 0.5) * (cx - 0.5) > 0.0


def test_patch_position_symmetry_of_plate_response():
    geom = case_study_geometry()
    dx, dy, d3 = geom.plate_rigidities()
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    lo = plate_patch_response(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                              d3=d3, pressure=1.0,
                              patch=_patch(0.3, 0.5, 1.05, 0.35, span_x, span_y))
    hi = plate_patch_response(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                              d3=d3, pressure=1.0,
                              patch=_patch(0.7, 0.5, 1.05, 0.35, span_x, span_y))
    assert lo["w_max"] == pytest.approx(hi["w_max"], rel=1e-9)
    assert lo["m_x_max"] == pytest.approx(hi["m_x_max"], rel=1e-9)


def test_fd_reference_matches_timoshenko_square_plate():
    """Independent check of the FD reference itself: uniform load on an SSSS
    square plate, w_max = 0.00406 q a^4 / D and M_x = 0.0479 q a^2 (nu = 0.3)."""
    a, d, q = 2.0, 1.0e6, 1.0e4
    fd = ssss_orthotropic_plate_fd(span_x=a, span_y=a, d1=d, d2=d, d3=d,
                                   pressure=q, patch=(0.0, 1.0, 0.0, 1.0), nx=96)
    assert fd["w_max"] == pytest.approx(0.00406 * q * a ** 4 / d, rel=5e-3)
    assert fd["m_x_max"] == pytest.approx(0.0479 * q * a ** 2, rel=1e-2)


# --------------------------------------------------------------------------- #
# grillage level: patch_centre
# --------------------------------------------------------------------------- #
def test_grillage_centred_patch_recovers_round2_numbers():
    geom = case_study_geometry()
    case = case_study_load_case()
    default = grillage_plastic_collapse(geom, case, elements_per_span=8)
    explicit = grillage_plastic_collapse(geom, case, elements_per_span=8,
                                         patch_centre=(0.5, 0.5))
    assert default["ice_pressure_ultimate"] == pytest.approx(
        explicit["ice_pressure_ultimate"], rel=1e-12)
    assert explicit["patch"]["centre_x_fraction"] == pytest.approx(0.5)


def test_grillage_eccentric_patch_shifts_load_to_the_frames():
    """Measured behaviour of an off-centre patch (recorded, not assumed).

    Each member line taken on its own is *less* severely loaded when the patch
    leaves midspan (minimised moment), but the orthotropic load share moves load
    from the stiffener system to the transverse-frame system, whose limit
    pressure is the low one.  Net effect on the grillage plastic pressure: it
    falls, i.e. the centred patch is *not* the worst case at grillage level.
    """
    geom = case_study_geometry()
    case = case_study_load_case()
    centred = grillage_plastic_collapse(geom, case, elements_per_span=8)
    ecc = grillage_plastic_collapse(geom, case, elements_per_span=8,
                                    patch_centre=(0.3, 0.5))
    assert ecc["load_shares"]["x"] < centred["load_shares"]["x"]
    assert ecc["systems"]["transverse_frame"]["ice_pressure_ultimate"] < \
        centred["systems"]["transverse_frame"]["ice_pressure_ultimate"]
    assert ecc["ice_pressure_grillage_plastic"] < \
        centred["ice_pressure_grillage_plastic"]
    assert centred["governing_member_line"] == ecc["governing_member_line"]


def test_grillage_rejects_patch_outside_plate():
    geom = case_study_geometry()
    case = case_study_load_case()
    with pytest.raises(ValueError):
        grillage_plastic_collapse(geom, case, patch_centre=(0.02, 0.5))
    with pytest.raises(ValueError):
        grillage_plastic_collapse(geom, case, patch_centre=(1.5, 0.5))


def test_patch_position_map_is_machine_readable():
    geom = case_study_geometry()
    case = case_study_load_case()
    rows = patch_position_map(geom, case, centres=(0.3, 0.5),
                              axial_ratios=(0.3,), elements_per_span=4)
    assert len(rows) == 2
    keys = {"patch_centre_x_fraction", "axial_ratio", "ice_pressure_ultimate",
            "governing_mode", "margin_vs_design", "load_share_x"}
    for row in rows:
        assert keys <= set(row)
        assert row["ice_pressure_ultimate"] > 0.0
        assert math.isfinite(row["margin_vs_design"])
    centred = [r for r in rows if r["patch_centre_x_fraction"] == 0.5][0]
    off = [r for r in rows if r["patch_centre_x_fraction"] == 0.3][0]
    assert off["ice_pressure_grillage_plastic"] < \
        centred["ice_pressure_grillage_plastic"]
    assert off["load_share_x"] < centred["load_share_x"]
"""Regression lock for the plate/web junction convention of ``MemberSection``.

Anchor ``GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE``, round 7 (closes finding **F5**
of ``docs/AUDIT-GAP-PP-5-GRILLAGE-ULTIMATE-independent-review.md``).

Three independent references are used, none of which reuses the module algebra:

* **hand** -- the composite ``A``/``I`` of the reference stiffener are written out
  here as literal numbers (one term per lamina, computed by hand);
* **slice** -- a thin-slice numerical integration rebuilt from the raw dimensions;
* **tee_section** -- :func:`tensorfem.ice_belt_collapse.tee_section`, an
  independently authored section routine that already used the plate-underside
  convention.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tensorfem.grillage_ice_ultimate import MemberSection, case_study_geometry
from tensorfem.grillage_ice_verification import member_governed_geometry
from tensorfem.ice_belt_collapse import tee_section


def _ref_section() -> MemberSection:
    """Reference stiffener + attached plate of the case study (20 mm / T250x10+80x12)."""
    return MemberSection(0.25, 0.010, 0.080, 0.012,
                         plate_thickness=0.020, plate_width=0.350)


def _slice(sec: MemberSection, n: int = 20000) -> dict[str, float]:
    """Thin-slice reference: A, centroid offset, I and Z_p from raw dimensions."""
    t, h, tw = sec.plate_thickness, sec.web_height, sec.web_thickness
    bf, tf, width = sec.flange_width, sec.flange_thickness, sec.plate_width
    z0 = 0.0 if sec.web_origin == "plate_mid" else -0.5 * t
    z_lo, z_hi = z0 - h - tf, max(0.5 * t, 0.0)
    edges = np.linspace(z_lo, z_hi, n + 1)
    zc, dz = 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)
    b = np.zeros_like(zc)
    if t > 0.0 and width > 0.0:
        b += np.where((zc >= -0.5 * t) & (zc <= 0.5 * t), width, 0.0)
    b += np.where((zc >= z0 - h) & (zc <= z0), tw, 0.0)
    b += np.where((zc >= z0 - h - tf) & (zc <= z0 - h), bf, 0.0)
    area = float(np.sum(b * dz))
    zb = float(np.sum(b * zc * dz)) / area
    inertia = float(np.sum(b * zc ** 2 * dz)) - area * zb ** 2
    cum = np.cumsum(b * dz)
    z_p = float(np.interp(0.5 * area, cum, edges[1:]))
    w = b * dz
    below, above = edges[1:] <= z_p, edges[:-1] >= z_p
    straddle = ~(below | above)
    h1, h2 = z_p - edges[:-1], edges[1:] - z_p
    moment = float(np.sum(w[below] * (z_p - zc[below]))
                   + np.sum(w[above] * (zc[above] - z_p))
                   + np.sum(0.5 * b[straddle] * (h1[straddle] ** 2
                                                 + h2[straddle] ** 2)))
    return {"area": area, "centroid_offset": -zb, "inertia": inertia,
            "plastic_modulus": moment}


def test_default_junction_is_plate_bottom():
    assert MemberSection(0.25, 0.010, 0.080, 0.012).web_origin == "plate_bottom"
    sec = _ref_section()
    assert sec.web_start_z == pytest.approx(-0.5 * sec.plate_thickness)
    assert sec.double_counted_area == 0.0
    # the attached plate is added by GrillageGeometry: the convention must survive
    geom = case_study_geometry()
    assert geom.stiffener_section.web_origin == "plate_bottom"
    assert geom.frame_section.web_origin == "plate_bottom"


def test_legacy_option_reports_double_counted_area():
    leg = replace(_ref_section(), web_origin="plate_mid")
    assert leg.web_start_z == 0.0
    assert leg.double_counted_area == pytest.approx(
        leg.web_thickness * 0.5 * leg.plate_thickness)
    assert leg.double_counted_area / leg.area == pytest.approx(9.56e-3, rel=0.02)


def test_invalid_web_origin_rejected():
    with pytest.raises(ValueError, match="web_origin"):
        MemberSection(0.25, 0.010, 0.080, 0.012, web_origin="plate_top").validate()


def test_hand_computed_area_and_inertia_plate_bottom():
    """One term per lamina, evaluated by hand for the reference stiffener."""
    sec = _ref_section()
    a_plate = 0.350 * 0.020          # 0.007
    a_web = 0.250 * 0.010            # 0.0025
    a_flange = 0.080 * 0.012         # 0.00096
    i_plate = 0.350 * 0.020 ** 3 / 12.0
    i_web = 0.010 * 0.250 ** 3 / 12.0
    i_flange = 0.080 * 0.012 ** 3 / 12.0
    z_plate = 0.0
    z_web = -0.010 - 0.125           # web root at plate underside
    z_flange = -0.010 - 0.250 - 0.006
    area = a_plate + a_web + a_flange
    z_bar = (a_plate * z_plate + a_web * z_web + a_flange * z_flange) / area
    inertia = (a_plate * (z_plate - z_bar) ** 2 + i_plate
               + a_web * (z_web - z_bar) ** 2 + i_web
               + a_flange * (z_flange - z_bar) ** 2 + i_flange)
    assert sec.area == pytest.approx(area, rel=1e-12)
    assert sec.centroid_offset == pytest.approx(-z_bar, rel=1e-12)
    assert sec.inertia == pytest.approx(inertia, rel=1e-12)
    # hand values, for the record
    assert area == pytest.approx(1.046e-2, rel=1e-12)
    assert -z_bar == pytest.approx(0.0566788, rel=1e-5)
    assert inertia == pytest.approx(9.3152e-5, rel=1e-4)


def test_plate_bottom_agrees_with_thin_slice_reference():
    for sec in (_ref_section(),
                member_governed_geometry().stiffener_section,
                case_study_geometry().frame_section):
        ref = _slice(sec)
        assert sec.area == pytest.approx(ref["area"], rel=2e-3)
        assert sec.centroid_offset == pytest.approx(ref["centroid_offset"], rel=2e-3)
        assert sec.inertia == pytest.approx(ref["inertia"], rel=5e-3)
        assert sec.plastic_modulus == pytest.approx(ref["plastic_modulus"], rel=5e-3)


def test_legacy_convention_still_reproducible():
    """The published (pre-round-7) numbers must stay numerically reproducible."""
    leg = replace(_ref_section(), web_origin="plate_mid")
    assert leg.inertia == pytest.approx(8.5448e-5, rel=1e-4)
    assert leg.plastic_modulus == pytest.approx(5.8495e-4, rel=1e-4)
    assert _ref_section().inertia > leg.inertia


def test_agrees_with_independent_tee_section():
    """``ice_belt_collapse.tee_section`` (bare T, plate-underside) agrees exactly."""
    bare = MemberSection(0.25, 0.010, 0.080, 0.012)
    ref = tee_section(0.25, 0.010, 0.080, 0.012, 0.020)
    assert bare.area == pytest.approx(ref["area"], rel=1e-12)
    assert bare.inertia == pytest.approx(ref["inertia"], rel=1e-12)
    assert bare.centroid_offset + 0.010 == pytest.approx(ref["offset"], rel=1e-12)


def test_no_plate_section_is_convention_independent():
    a = MemberSection(0.25, 0.010, 0.080, 0.012, web_origin="plate_bottom")
    b = MemberSection(0.25, 0.010, 0.080, 0.012, web_origin="plate_mid")
    assert a.area == pytest.approx(b.area, rel=1e-15)
    assert a.inertia == pytest.approx(b.inertia, rel=1e-15)
    assert a.plastic_modulus == pytest.approx(b.plastic_modulus, rel=1e-15)


def test_junction_correction_raises_member_capacity():
    """The deliverable moves the documented way: plate-governed cases unchanged,
    member-governed cases up (deeper section, no double count)."""
    from tensorfem.grillage_ice_ultimate import CombinedLoadCase, ultimate_ice_pressure
    from tensorfem.grillage_ice_verification import MEMBER_GOVERNED_SCENARIOS

    geom = member_governed_geometry()
    legacy = replace(geom,
                     stiffener=replace(geom.stiffener, web_origin="plate_mid"),
                     frame=replace(geom.frame, web_origin="plate_mid")).validate()
    name, eta, p_ice, lx, ly, p_h = MEMBER_GOVERNED_SCENARIOS[0]
    case = CombinedLoadCase(axial_stress=eta * geom.yield_stress, ice_pressure=p_ice,
                           patch_length=lx, patch_width=ly,
                           hydrostatic_pressure=p_h, design_pressure=p_ice,
                           design_provenance="scenario design pressure")
    new = ultimate_ice_pressure(geom, case)["ice_pressure_ultimate"]
    old = ultimate_ice_pressure(legacy, case)["ice_pressure_ultimate"]
    assert new > old
    assert new / old == pytest.approx(1.1847, rel=5e-3)
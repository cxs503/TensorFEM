"""Tests for the grillage ice instability integration (tripping + web buckling).

Verification gates T14-T17:
- T14: instability utilisation is well-formed and the tripping stress is finite
- T15: tripping-governed geometry -> instability > beam-column at moderate axial
- T16: instability utilisation monotone in ice pressure
- T17: tripping-governed geometry: p_ult with instability < p_ult without
"""
from __future__ import annotations

import math

import pytest

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    MemberSection,
    case_study_geometry,
    grillage_utilisations,
    ultimate_ice_pressure,
)
from tensorfem.grillage_ice_instability import (
    stiffener_instability_utilisation,
    grillage_utilisations_with_instability,
    ultimate_ice_pressure_with_instability,
    tripping_governed_geometry,
)


@pytest.fixture(scope="module")
def geom_stocky():
    return case_study_geometry()


@pytest.fixture(scope="module")
def geom_tripping():
    return tripping_governed_geometry()


# ---- T14: instability utilisation is well-formed ----

def test_t14_instability_well_formed(geom_stocky):
    """For the case-study geometry, the instability utilisation must be a
    finite positive number and the tripping stress must be below yield."""
    case = CombinedLoadCase(
        axial_stress=100e6, ice_pressure=2.0e6,
        patch_length=0.70, patch_width=0.35,
        hydrostatic_pressure=0.10e6,
    ).validate(geom_stocky)
    u = stiffener_instability_utilisation(geom_stocky, case)
    assert math.isfinite(u["stiffener_instability"])
    assert u["stiffener_instability"] > 0.0
    assert u["limiting_stress_Pa"] > 0.0
    assert u["limiting_stress_Pa"] <= geom_stocky.yield_stress * 1.5, (
        f"limiting stress {u['limiting_stress_Pa']/1e6:.1f} MPa unreasonably high"
    )
    # The tripping stress for T250x10+80x12 is ~223 MPa (below beam-column
    # sigma_L ~331 MPa), so instability can govern over beam-column
    assert u["tripping_sigma_free_Pa"] < geom_stocky.yield_stress, (
        "tripping stress should be below yield for a realistic stiffener"
    )


# ---- T15: tripping-governed geometry -> instability > beam-column ----

def test_t15_tripping_governed_instability_exceeds_beam_column(geom_tripping):
    """For the tripping-governed geometry (T350x7+70x10), the instability
    utilisation must exceed the beam-column utilisation at moderate axial stress."""
    case = CombinedLoadCase(
        axial_stress=150e6, ice_pressure=1.5e6,
        patch_length=1.0, patch_width=0.50,
        hydrostatic_pressure=0.10e6,
    ).validate(geom_tripping)
    u = grillage_utilisations_with_instability(geom_tripping, case)
    assert u["stiffener_instability"] > u["longitudinal_stiffener"], (
        f"T15: instability {u['stiffener_instability']:.4f} <= "
        f"beam-column {u['longitudinal_stiffener']:.4f}"
    )


# ---- T16: monotonicity of instability utilisation in ice pressure ----

def test_t16_instability_monotone_in_ice_pressure(geom_tripping):
    """The instability utilisation must be monotone increasing in ice pressure."""
    case_base = CombinedLoadCase(
        axial_stress=100e6, ice_pressure=0.0,
        patch_length=1.0, patch_width=0.50,
        hydrostatic_pressure=0.10e6,
    ).validate(geom_tripping)
    pressures = [0.5e6, 1.0e6, 1.5e6, 2.0e6, 3.0e6]
    prev = 0.0
    for p in pressures:
        u = stiffener_instability_utilisation(geom_tripping, case_base, ice_pressure=p)
        curr = u["stiffener_instability"]
        assert curr >= prev - 1e-12, (
            f"T16: instability not monotone at p={p/1e6:.1f} MPa: "
            f"{curr:.6f} < {prev:.6f}"
        )
        prev = curr


# ---- T17: tripping-governed geometry: p_ult with instability < without ----

def test_t17_tripping_reduces_ultimate_pressure(geom_tripping):
    """The ultimate ice pressure with the instability check must be strictly
    lower than without it for the tripping-governed geometry."""
    case = CombinedLoadCase(
        axial_stress=100e6, ice_pressure=0.0,
        patch_length=1.0, patch_width=0.50,
        hydrostatic_pressure=0.10e6,
    ).validate(geom_tripping)
    res_without = ultimate_ice_pressure(geom_tripping, case)
    res_with = ultimate_ice_pressure_with_instability(geom_tripping, case)
    p_without = float(res_without["ice_pressure_ultimate"])
    p_with = float(res_with["ice_pressure_ultimate"])
    assert p_with < p_without, (
        f"T17: p_ult with instability ({p_with/1e6:.4f} MPa) >= "
        f"p_ult without ({p_without/1e6:.4f} MPa)"
    )
    assert res_with["governing_member"] == "stiffener_instability", (
        f"T17: governing member is {res_with['governing_member']}, "
        f"expected stiffener_instability"
    )


# ---- Additional: instability utilisation is well-formed ----

def test_instability_utilisation_keys(geom_stocky):
    case = CombinedLoadCase(
        axial_stress=50e6, ice_pressure=1.0e6,
        patch_length=0.35, patch_width=0.35,
        hydrostatic_pressure=0.05e6,
    ).validate(geom_stocky)
    u = stiffener_instability_utilisation(geom_stocky, case)
    assert "stiffener_instability" in u
    assert "limiting_stress_Pa" in u
    assert "governing_instability" in u
    assert u["limiting_stress_Pa"] > 0
    assert u["stiffener_instability"] >= 0


def test_utilisations_with_instability_preserves_base_keys(geom_stocky):
    case = CombinedLoadCase(
        axial_stress=50e6, ice_pressure=1.0e6,
        patch_length=0.35, patch_width=0.35,
        hydrostatic_pressure=0.05e6,
    ).validate(geom_stocky)
    u_base = grillage_utilisations(geom_stocky, case)
    u_full = grillage_utilisations_with_instability(geom_stocky, case)
    # All base keys must be preserved exactly
    for k in ("plate_local_bending", "longitudinal_stiffener", "transverse_frame"):
        assert abs(u_full[k] - u_base[k]) < 1e-12, f"key {k} changed"
    # New keys must be present
    assert "stiffener_instability" in u_full
    assert "instability_governing" in u_full


def test_tripping_governed_geometry_valid():
    geom = tripping_governed_geometry()
    assert geom.frame_spacing == 2.5
    assert geom.bay_spacing == 0.50
    assert geom.stiffener.web_height == 0.350
    assert geom.stiffener.web_thickness == 0.007
    # h_w / t_w = 50 -> deep, thin web
    assert geom.stiffener.web_height / geom.stiffener.web_thickness == pytest.approx(50.0)


def test_case_study_tripping_stress_below_beam_column_limit(geom_stocky):
    """Key finding: even the case-study T250x10+80x12 stiffener has a tripping
    stress (~223 MPa) below the beam-column limiting stress (~331 MPa).
    This means the instability check is relevant for typical ice-belt stiffeners."""
    case = CombinedLoadCase(
        axial_stress=0.0, ice_pressure=0.0,
        patch_length=0.35, patch_width=0.35,
        hydrostatic_pressure=0.0,
    ).validate(geom_stocky)
    u = stiffener_instability_utilisation(geom_stocky, case)
    from tensorfem.grillage_ice_ultimate import stiffener_axial_capacity
    cap = stiffener_axial_capacity(geom_stocky)
    sigma_L = cap["limiting_stress"]
    sigma_trip = u["tripping_sigma_free_Pa"]
    assert sigma_trip < sigma_L, (
        f"tripping stress {sigma_trip/1e6:.1f} MPa >= "
        f"beam-column limit {sigma_L/1e6:.1f} MPa"
    )

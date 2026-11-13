"""Tests for the ice-grillage ultimate strength module and its verification face.

Anchors: the T1-T10 verification suite (classical plate coefficients, exact
beam-column magnification, rigid-plastic frame collapse, Paik-Thampalli
comparison, monotonicity, load-path equilibrium, independent statics), plus
API-level validation and the machine-readable map artifact.
"""
from __future__ import annotations

import json
import math

import pytest

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    MemberSection,
    beam_column_amplification_approx,
    beam_column_amplification_exact,
    beam_moment_centre_patch,
    case_study_geometry,
    frame_plastic_collapse_force,
    grillage_utilisations,
    ice_pressure_area_rule,
    panel_stress_ultimate_map,
    plate_patch_response,
    stiffener_axial_capacity,
    ultimate_ice_pressure,
    write_ultimate_map,
)
from tensorfem.grillage_ice_verification import (
    case_study_geometry_slender,
    combined_case_table,
    run_grillage_verification,
)


@pytest.fixture(scope="module")
def geom():
    return case_study_geometry()


def test_verification_suite_passes():
    result = run_grillage_verification()
    failed = [c for c in result["checks"] if not c["passed"]]
    assert result["passed"], failed
    assert result["n_checks"] >= 18


def test_classical_plate_coefficients():
    side, t, q, e, nu = 1.0, 0.010, 1.0e4, 2.06e11, 0.3
    d = e * t ** 3 / (12.0 * (1.0 - nu ** 2))
    r = plate_patch_response(span_x=side, span_y=side, d1=d, d2=d, d3=d,
                             pressure=q, terms=121, sample=21, poisson=nu)
    assert abs(r["w_max"] / (0.00406 * q * side ** 4 / d) - 1.0) < 0.03
    assert abs(r["m_x_max"] / (0.0479 * q * side ** 2) - 1.0) < 0.03


def test_plate_patch_scales_linearly_with_pressure():
    d = 1.887e4
    a = plate_patch_response(span_x=1.0, span_y=0.5, d1=d, d2=d, d3=d,
                             pressure=1.0e4, patch=(0.2, 0.5, 0.1, 0.4),
                             terms=41, sample=9)
    b = plate_patch_response(span_x=1.0, span_y=0.5, d1=d, d2=d, d3=d,
                             pressure=3.0e4, patch=(0.2, 0.5, 0.1, 0.4),
                             terms=41, sample=9)
    assert b["total_force"] == pytest.approx(3.0 * a["total_force"])
    assert b["w_max"] == pytest.approx(3.0 * a["w_max"], rel=1e-9)
    assert b["m_x_max"] == pytest.approx(3.0 * a["m_x_max"], rel=1e-9)


def test_plate_patch_rejects_bad_input():
    d = 1.887e4
    with pytest.raises(ValueError):
        plate_patch_response(span_x=0.0, span_y=1.0, d1=d, d2=d, d3=d, pressure=1e4)
    with pytest.raises(ValueError):
        plate_patch_response(span_x=1.0, span_y=1.0, d1=d, d2=d, d3=d, pressure=1e4,
                             patch=(0.6, 0.4, 0.0, 1.0))
    with pytest.raises(ValueError):
        plate_patch_response(span_x=1.0, span_y=1.0, d1=d, d2=d, d3=d, pressure=0.0)


def test_beam_column_magnification_ordering():
    for ratio in (0.05, 0.25, 0.5, 0.75):
        exact = beam_column_amplification_exact(ratio, 1.0)
        approx = beam_column_amplification_approx(ratio, 1.0)
        assert exact >= approx - 1e-12
        assert abs(exact - approx) / approx < 0.05
    assert beam_column_amplification_exact(0.0, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        beam_column_amplification_exact(1.0, 1.0)


def test_frame_collapse_closed_forms():
    mp, span = 2.0e6, 3.0
    assert frame_plastic_collapse_force(mp, span, span) == pytest.approx(16.0 * mp / span)
    assert frame_plastic_collapse_force(mp, span, 0.0) == pytest.approx(8.0 * mp / span)
    forced = [frame_plastic_collapse_force(mp, span, c) for c in (0.2, 1.0, 3.0)]
    assert forced[0] < forced[1] < forced[2]


def test_beam_moment_closed_form_matches_statics():
    F, L, c = 5.0e4, 2.0, 0.6
    assert beam_moment_centre_patch(F, L, c) == pytest.approx(F * (2 * L - c) / 8.0)
    assert beam_moment_centre_patch(F, L, 0.0) == pytest.approx(F * L / 4.0)
    assert beam_moment_centre_patch(F, L, L) == pytest.approx(F * L / 8.0)


def test_load_path_is_in_equilibrium(geom):
    for eta, p_ice, p_h in ((0.0, 1.0e6, 0.0), (0.3, 2.0e6, 1.0e5), (0.6, 0.5e6, 5e4)):
        case = CombinedLoadCase(axial_stress=eta * geom.yield_stress,
                                ice_pressure=p_ice, patch_length=0.7,
                                patch_width=0.35, hydrostatic_pressure=p_h)
        u = grillage_utilisations(geom, case)
        total = u["force_ice"] + u["force_hydrostatic"]
        carried = u["force_stiffener_system"] + u["force_frame_system"]
        assert carried == pytest.approx(total, rel=1e-12)


def test_ultimate_pressure_is_monotone_and_bracketed(geom):
    prev = math.inf
    for eta in (0.0, 0.2, 0.4, 0.6):
        case = CombinedLoadCase(axial_stress=eta * geom.yield_stress,
                                patch_length=0.7, patch_width=0.35,
                                hydrostatic_pressure=5.0e4)
        res = ultimate_ice_pressure(geom, case)
        p_ult = res["ice_pressure_ultimate"]
        assert res["converged"] and math.isfinite(p_ult)
        assert p_ult < prev
        prev = p_ult
        # the utilisation at the located limit must be at unity
        u = grillage_utilisations(geom, case, ice_pressure=p_ult)
        assert max(u["plate_local_bending"], u["longitudinal_stiffener"],
                   u["transverse_frame"]) == pytest.approx(1.0, abs=1e-4)


def test_ultimate_pressure_shrinks_when_members_are_preloaded(geom):
    """A member already at its limit must report a zero ice reserve, not a crash."""
    reference = ultimate_ice_pressure(geom, CombinedLoadCase(
        axial_stress=0.3 * geom.yield_stress, patch_length=0.7, patch_width=0.35,
        hydrostatic_pressure=1.0e5))["ice_pressure_ultimate"]
    exhausted = ultimate_ice_pressure(geom, CombinedLoadCase(
        axial_stress=0.97 * geom.yield_stress, patch_length=0.7, patch_width=0.35,
        hydrostatic_pressure=5.0e6))
    assert exhausted["ice_pressure_ultimate"] == 0.0
    assert exhausted["utilisation_without_ice"] >= 1.0
    assert reference > 0.0
    assert exhausted["governing_member"] in {"plate_local_bending",
                                             "longitudinal_stiffener",
                                             "transverse_frame"}


def test_stiffener_capacity_agrees_with_paik(geom):
    from tensorfem.stiffened_panel_ultimate import StiffenedPanel
    mine = stiffener_axial_capacity(geom)
    ref = StiffenedPanel(span=geom.frame_spacing, spacing=geom.bay_spacing,
                         plate_thickness=geom.plate_thickness,
                         web_height=geom.stiffener.web_height,
                         web_thickness=geom.stiffener.web_thickness,
                         flange_width=geom.stiffener.flange_width,
                         flange_thickness=geom.stiffener.flange_thickness,
                         young=geom.young, poisson=geom.poisson,
                         yield_stress=geom.yield_stress).validate()
    rel = abs(mine["panel_ultimate_stress"] - ref.panel_ultimate_stress) / \
        ref.panel_ultimate_stress
    assert rel < 0.15, rel
    assert mine["limiting_stress"] <= geom.yield_stress


def test_frame_attached_plating_shear_lag_cap(geom):
    assert geom.frame_attached_width == pytest.approx(min(geom.frame_spacing,
                                                         0.4 * geom.frame_span))
    assert geom.frame_section.plastic_modulus > 0.0


def test_geometry_validation_rejects_impossible_scantlings():
    with pytest.raises(ValueError):
        GrillageGeometry(frame_spacing=0.2, bay_spacing=0.35, n_bays=4, n_spans=4,
                         plate_thickness=0.02,
                         stiffener=MemberSection(0.25, 0.01, 0.08, 0.012)).validate()
    with pytest.raises(ValueError):
        GrillageGeometry(frame_spacing=2.1, bay_spacing=0.35, n_bays=1, n_spans=4,
                         plate_thickness=0.02).validate()  # frame taller than its span? no: n_bays=1 -> span 0.35
    with pytest.raises(ValueError):
        CombinedLoadCase(ice_pressure=1.0e6, patch_length=0.0, patch_width=0.0) \
            .validate(case_study_geometry())


def test_combined_case_table_has_three_scenarios(geom):
    rows = combined_case_table(geom)
    assert len(rows) >= 3
    for row in rows:
        assert row["governing_member"] in {"plate_local_bending",
                                           "longitudinal_stiffener",
                                           "transverse_frame"}
        assert row["ice_pressure_ultimate_MPa"] > 0.0
        assert row["force_stiffener_system_kN"] >= 0.0
        assert row["force_frame_system_kN"] >= 0.0


def test_ultimate_map_is_machine_readable(tmp_path, geom):
    mapping = panel_stress_ultimate_map(
        geom, axial_ratios=(0.0, 0.3), patch_area_ratios=(0.2, 0.6),
        hydrostatic_pressures=(0.0, 1.0e5), design_pressure=2.0e6,
        design_provenance="test")
    assert len(mapping["rows"]) == 8
    path = write_ultimate_map(mapping, tmp_path / "map.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "tensorfem.grillage_ice_ultimate.map/v1"
    assert len(payload["rows"]) == 8
    assert all(row["converged"] for row in payload["rows"])
    ult = [row["ice_pressure_ultimate_MPa"] for row in payload["rows"]]
    assert all(value is not None and value > 0.0 for value in ult)
    # the margin column is only present when a design pressure is supplied
    assert all(row["margin_vs_design"] is not None for row in payload["rows"])


def test_slender_panel_shifts_governing_mode(geom):
    slender = case_study_geometry_slender()
    rows = panel_stress_ultimate_map(slender, axial_ratios=(0.6,),
                                     patch_area_ratios=(1.0,),
                                     hydrostatic_pressures=(1.0e5,))
    row = rows["rows"][0]
    assert row["converged"]
    assert row["governing_member"] != "transverse_frame"
    # a slender panel must be weaker per unit area than the stocky reference
    stocky = panel_stress_ultimate_map(geom, axial_ratios=(0.6,),
                                       patch_area_ratios=(1.0,),
                                       hydrostatic_pressures=(1.0e5,))["rows"][0]
    assert row["ice_pressure_ultimate_MPa"] < stocky["ice_pressure_ultimate_MPa"]


def test_ice_pressure_area_rule_is_delegated():
    rule = ice_pressure_area_rule(p0=2.0e6)
    assert rule.pressure(1.0) == pytest.approx(2.0e6)
    assert rule.pressure(4.0) < rule.pressure(1.0)

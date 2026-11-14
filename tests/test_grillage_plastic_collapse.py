"""Tests for the grillage plastic-collapse (limit analysis) layer."""
from __future__ import annotations

import math

import pytest

from tensorfem.grillage_ice_ultimate import MemberSection, case_study_geometry
from tensorfem.grillage_plastic_collapse import (
    MemberLine,
    case_study_load_case,
    grillage_plastic_collapse,
    member_collapse_pressure,
    two_span_collapse_upper_bound,
)
from tensorfem.grillage_plastic_verification import (
    SPAN,
    YIELD,
    run_grillage_plastic_verification,
)


def _section() -> MemberSection:
    return MemberSection(0.25, 0.010, 0.080, 0.012,
                         plate_thickness=0.020, plate_width=0.35)


def _mp() -> float:
    return _section().plastic_modulus * YIELD


def test_simply_supported_uniform_collapse():
    member = MemberLine("ss", [SPAN], _section(), YIELD, elements_per_span=16)
    res = member_collapse_pressure(member, ice_tributary=1.0, ice_interval=(0.0, SPAN))
    exact = 8.0 * _mp() / SPAN ** 2
    assert res["ice_pressure_ultimate"] == pytest.approx(exact, rel=0.01)
    assert res["mechanism"] is True
    assert len(res["hinges"]) == 1
    assert res["hinges"][0]["node"] == 8


def test_central_patch_closed_form():
    c = 1.0
    member = MemberLine("ss", [SPAN], _section(), YIELD, elements_per_span=16)
    res = member_collapse_pressure(
        member, ice_tributary=1.0,
        ice_interval=(0.5 * (SPAN - c), 0.5 * (SPAN + c)))
    exact = 8.0 * _mp() / (2.0 * SPAN * c - c ** 2)
    assert res["ice_pressure_ultimate"] == pytest.approx(exact, rel=0.01)


def test_clamped_beam_three_hinge_mechanism():
    member = MemberLine("fix", [SPAN], _section(), YIELD,
                        elements_per_span=16, fixed_ends=True)
    res = member_collapse_pressure(member, ice_tributary=1.0,
                                   ice_interval=(0.0, SPAN))
    assert res["ice_pressure_ultimate"] == pytest.approx(16.0 * _mp() / SPAN ** 2,
                                                         rel=0.01)
    xs = sorted(h["x"] for h in res["hinges"])
    assert len(xs) == 3
    assert xs[0] == pytest.approx(0.0) and xs[-1] == pytest.approx(SPAN)
    assert xs[1] == pytest.approx(0.5 * SPAN, abs=0.02 * SPAN)
    signs = sorted(math.copysign(1.0, h["section_moment"]) for h in res["hinges"])
    assert signs == [-1.0, -1.0, 1.0]


def test_two_span_combined_mechanism():
    ub = two_span_collapse_upper_bound(span=SPAN, plastic_moment=_mp())
    assert ub["collapse_load_exact"] == pytest.approx(11.65685 * _mp() / SPAN ** 2,
                                                      rel=1e-4)
    member = MemberLine("ss2", [SPAN, SPAN], _section(), YIELD, elements_per_span=16)
    res = member_collapse_pressure(member, ice_tributary=1.0,
                                   ice_interval=(0.0, 2.0 * SPAN))
    assert res["ice_pressure_ultimate"] == pytest.approx(
        ub["collapse_load_exact"], rel=0.02)
    assert len(res["hinges"]) == 2  # interior support + the span hinge


def test_axial_force_reduces_plastic_moment_and_capacity():
    section = _section()
    py = section.area * YIELD
    full = member_collapse_pressure(
        MemberLine("ss", [SPAN], section, YIELD, elements_per_span=8),
        ice_tributary=1.0, ice_interval=(0.0, SPAN))
    loaded = member_collapse_pressure(
        MemberLine("ss", [SPAN], section, YIELD, elements_per_span=8,
                   axial_force=0.3 * py),
        ice_tributary=1.0, ice_interval=(0.0, SPAN))
    ratio = loaded["ice_pressure_ultimate"] / full["ice_pressure_ultimate"]
    assert ratio == pytest.approx(1.0 - 0.3 ** 2, rel=0.02)
    assert loaded["axial_reduction"] == pytest.approx(0.91)


def test_hydrostatic_dead_load_reduces_ice_capacity():
    without = member_collapse_pressure(
        MemberLine("f", [2.8], case_study_geometry().frame_section, YIELD,
                   elements_per_span=8),
        ice_tributary=0.4, ice_interval=(1.2, 1.6))
    with_hydro = member_collapse_pressure(
        MemberLine("f", [2.8], case_study_geometry().frame_section, YIELD,
                   elements_per_span=8),
        ice_tributary=0.4, ice_interval=(1.2, 1.6),
        hydro_pressure=1.0e5, hydro_tributary=2.1)
    assert with_hydro["ice_pressure_ultimate"] < without["ice_pressure_ultimate"]
    assert without["hydro_moment_max"] == 0.0
    assert with_hydro["hydro_moment_max"] > 0.0


def test_case_study_reports_margin_and_governing_mode():
    geom = case_study_geometry()
    case = case_study_load_case()
    res = grillage_plastic_collapse(geom, case, elements_per_span=8)
    assert res["ice_pressure_ultimate"] <= res["ice_pressure_grillage_plastic"]
    assert res["ice_pressure_ultimate"] <= res["ice_pressure_panel_limit"]
    assert res["margin_vs_design"] == pytest.approx(
        res["ice_pressure_ultimate"] / case.design_pressure)
    assert set(res["systems"]) == {"longitudinal_stiffener", "transverse_frame"}
    shares = res["load_shares"]
    assert shares["x"] + shares["y"] == pytest.approx(1.0)
    assert shares["x"] > 0.0 and shares["y"] > 0.0
    frame = res["systems"]["transverse_frame"]
    assert frame["plastic_reserve_ratio"] == pytest.approx(1.0)  # statically determinate
    assert len(frame["hinges"]) == 1


def test_member_line_validation():
    section = _section()
    with pytest.raises(ValueError):
        MemberLine("bad", [], section, YIELD)
    with pytest.raises(ValueError):
        MemberLine("bad", [SPAN], section, YIELD, elements_per_span=1)
    with pytest.raises(ValueError):
        MemberLine("bad", [SPAN], section, YIELD,
                   axial_force=1.2 * section.area * YIELD)
    member = MemberLine("ok", [SPAN, SPAN], section, YIELD)
    assert member.total_length == pytest.approx(2.0 * SPAN)
    assert member.support_positions() == (0.0, SPAN, 2.0 * SPAN)


def test_patch_interval_validation():
    member = MemberLine("ss", [SPAN], _section(), YIELD)
    with pytest.raises(ValueError):
        member_collapse_pressure(member, ice_tributary=1.0, ice_interval=(1.0, 1.0))


def test_verification_suite_passes():
    res = run_grillage_plastic_verification(out_dir=None, make_figures=False)
    failed = [c for c in res["checks"] if not c["passed"]]
    assert not failed, failed
    assert len(res["checks"]) >= 20
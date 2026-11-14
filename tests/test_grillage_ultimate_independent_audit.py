"""Independent audit gates for the ice-grillage ultimate strength anchor.

The case study of ``GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`` is governed by the
**plate field** in every combined case, so the binding number of the whole anchor
is the peak plate bending moment of one 2.1 m x 0.35 m panel.  These tests
re-derive that number with a discretisation that shares nothing with the Navier
series (``orthotropic_plate_fd``), re-derive the member-limited ultimate in
closed form with own section algebra, and check that the member-governed branch
is exercised at all -- the three things the pre-audit verification face did not
cover.

Every number is computed.  The gates are 0.5 % unless stated.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    case_study_geometry,
    plate_patch_response,
    ultimate_ice_pressure,
)
from tensorfem.grillage_ice_verification import (
    CASE_STUDY_SCENARIOS,
    MEMBER_GOVERNED_SCENARIOS,
    case_study_geometry_slender,
    hand_stiffener_limit,
    member_governed_case_table,
    member_governed_geometry,
    plate_field_fd_check,
)
from tensorfem.orthotropic_plate_fd import ssss_orthotropic_plate_fd


def _case(geom, eta, p_ice, lx, ly, p_h):
    return CombinedLoadCase(
        axial_stress=eta * geom.yield_stress, ice_pressure=p_ice,
        patch_length=lx, patch_width=ly, hydrostatic_pressure=p_h,
        design_pressure=p_ice, design_provenance="test")


@pytest.fixture(scope="module")
def geom():
    return case_study_geometry()


@pytest.fixture(scope="module")
def fd_rows(geom):
    # nx = 160 resolves the peak to ~4e-4 relative (grid convergence checked in
    # test_fd_grid_convergence), i.e. two orders inside the 0.5 % gate.
    return plate_field_fd_check(geom, nx=160)


def test_fd_reproduces_binding_plate_moment(fd_rows):
    """Navier series vs an independent FD discretisation, real panel + patch."""
    assert len(fd_rows) == len(CASE_STUDY_SCENARIOS)
    for row in fd_rows:
        assert abs(row["peak_moment_rel_error"]) < 0.005, row


def test_fd_based_ultimate_matches_module_bisection(fd_rows):
    """The governing ultimate pressure, recomputed without the bisection."""
    for row in fd_rows:
        assert abs(row["p_ult_rel_error"]) < 0.005, row


def test_case_study_is_plate_governed(fd_rows):
    """Honest scope note: the documented case never reaches a member limit.

    This is the audit finding that motivated ``member_governed_geometry``: the
    acceptance case study is plate-field governed, so on its own it does not
    demonstrate the grillage (member) branch.
    """
    for row in fd_rows:
        assert row["governing_member"] == "plate_local_bending", row
        assert abs(row["utilisation_plate_at_limit"] - 1.0) < 1e-3, row


def test_slender_variant_is_also_plate_governed():
    """The ``case_study_geometry_slender`` docstring claim was falsified.

    It claimed to exercise the stiffener-governed branch; measured, the plate
    field still binds for every combined load tried here.  Locked in so the
    claim cannot silently come back.
    """
    geom = case_study_geometry_slender()
    for eta in (0.0, 0.3, 0.6):
        for lx in (0.45, 0.90, 1.35):
            case = _case(geom, eta, 1.0e6, lx, 0.45, 1.0e5)
            res = ultimate_ice_pressure(geom, case, terms=31)
            assert res["governing_member"] == "plate_local_bending", (eta, lx, res)
            assert res["utilisations_at_limit"]["longitudinal_stiffener"] < 1.0


def test_member_governed_branch_is_exercised():
    rows = member_governed_case_table()
    assert len(rows) >= 3
    for row in rows:
        assert row["governing_member"] == "longitudinal_stiffener", row
        # the plate field must stay clear of its own limit, otherwise the case
        # is not a member-governed one
        assert row["utilisation_plate_at_limit"] < 0.9, row
        assert abs(row["utilisation_stiffener_at_limit"] - 1.0) < 1e-3, row


def test_member_ultimate_matches_independent_closed_form():
    geom = member_governed_geometry()
    for name, eta, p_ice, lx, ly, p_h in MEMBER_GOVERNED_SCENARIOS:
        case = _case(geom, eta, p_ice, lx, ly, p_h)
        module = ultimate_ice_pressure(geom, case)["ice_pressure_ultimate"]
        hand = hand_stiffener_limit(geom, case)["p_ult"]
        assert abs(hand / module - 1.0) < 0.005, (name, hand, module)


def test_hand_section_properties_are_independent_algebra():
    """The hand section must match ``MemberSection`` to display precision.

    Two independent transcriptions of the same laminae; a mismatch means one of
    them (or the composite centroid/inertia helper) is wrong.
    """
    geom = member_governed_geometry()
    from tensorfem.grillage_ice_verification import _effective_section

    be = geom.effective_plate_width
    sec = geom.with_plate(geom.stiffener, be)
    area, inertia, c_ext, _ = _effective_section(geom, be)
    assert abs(area / sec.area - 1.0) < 1e-12
    assert abs(inertia / sec.inertia - 1.0) < 1e-12
    assert abs(c_ext / sec.extreme_fibre - 1.0) < 1e-12


def test_fd_grid_convergence(geom):
    """The FD reference itself is grid-converged (2nd order in the stencil)."""
    a, b, d = geom.frame_spacing, geom.bay_spacing, geom.plate_rigidity
    patch = (0.25, 0.75, 0.0, 1.0)
    series = plate_patch_response(span_x=a, span_y=b, d1=d, d2=d, d3=d,
                                  pressure=1.0, patch=patch, terms=121,
                                  sample=41, poisson=geom.poisson)
    ref = max(series["m_x_max"], series["m_y_max"])
    errs = []
    for nx in (60, 120):
        fd = ssss_orthotropic_plate_fd(span_x=a, span_y=b, d1=d, d2=d, d3=d,
                                       pressure=1.0, patch=patch, nx=nx,
                                       poisson=geom.poisson)
        errs.append(abs(max(fd["m_x_max"], fd["m_y_max"]) / ref - 1.0))
    assert errs[1] < errs[0], errs           # refinement improves
    assert errs[1] < 0.005, errs             # and is inside the gate
    # second-order stencil: halving the cell size cuts the error ~4x
    assert errs[0] / max(errs[1], 1e-12) > 3.0, errs


def test_patch_smaller_than_one_bay_is_handled(geom):
    """Guards the clamping path: a patch narrower than one bay still solves."""
    case = _case(geom, 0.0, 1.0e6, 0.20, 0.20, 0.0)
    res = ultimate_ice_pressure(geom, case)
    assert res["converged"]
    assert res["ice_pressure_ultimate"] > 0.0
    row = plate_field_fd_check(geom, nx=120)[0]
    assert math.isfinite(row["peak_moment_fd_per_Pa"])


def test_cross_method_plastic_not_below_elastic():
    """Rigid-plastic redistribution cannot reduce the member-limited load."""
    from tensorfem.grillage_plastic_collapse import grillage_plastic_collapse

    geom = member_governed_geometry()
    name, eta, p_ice, lx, ly, p_h = MEMBER_GOVERNED_SCENARIOS[1]
    case = _case(geom, eta, p_ice, lx, ly, p_h)
    elastic = ultimate_ice_pressure(geom, case)["ice_pressure_ultimate"]
    plastic = grillage_plastic_collapse(geom, case, elements_per_span=4)
    assert plastic["ice_pressure_grillage_plastic"] >= elastic
    # and the documented reserve is reported, not hidden
    assert plastic["ice_pressure_grillage_plastic"] / elastic > 1.5
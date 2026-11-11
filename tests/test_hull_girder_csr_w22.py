"""Tests for the W22 hull-girder ULS module (Smith M_u vs CSR envelope).

The suite is deliberately fast: one case, coarse discretisation, no plotting.
It asserts *invariants*, not just execution:

* the geometry actually reproduces ``Cb`` and the UR S11 modulus closure is met;
* the panel component law is area-consistent, is capped by the weaker (column)
  mechanism and does soften after its peak (the two properties the
  limiting-stress model lacks);
* the registered defect of
  :func:`tensorfem.hull_girder_smith_collapse.interaction_law` (pointwise signed
  minimum keeps the *stronger* compressive mechanism) is reproduced, so that a
  future fix changes this test on purpose;
* the incremental solve conserves fibre area, has an interior peak and closes
  its energy ledger;
* M_u converges with fibre count and curvature steps, stays below the fully
  plastic moment, and the ULS utilization is reported for both directions.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.hull_girder_csr_w22 import (
    CASE_CATALOGUE,
    analyse_direction,
    build_scantlings,
    panel_component_law,
    plastic_moment_analytic,
    run_w22_study,
    section_fibers,
    still_water_and_wave_moments,
    validate_case_geometry,
)
from tensorfem.hull_girder_smith_collapse import (
    column_law,
    interaction_law,
    plate_field_law,
)


CASE = CASE_CATALOGUE[0]


@pytest.fixture(scope="module")
def scantlings():
    members, closure = build_scantlings(CASE)
    return members, closure


def test_geometry_reproduces_block_coefficient():
    geometry = validate_case_geometry(CASE)
    assert geometry["Cb_error"] < 1e-9
    assert 4.0 < geometry["L_over_B"] < 8.0
    assert 2.0 < geometry["B_over_T"] < 3.5


def test_deck_modulus_closes_on_ur_s11_minimum(scantlings):
    _, closure = scantlings
    assert closure["converged"]
    assert closure["W_deck_final"] >= closure["target_modulus_UR_S11"]
    assert closure["W_deck_final"] == pytest.approx(closure["target_modulus_UR_S11"], rel=1e-6)


def test_panel_law_is_area_consistent_and_capped_by_column(scantlings):
    members, _ = scantlings
    deck = members[0]
    law = panel_component_law(deck)
    stiffener = deck.stiffener(law.yield_stress, law.young)
    # peak compressive stress is set by the weaker mechanism, not by yield
    assert law.peak_compressive_stress < 0.9 * law.yield_stress
    assert law.peak_compressive_stress <= stiffener.ultimate_stress * (1.0 + 1e-9)
    assert law.peak_compressive_stress == pytest.approx(stiffener.ultimate_stress, rel=0.02)
    # elastic slope must be E (area weighting is exact)
    eps = 1e-6
    assert abs(law.stress(eps) / (eps * law.young) - 1.0) < 1e-6


def test_panel_law_softens_after_its_peak(scantlings):
    members, _ = scantlings
    law = panel_component_law(members[0])
    peak = law.peak_compressive_stress
    far = abs(law.stress(-0.008))
    assert far < 0.6 * peak, "the component curve must soften (Smith ingredient 1)"


def test_no_softening_variant_plateaus_above_the_softening_peak(scantlings):
    members, _ = scantlings
    soft = panel_component_law(members[0])
    plate = panel_component_law(members[0], softening=False)
    assert abs(plate.stress(-0.008)) > abs(soft.stress(-0.008))
    assert plate.peak_compressive_stress == pytest.approx(soft.peak_compressive_stress, rel=0.02)


def test_interaction_law_defect_is_reproduced(scantlings):
    """Registered defect: ``interaction_law`` keeps the stronger mechanism.

    The existing helper takes the pointwise *signed* minimum; under compression
    that is the larger magnitude, so the column limit (and the area referencing)
    is silently dropped.  This test pins the behaviour so the finding in
    ``papers/reports/w22_hull_girder1.md`` stays reproducible.
    """
    members, _ = scantlings
    deck = members[0]
    plate = plate_field_law(deck.strip(2.35e8, 2.06e11))
    column = column_law(deck.stiffener(2.35e8, 2.06e11))
    combined = interaction_law((plate, column))
    assert combined.peak_compressive_stress > column.peak_compressive_stress * 1.4
    assert combined.stress(-0.002) == pytest.approx(plate.stress(-0.002))


def test_fibre_discretisation_conserves_area(scantlings):
    members, _ = scantlings
    fibers = section_fibers(members, strips=4)
    total = sum(member.total_area for member in members)
    assert sum(f.area for f in fibers) == pytest.approx(total, rel=1e-12)
    assert len(fibers) > len(members)


def test_incremental_solve_has_interior_peak_and_closes_ledger(scantlings):
    members, _ = scantlings
    hog = analyse_direction(members, condition="hogging", strips=4, steps=160)
    sag = analyse_direction(members, condition="sagging", strips=4, steps=160)
    for analysis in (hog, sag):
        assert analysis["peak_is_interior"]
        assert analysis["ultimate_moment"] > 0.0
        assert analysis["first_yield_moment"] < analysis["ultimate_moment"]
        assert analysis["ledger_residual"] < 5e-3
        assert analysis["fiber_area_relative_error"] < 1e-12


def test_ultimate_moment_converges(scantlings):
    members, _ = scantlings
    coarse = analyse_direction(members, condition="hogging", strips=4, steps=160)
    fine = analyse_direction(members, condition="hogging", strips=8, steps=320)
    relative = abs(fine["ultimate_moment"] - coarse["ultimate_moment"]) / fine["ultimate_moment"]
    assert relative < 0.02


def test_ultimate_below_fully_plastic_moment(scantlings):
    members, _ = scantlings
    plastic = plastic_moment_analytic(members)
    hog = analyse_direction(members, condition="hogging", strips=4, steps=160)
    sag = analyse_direction(members, condition="sagging", strips=4, steps=160)
    assert 0.3 < hog["ultimate_moment"] / plastic["M_p"] < 1.0
    assert 0.3 < sag["ultimate_moment"] / plastic["M_p"] < 1.0
    assert plastic["M_p"] > 0.0


def test_still_water_equilibrium_and_wave_moment_are_physical():
    loads = still_water_and_wave_moments(CASE)
    assert loads["equilibrium"]["force_residual_rel"] < 1e-9
    assert loads["equilibrium"]["moment_residual_rel"] < 1e-9
    assert math.isfinite(loads["M_sw_mid_signed"])
    assert 0.5 < loads["M_wave_design_to_rule_ratio"] < 1.5
    # rule wave moments are both positive; with the UR S11 coefficients
    # implemented in this repository the sagging value exceeds the hogging value
    # for every realistic Cb (0.19 Cb < 0.11 (Cb + 0.7) below Cb = 0.96)
    assert loads["M_wh_rule"] > 0.0
    assert loads["M_ws_rule"] > 0.0


def test_study_reports_criteria_without_inventing_factors():
    study = run_w22_study(cases=(CASE,), strips=4, steps=160,
                          convergence_strips=8, convergence_steps=320)
    verdicts = {row["id"]: row for row in study["criteria"]}
    assert set(verdicts) == {"C1", "C2", "C3", "C4"}
    # the anchor table is not available: C1 must be an honest FAIL, not a PASS
    assert verdicts["C1"]["verdict"] == "FAIL"
    assert verdicts["C1"]["value"] == 0.0
    row = study["cases"][0]
    assert row["uls"]["factors_applied"].startswith("none")
    assert row["uls"]["gamma_admissible_sagging"] == pytest.approx(
        row["ultimate"]["M_u_sagging"] / row["uls"]["envelope_unfactored_sagging"])
    assert row["yield_check"]["gamma_R_admissible_deck_hogging"] > 1.0
    assert study["anchor_file"] == "benchmarks/anchors/iacs-csr-w22.json"
    assert study["anchor_sha256"] is not None
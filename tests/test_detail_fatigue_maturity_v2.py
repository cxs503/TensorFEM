"""Acceptance tests of the welded-detail fatigue maturity package (P106W2-N10).

Every test targets one machine criterion of the node specification or one of the
defects fixed in this package:

* the load-case defect -- the historical ``carried`` load case pulls the attachment
  while a single pinned node reacts the vertical load, which suppressed the toe
  notch stress by ~20 % (FE/Nii = 0.79-0.82); with the load-free attachment
  (``plate_axial``) the same mesh reproduces the Nii closed form to 4 %;
* the butt-weld mesh degeneracy -- ``surface_band_m >= plate_depth_m`` collapsed the
  graded tail rows onto the anchor row and the Q8 Jacobian went non-positive for
  most of the acceptance family (t = 10/15/20 mm, r = 0.5/1/2 mm);
* the tautological ``R^2`` of a two-station extrapolation, replaced by a
  genuine three/four-station least-squares linearity check;
* the honest FAIL of the IIW ``1.2-1.5`` structural SCF band, locked so that it
  cannot be "fixed" by tuning instead of by physics.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.fatigue import (
    BeamSection,
    beam_nominal_stress,
    fatigue_life,
    hot_spot_stress,
    solve_weld_detail,
)
from tensorfem.fatigue.detail_maturity import (
    IIW_STRUCTURAL_SCF_BAND,
    JOINT_FAMILY,
    beam_to_detail_validation,
    extrapolation_linearity,
    family_geometry,
    gci_hotspot,
    power_law_response,
    radius_negative_controls,
    run_maturity_assessment,
    scf_vs_iiw,
    toe_mesh_resolution,
)
from tensorfem.fatigue.weld_toe import nii_stress_concentration


@pytest.fixture(scope="module")
def ledger() -> dict[str, object]:
    return run_maturity_assessment()


# --------------------------------------------------------------------------- #
# D1: load case -- the attachment load must not short-circuit through the pin
# --------------------------------------------------------------------------- #
def test_plate_axial_recovers_the_nii_anchor():
    """FE notch SCF of the load-free attachment family is within 4 % of Nii."""
    ratios = []
    for t, r in JOINT_FAMILY:
        detail = family_geometry(t, r, load_case="plate_axial")
        sol = solve_weld_detail(detail, level=1)
        anchor = nii_stress_concentration(t, r, detail.attachment_height_m)
        ratios.append(sol.kt_toe / anchor)
    assert min(ratios) > 0.95, ratios
    assert max(ratios) < 1.05, ratios


def test_carried_load_case_underpredicts_the_toe_notch():
    """Regression lock of the fixed defect: the old load case gives ~0.8 of Nii."""
    ratios = []
    for t, r in JOINT_FAMILY:
        detail = family_geometry(t, r, load_case="carried")
        sol = solve_weld_detail(detail, level=1)
        anchor = nii_stress_concentration(t, r, detail.attachment_height_m)
        ratios.append(sol.kt_toe / anchor)
    assert max(ratios) < 0.85, ratios


def test_unknown_load_case_is_rejected():
    with pytest.raises(ValueError):
        family_geometry(0.020, 0.001, load_case="bending")


def test_load_carrying_case_matches_beam_theory():
    """The supported load-carrying case reproduces the beam section stress."""
    result = beam_to_detail_validation(0.020, 0.002)
    assert abs(result["relative_error"]) <= 0.05
    assert result["pass"] is True


# --------------------------------------------------------------------------- #
# D2: butt-weld (and large radius) mesh degeneracy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("t,r", JOINT_FAMILY)
def test_butt_family_builds_without_degenerate_jacobian(t, r):
    """Every acceptance butt weld meshes and solves (was a hard crash)."""
    detail = family_geometry(t, r, joint_type="butt", load_case="plate_axial")
    sol = solve_weld_detail(detail, level=1)
    assert sol.n_elements > 500
    assert math.isfinite(sol.kt_toe) and sol.kt_toe > 1.0


def test_surface_band_never_reaches_the_anchor_row():
    """The surface band stays inside the modelled depth (root cause of D2)."""
    for joint_type in ("t_fillet", "butt", "corner"):
        for t, r in ((0.010, 0.002), (0.020, 0.002), (0.020, 0.020)):
            detail = family_geometry(t, r, joint_type=joint_type)
            assert detail.surface_band_m < detail.plate_depth_m
            assert detail.surface_band_m <= 0.85 * detail.plate_depth_m + 1e-15


def test_large_toe_radius_t_joint_solves():
    """r = t (weld leg above the plate depth) used to degenerate."""
    detail = family_geometry(0.020, 0.020, load_case="plate_axial")
    sol = solve_weld_detail(detail, level=1)
    assert sol.kt_toe > 1.0


# --------------------------------------------------------------------------- #
# Criterion: toe mesh size <= t / 10
# --------------------------------------------------------------------------- #
def test_toe_mesh_size_is_below_a_tenth_of_the_thickness():
    detail = family_geometry(0.020, 0.002)
    result = toe_mesh_resolution(detail, level=1)
    assert result["pass"] is True
    assert result["surface_element_over_t"] < 0.02


# --------------------------------------------------------------------------- #
# Criterion: mesh convergence and GCI
# --------------------------------------------------------------------------- #
def test_gci_of_the_hot_spot_is_within_three_percent():
    detail = family_geometry(0.020, 0.002)
    result = gci_hotspot(detail, levels=(1, 2, 3))
    assert result["pass"] is True
    assert result["gci_fine"] <= 0.03
    assert result["change_finest_pair"] <= 0.02


# --------------------------------------------------------------------------- #
# Criterion: hot-spot extrapolation linearity (strict R^2)
# --------------------------------------------------------------------------- #
def test_two_station_r_squared_is_flagged_degenerate():
    """A two-point 'linearity check' has R^2 = 1 by construction, not by physics."""
    detail = family_geometry(0.020, 0.002, load_case="attachment_supported")
    sol = solve_weld_detail(detail, level=1)
    result = extrapolation_linearity(sol, 0.020)
    assert result["sets"]["linear_0_5_1_5"]["degenerate"] is True
    assert result["sets"]["linear_0_5_1_5"]["r_squared"] == pytest.approx(1.0)
    strict = {k: v for k, v in result["sets"].items() if not v["degenerate"]}
    assert strict, "no strict station set was evaluated"


def test_strict_linearity_of_the_load_carrying_hot_spot():
    detail = family_geometry(0.020, 0.002, load_case="attachment_supported")
    sol = solve_weld_detail(detail, level=1)
    result = extrapolation_linearity(sol, 0.020)
    assert result["pass"] is True
    for name in ("half_1_0_1_5", "dense_0_4_to_1_4", "iiw_quadratic"):
        assert result["sets"][name]["r_squared"] >= 0.999


def test_flat_field_case_is_not_scored_as_noise():
    """Non-load-carrying detail: the surface path is flat, R^2 is noise."""
    detail = family_geometry(0.020, 0.002, load_case="plate_axial")
    sol = solve_weld_detail(detail, level=1)
    result = extrapolation_linearity(sol, 0.020)
    values = result["sets"]["dense_0_4_to_1_4"]["sampled_stress_ratio"]
    assert max(values) - min(values) < 0.01   # structural stress equals nominal
    assert result["sets"]["dense_0_4_to_1_4"]["r_squared"] < 0.999


# --------------------------------------------------------------------------- #
# Criterion: S-N power law
# --------------------------------------------------------------------------- #
def test_stress_range_doubling_multiplies_life_by_eight():
    result = power_law_response(100.0e6)
    assert result["pass"] is True
    assert result["measured_exponent"] == pytest.approx(3.0, abs=1e-6)
    assert result["life_ratio_measured"] == pytest.approx(8.0, rel=1e-9)
    assert result["max_log_residual"] <= 0.01


def test_dnv_curve_d_life_is_reproducible_from_the_table():
    """N = 10^12.164 * ds^-3 on the m1 branch of DNV-RP-C203 curve D."""
    for stress_mpa in (60.0, 80.0, 120.0):
        n = fatigue_life(stress_mpa * 1.0e6, environment="air")
        expected = 10.0**12.164 * stress_mpa ** -3.0
        assert n / expected == pytest.approx(1.0, rel=1e-9)


def test_thickness_correction_only_above_25_mm():
    thin = fatigue_life(100.0e6, environment="air", thickness_m=0.020)
    thick = fatigue_life(100.0e6, environment="air", thickness_m=0.040)
    assert fatigue_life(100.0e6, environment="air", thickness_m=0.025) == pytest.approx(thin)
    assert thick < thin


# --------------------------------------------------------------------------- #
# Criterion: negative controls on the toe radius
# --------------------------------------------------------------------------- #
def test_radius_sweep_is_monotone_and_has_a_singularity_trend():
    result = radius_negative_controls(0.020)
    assert result["monotone_decreasing_kt"] is True
    assert result["singularity_trend"] is True
    rows = result["rows"]
    assert rows[0]["kt_peak"] > 3.0            # r -> 0: notch grows, no crash
    assert rows[-1]["kt_peak"] < rows[0]["kt_peak"]


def test_structural_hot_spot_tends_to_one_at_large_radius():
    result = radius_negative_controls(0.020)
    assert result["structural_limit_pass"] is True
    assert result["structural_deviation_largest_radius"] <= 0.005


def test_notch_scf_limit_is_documented_as_asymptotic():
    """The 0.5 % band on the notch SCF is only reached at r/t ~ 650 (honest)."""
    result = radius_negative_controls(0.020)
    assert result["notch_limit_pass"] is False
    assert result["radius_over_t_for_half_percent"] > 100.0
    assert 0.0 < result["decay_exponent"] < 1.5


# --------------------------------------------------------------------------- #
# Criterion: IIW structural SCF band (honest FAIL, locked)
# --------------------------------------------------------------------------- #
def test_iiw_band_is_not_met_and_the_reason_is_locked():
    """The transverse non-load-carrying fillet has a structural SCF of ~1.0.

    The IIW 1.2-1.5 band quoted by the node belongs to structural-discontinuity
    details (bracket toes, load-carrying joints).  This test locks the measured
    value so that the gap cannot be closed by tuning.
    """
    result = scf_vs_iiw()
    lo, hi = IIW_STRUCTURAL_SCF_BAND
    assert result["band_pass"] is False
    assert 0.98 <= result["structural_min"] <= 1.02
    assert result["structural_max"] <= 1.02
    t_peaks = [row["kt_peak"] for row in result["rows"] if row["joint_type"] == "t_fillet"]
    assert min(t_peaks) > hi      # the T-joint notch SCF overshoots the band
    assert result["worst_structural_deviation_from_mid"] > 0.15
    # the butt weld at t/r = 5 does fall inside the band, which is why the band is
    # quoted for the notch stress of low t/r details rather than for the T-joint
    b_peaks = [row["kt_peak"] for row in result["rows"] if row["joint_type"] == "butt"]
    assert lo <= min(b_peaks) <= hi


def test_anchor_errors_are_reported_per_joint_type():
    result = scf_vs_iiw()
    t_errors = [row["anchor_error"] for row in result["rows"] if row["joint_type"] == "t_fillet"]
    b_errors = [row["anchor_error"] for row in result["rows"] if row["joint_type"] == "butt"]
    assert max(abs(e) for e in t_errors) <= 0.05
    assert min(b_errors) > 0.0            # the Maddox fit is exceeded, documented
    assert max(b_errors) <= 0.20


# --------------------------------------------------------------------------- #
# Beam -> detail chain
# --------------------------------------------------------------------------- #
def test_beam_theory_to_hot_spot_chain():
    section = BeamSection(area_m2=0.020, I_m4=0.020**3 / 12.0, c_m=0.010)
    sigma_nom = beam_nominal_stress(axial_force_n=1.2e6, bending_moment_nm=0.0, section=section)
    assert sigma_nom == pytest.approx(60.0e6, rel=1e-12)
    detail = family_geometry(0.020, 0.002, load_case="plate_axial")
    sol = solve_weld_detail(detail, level=1)
    lin = hot_spot_stress(sol, detail.plate_thickness_m, method="linear")
    sigma_hot = lin.stress_ratio * sigma_nom
    n = fatigue_life(sigma_hot, environment="air", thickness_m=0.020)
    assert n == pytest.approx(10.0**12.164 * (sigma_hot / 1e6) ** -3.0, rel=1e-9)
    assert 1.0 < sigma_hot / sigma_nom < 1.05


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
def test_ledger_reports_every_criterion(ledger):
    criteria = ledger["criteria"]
    expected = {
        "scf_iiw_band",
        "scf_closed_form",
        "mesh_size",
        "mesh_convergence",
        "hotspot_linearity",
        "fatigue_power_law",
        "radius_negative_control",
        "beam_detail_chain",
    }
    assert set(criteria) == expected
    for name, row in criteria.items():
        assert isinstance(row["pass"], bool)
        assert row["requirement"]
    assert ledger["all_pass"] is False          # the IIW band criterion is not met
    passing = {name for name, row in criteria.items() if row["pass"]}
    assert passing == expected - {"scf_iiw_band"}
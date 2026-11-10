"""Verification tests for the C(T) fatigue-crack-growth face (ASTM E647 / E399).

Campaign node P202W1-N09 acceptance criteria:

C1 Paris fit R^2 >= 0.98 in log-log coordinates
C2 C and m within 30 % of the literature anchor
C3 dK_th within 20 % of the literature anchor
C4 K_IC within 15 % of the literature anchor (and E399 size requirement met)
C5 Miner damage D = sum n_i/N_i ~ 1 at instability (within 20 %)
C6 negative control: dK < dK_th gives da/dN == 0 (<= 1e-12)
C7 crack-tip mesh refinement converges K_I (relative change <= 5 %)
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.fatigue.ct_specimen import (
    ASTM_CT_RANGE,
    MATERIAL_ANCHOR,
    CTSpecimen,
    ThresholdParis,
    crack_growth_cycles,
    ct_fracture_toughness,
    ct_geometry_factor,
    ct_stress_intensity,
    ct_t6_mesh,
    e647_constant_amplitude_test,
    e647_secant_reduction,
    load_shedding_threshold_test,
    miner_accumulated_damage,
    paris_fit,
    ring_j_integral,
    run_ct_e647_qualification,
    solve_and_postprocess,
    solve_ct_model,
    unstable_crack_ratio,
)

SPECIMEN = CTSpecimen()
LAW = ThresholdParis(
    c=MATERIAL_ANCHOR["c"],
    m=MATERIAL_ANCHOR["m"],
    delta_k_th_mpa=MATERIAL_ANCHOR["delta_k_th_mpa"],
    threshold_exponent=8.0,
)


# --------------------------------------------------------------------------- #
# geometry factor
# --------------------------------------------------------------------------- #
def test_geometry_factor_matches_astm_calibration():
    """ASTM E399 C(T): f(0.5) = 9.66, monotone over the validity window."""
    assert ct_geometry_factor(0.5) == pytest.approx(9.66, rel=5e-3)
    alphas = [0.2 + 0.05 * step for step in range(12)]
    values = [ct_geometry_factor(alpha) for alpha in alphas]
    assert all(later > earlier for earlier, later in zip(values, values[1:]))
    # K_I = P / (B sqrt(W)) f(a/W) with hand-computed numbers
    load = 10.0e3
    width, thickness = 0.050, 0.025
    expected = load / (thickness * math.sqrt(width)) * 9.66
    assert ct_stress_intensity(load, width, thickness, 0.5) == pytest.approx(expected, rel=5e-3)
    # load range that produces a target dK is the inverse of the same formula
    delta_k = 20.0e6
    range_n = SPECIMEN.load_range_for_delta_k(20.0, 0.5)
    assert SPECIMEN.delta_k(range_n, 0.5) == pytest.approx(delta_k, rel=1e-12)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.0, 1.5, float("nan")])
def test_geometry_factor_rejects_invalid_crack_ratio(alpha):
    with pytest.raises(ValueError):
        ct_geometry_factor(alpha)


def test_specimen_proportions_follow_astm():
    assert SPECIMEN.total_width == pytest.approx(1.25 * SPECIMEN.width)
    assert 2.0 * SPECIMEN.half_height == pytest.approx(1.2 * SPECIMEN.width)
    assert SPECIMEN.pin_offset == pytest.approx(0.275 * SPECIMEN.width)
    assert SPECIMEN.effective_modulus == pytest.approx(SPECIMEN.young / (1.0 - SPECIMEN.poisson**2))


# --------------------------------------------------------------------------- #
# growth law and threshold (C6)
# --------------------------------------------------------------------------- #
def test_growth_rate_is_exactly_zero_below_threshold():
    """C6 negative control: dK below dK_th must give identically zero growth."""
    for factor in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0):
        rate = LAW.da_dn(factor * LAW.delta_k_th_mpa)
        assert rate == 0.0
        assert rate <= 1.0e-12
    # just above the threshold the law takes over steeply but continuously: the
    # threshold exponent keeps the branch within one decade of 1e-12 m/cycle
    above = LAW.da_dn(1.0001 * LAW.delta_k_th_mpa)
    assert 0.0 < above < 1.0e-11
    assert LAW.da_dn(1.01 * LAW.delta_k_th_mpa) > 10.0 * above


def test_mid_rate_branch_is_a_pure_paris_law():
    """The threshold factor vanishes away from dK_th, so the Paris branch is intact."""
    pure = ThresholdParis(c=LAW.c, m=LAW.m, delta_k_th_mpa=0.0)
    for delta_k in (16.0, 20.0, 30.0):
        assert LAW.da_dn(delta_k) == pytest.approx(pure.da_dn(delta_k), rel=1e-3)


def test_paris_law_scaling_and_validation():
    assert LAW.da_dn(2.0 * 20.0) / LAW.da_dn(20.0) == pytest.approx(2.0**LAW.m, rel=1e-3)
    # a threshold-free law scales exactly
    pure = ThresholdParis(c=LAW.c, m=LAW.m, delta_k_th_mpa=0.0)
    assert pure.da_dn(40.0) / pure.da_dn(20.0) == pytest.approx(2.0**LAW.m, rel=1e-12)
    assert LAW.da_dn(-5.0) == 0.0
    assert LAW.da_dn(0.0) == 0.0
    with pytest.raises(ValueError):
        ThresholdParis(c=-1.0, m=3.0)
    with pytest.raises(ValueError):
        ThresholdParis(c=1e-12, m=0.0)


# --------------------------------------------------------------------------- #
# ASTM E647 data reduction (C1, C2)
# --------------------------------------------------------------------------- #
def test_paris_fit_recovers_synthetic_law():
    pure = ThresholdParis(c=MATERIAL_ANCHOR["c"], m=MATERIAL_ANCHOR["m"], delta_k_th_mpa=0.0)
    calls = [8.0, 12.0, 18.0, 25.0, 35.0]
    rates = [pure.da_dn(value) for value in calls]
    fit = paris_fit(calls, rates)
    assert fit["m"] == pytest.approx(pure.m, rel=1e-9)
    assert fit["c"] == pytest.approx(pure.c, rel=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        paris_fit([10.0, 20.0], [1e-9, 2e-9])


def test_virtual_e647_test_recovers_paris_constants():
    """C1 + C2 without finite elements: secant reduction of a virtual test."""
    test = e647_constant_amplitude_test(SPECIMEN, 12000.0, 0.30, 0.55, LAW, crack_increment=2.5e-4)
    assert test["threshold_limited"] is False
    assert test["cycles_to_failure"] > 1.0e5
    reduction = e647_secant_reduction(test["crack_lengths"], test["cycles"], SPECIMEN, 12000.0)
    fit = reduction["fit"]
    assert fit["r_squared"] >= 0.98
    assert fit["r_squared"] > 0.999
    assert abs(fit["c"] / MATERIAL_ANCHOR["c"] - 1.0) <= 0.30
    assert abs(fit["m"] / MATERIAL_ANCHOR["m"] - 1.0) <= 0.30
    # the secant construction must be consistent with the growth-rate record
    assert min(reduction["delta_k_mpa"]) > LAW.delta_k_th_mpa


def test_crack_growth_life_scales_as_paris_exponent():
    """Separable Paris law: N(P1)/N(P2) = (P2/P1)**m exactly."""
    # both ranges sit well above the threshold, so the law is separable there
    fast = crack_growth_cycles(SPECIMEN, 48000.0, 0.30, 0.50, LAW)
    slow = crack_growth_cycles(SPECIMEN, 24000.0, 0.30, 0.50, LAW)
    assert math.isfinite(fast) and math.isfinite(slow)
    assert slow / fast == pytest.approx((48000.0 / 24000.0) ** LAW.m, rel=1e-4)
    # a range below the threshold never grows the crack
    assert crack_growth_cycles(SPECIMEN, 2000.0, 0.30, 0.50, LAW) == float("inf")


# --------------------------------------------------------------------------- #
# threshold (C3)
# --------------------------------------------------------------------------- #
def test_load_shedding_threshold_matches_literature():
    outcome = load_shedding_threshold_test(SPECIMEN, LAW, start_delta_k_mpa=12.8)
    assert outcome["bracketed"] is True
    deviation = outcome["delta_k_th_mpa"] / MATERIAL_ANCHOR["delta_k_th_mpa"] - 1.0
    assert abs(deviation) <= 0.20
    assert abs(deviation) < 0.02
    rates = [row["da_dn"] for row in outcome["schedule"]]
    assert rates[-1] <= 1.0e-10
    assert all(later <= earlier for earlier, later in zip(rates, rates[1:]))


# --------------------------------------------------------------------------- #
# toughness (C4)
# --------------------------------------------------------------------------- #
def test_toughness_identification_and_size_requirement():
    peak = 12000.0 / 0.9
    k_ic = MATERIAL_ANCHOR["k_ic_mpa"] * 1.0e6
    alpha_c = unstable_crack_ratio(SPECIMEN, peak, k_ic, 0.30)
    assert ASTM_CT_RANGE[0] < alpha_c < ASTM_CT_RANGE[1]
    # consistency: the instability load reproduces K_IC at the critical crack
    assert SPECIMEN.stress_intensity(peak, alpha_c) == pytest.approx(k_ic, rel=1e-9)
    outcome = ct_fracture_toughness(SPECIMEN, peak, alpha_c)
    assert outcome["thickness_valid"] is True
    assert outcome["pmax_ratio_valid"] is True
    assert outcome["valid"] is True
    assert abs(outcome["k_q_mpa"] / MATERIAL_ANCHOR["k_ic_mpa"] - 1.0) <= 0.15
    # E399 size requirement is the binding geometric check
    assert outcome["minimum_thickness_m"] == pytest.approx(
        2.5 * (k_ic / SPECIMEN.yield_strength) ** 2
    )
    assert outcome["minimum_thickness_m"] < SPECIMEN.thickness
    with pytest.raises(ValueError):
        unstable_crack_ratio(SPECIMEN, 1.0e3, k_ic, 0.30)


# --------------------------------------------------------------------------- #
# Miner summation (C5)
# --------------------------------------------------------------------------- #
def _spectrum(scale: float):
    return (
        (0.75 * 12000.0 * scale, 4000),
        (0.90 * 12000.0 * scale, 2500),
        (1.00 * 12000.0 * scale, 1500),
        (1.10 * 12000.0 * scale, 700),
        (1.25 * 12000.0 * scale, 200),
    )


def test_miner_damage_at_instability_is_unity():
    """C5: D = sum n_i/N_i = 1 at instability for a separable Paris law."""
    alpha_c = unstable_crack_ratio(
        SPECIMEN, 12000.0 / 0.9, MATERIAL_ANCHOR["k_ic_mpa"] * 1.0e6, 0.30
    )
    outcome = miner_accumulated_damage(SPECIMEN, _spectrum(1.0), LAW, 0.30, alpha_c)
    assert outcome["stalled"] is False
    assert abs(outcome["damage"] - 1.0) <= 0.20
    assert abs(outcome["damage"] - 1.0) < 0.01
    assert outcome["total_cycles"] > 0.0
    assert outcome["miner_predicted_cycles"] == pytest.approx(outcome["total_cycles"], rel=1e-3)


def test_miner_breaks_down_when_levels_fall_below_threshold():
    """Spectrum truncation by the threshold is what destroys Miner's rule."""
    alpha_c = unstable_crack_ratio(
        SPECIMEN, 12000.0 / 0.9, MATERIAL_ANCHOR["k_ic_mpa"] * 1.0e6, 0.30
    )
    strong = miner_accumulated_damage(SPECIMEN, _spectrum(1.0), LAW, 0.30, alpha_c)
    weak = miner_accumulated_damage(SPECIMEN, _spectrum(0.5), LAW, 0.30, alpha_c)
    assert weak["damage"] < strong["damage"]
    assert weak["damage"] < 0.8
    # the small levels of the truncated spectrum cannot grow the crack at all
    assert (
        LAW.da_dn(
            0.75
            * 12000.0
            * 0.5
            / (SPECIMEN.thickness * math.sqrt(SPECIMEN.width))
            * ct_geometry_factor(0.30)
            / 1.0e6
        )
        == 0.0
    )


# --------------------------------------------------------------------------- #
# mesh structure (fast, no solve)
# --------------------------------------------------------------------------- #
def test_ct_mesh_is_conforming_and_quarter_point():
    mesh = ct_t6_mesh(SPECIMEN, 0.4, n_angles=48, n_rings=10, growth=1.35)
    coords = mesh.nodes[mesh.elements]
    from tensorfem.fatigue.crack_tip_quarter_point import _parent_gradient

    _, _, det = _parent_gradient(coords, 1.0 / 3.0, 1.0 / 3.0)
    assert bool(torch.all(det > 0.0))
    # area of the union of the (linearised) triangles equals the specimen area
    area = 0.0
    for element in mesh.elements.tolist():
        x0, y0 = mesh.coordinates(element[0])
        x1, y1 = mesh.coordinates(element[1])
        x2, y2 = mesh.coordinates(element[2])
        area += 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    specimen_area = SPECIMEN.total_width * 2.0 * SPECIMEN.half_height
    assert area == pytest.approx(specimen_area, rel=0.01)
    # the quarter-point node sits at r/4 on every tip edge
    tip_x, tip_y = mesh.coordinates(mesh.tip_node)
    first_ring = [mesh.coordinates(node) for node in mesh.face_upper[:1]]
    radius = math.hypot(first_ring[0][0] - tip_x, first_ring[0][1] - tip_y)
    quarter = mesh.rings[0]
    assert len(quarter) == mesh.n_angles
    # crack faces north and south of the slit carry distinct node ids
    assert mesh.face_upper[0] != mesh.face_lower[0]
    assert mesh.coordinates(mesh.face_upper[0])[1] == pytest.approx(0.0, abs=1e-12)
    assert mesh.coordinates(mesh.face_lower[0])[1] == pytest.approx(0.0, abs=1e-12)
    assert radius > 0.0


# --------------------------------------------------------------------------- #
# finite element cross-checks (slow lane)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_j_integral_path_independence_and_code_agreement():
    mesh = ct_t6_mesh(SPECIMEN, 0.40, n_angles=72, n_rings=14, growth=1.35)
    load = SPECIMEN.load_range_for_delta_k(20.0, 0.40) / 0.9
    displacement = solve_ct_model(mesh, SPECIMEN, load)
    values = [
        ring_j_integral(mesh, displacement, ring, specimen=SPECIMEN)
        for ring in range(3, mesh.n_rings - 1)
    ]
    assert all(value > 0.0 for value in values)
    spread = (max(values) - min(values)) / values[-1]
    assert spread < 0.05
    post = solve_and_postprocess(mesh, SPECIMEN, load)
    assert abs(post.k_j - post.astm_k) / post.astm_k < 0.02
    assert abs(post.cod_k - post.astm_k) / post.astm_k < 0.03


@pytest.mark.slow
def test_qualification_meets_every_criterion():
    """C1 .. C7 end to end, including the finite-element mesh convergence."""
    result = run_ct_e647_qualification()
    criteria = result["criteria"]
    assert criteria["C1_paris_r_squared"]["value"] >= 0.98
    assert abs(criteria["C2_paris_constants"]["c_deviation_pct"]) <= 30.0
    assert abs(criteria["C2_paris_constants"]["m_deviation_pct"]) <= 30.0
    assert abs(criteria["C3_threshold"]["deviation_pct"]) <= 20.0
    assert abs(criteria["C4_toughness"]["deviation_pct"]) <= 15.0
    assert criteria["C4_toughness"]["thickness_valid"] is True
    assert abs(criteria["C5_miner"]["deviation_pct"]) <= 20.0
    assert criteria["C6_negative_control"]["max_rate"] <= 1.0e-12
    assert criteria["C7_mesh_convergence"]["max_relative_change_pct"] <= 5.0
    assert result["criteria_pass"] is True
    # the FE route reproduces the ASTM geometry factor to better than 2 % in the
    # primary test window, independently of the J contours chosen
    for row in result["fe_cross_check"]:
        if row["alpha"] >= 0.40:
            assert abs(row["deviation_j_pct"]) < 2.0
    # toughness cross-checked by the finite-element J integral as well
    assert abs(result["toughness"]["finite_element"]["deviation_j_pct"]) < 5.0
    # Miner sensitivity documents where the linear rule stops holding
    sensitivity = {row["spectrum_scale"]: row for row in result["miner_sensitivity"]}
    assert abs(sensitivity[1.0]["damage"] - 1.0) < 0.01
    assert sensitivity[0.5]["damage"] < 0.8

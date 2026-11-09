"""Acceptance tests for the geometry-aware fatigue crack-growth face.

Covers the campaign criteria (a)-(g) plus regression tests for the two SIF
defects fixed in ``tensorfem.fatigue_crack_propagation`` (discontinuous
hole-edge amplification, surface-point Newman-Raju branch dropping ``M1..M3``)
and for the discarded geometry factor in the residual-life helpers.
"""
import math

import pytest

from tensorfem.fatigue.crack_growth_geometry import (
    SNCurveIIW,
    centre_crack_finite_width_delta_k,
    compare_paris_vs_iiw,
    convergence_study,
    critical_crack_length,
    hole_edge_crack_delta_k,
    hole_edge_crack_life,
    integrate_paris_ode,
    richardson_gci,
    walker_effective_delta_k,
    weld_toe_surface_crack_life,
)
from tensorfem.fatigue.crack_growth_qualification import (
    NEWMAN_RAJU_SEMICIRCULAR_F,
    run_geometry_aware_qualification,
)
from tensorfem.fatigue_crack_propagation import (
    FilletWeldDetail,
    HoleEdgeDetail,
    ParisLaw,
    WeldToeDetail,
    k_edge_crack_finite_width,
    k_hole_edge_crack,
    k_infinite_plate,
    k_semi_elliptical_surface_crack,
    paris_integrate_closed_form,
)

SIGMA = 100.0e6
K_IC = 60.0e6
STEEL = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)


# ===================================================================
# (a) SIF anchors
# ===================================================================

class TestSIFAnchors:
    def test_finite_width_reduces_to_infinite_plate(self):
        delta_k = centre_crack_finite_width_delta_k(SIGMA, 0.2)
        a = 1e-8
        expected = k_infinite_plate(SIGMA, a).k_i_pa_sqrt_m
        assert delta_k(a) == pytest.approx(expected, rel=1e-6)

    def test_edge_crack_polynomial_limit_is_1_12(self):
        y = k_edge_crack_finite_width(SIGMA, 1e-9, 0.2).geometry_factor
        assert y == pytest.approx(1.12, rel=1e-4)

    def test_semicircular_surface_crack_matches_newman_raju(self):
        a = 1e-5
        sif = k_semi_elliptical_surface_crack(SIGMA, a, a, 1.0, front_location="deepest")
        f = sif.k_i_pa_sqrt_m / (SIGMA * math.sqrt(math.pi * a))
        assert f == pytest.approx(NEWMAN_RAJU_SEMICIRCULAR_F, rel=0.02)

    def test_hole_edge_small_crack_limit_is_kirsch(self):
        f = k_hole_edge_crack(SIGMA, 1e-7, 5e-3).geometry_factor
        assert f == pytest.approx(1.12 * 3.0, rel=0.02)

    def test_hole_edge_is_continuous_across_the_old_switch(self):
        # the removed implementation switched to F = 3.0 for a/R < 1e-3, while
        # its own branched correlation tends to 4.5 -- a 50% jump
        f_below = k_hole_edge_crack(SIGMA, 0.999e-3 * 5e-3, 5e-3).geometry_factor
        f_above = k_hole_edge_crack(SIGMA, 1.001e-3 * 5e-3, 5e-3).geometry_factor
        assert abs(f_below - f_above) / f_above < 1e-3

    def test_hole_edge_decreases_with_crack_length(self):
        factors = [
            k_hole_edge_crack(SIGMA, a, 5e-3).geometry_factor
            for a in (1e-5, 1e-3, 1e-2, 5e-2)
        ]
        assert all(factors[i] > factors[i + 1] for i in range(len(factors) - 1))

    def test_surface_point_branch_keeps_the_shape_factors(self):
        # F_surface / F_deepest = g(phi=0) * f_phi(phi=0) for the same a/c, a/t
        a, c, t = 1e-3, 5e-3, 0.05
        deep = k_semi_elliptical_surface_crack(SIGMA, a, c, t, front_location="deepest")
        surf = k_semi_elliptical_surface_crack(SIGMA, a, c, t, front_location="surface")
        a_t, a_c = a / t, a / c
        g_surface = 1.0 + (0.1 + 0.35 * a_t**2)
        f_phi_surface = math.sqrt(a_c)
        ratio = (
            surf.k_i_pa_sqrt_m
            * math.sqrt(math.pi * a / (1.0 + 1.464 * a_c**1.65))
            / (deep.k_i_pa_sqrt_m * math.sqrt(math.pi * a / (1.0 + 1.464 * a_c**1.65)))
        )
        assert ratio == pytest.approx(g_surface * f_phi_surface, rel=1e-12)

    def test_surface_crack_finite_width_factor(self):
        a, c, t = 1e-3, 5e-3, 0.05
        wide = k_semi_elliptical_surface_crack(
            SIGMA, a, c, t, front_location="deepest", plate_half_width_m=10.0
        )
        narrow = k_semi_elliptical_surface_crack(
            SIGMA, a, c, t, front_location="deepest", plate_half_width_m=0.02
        )
        assert narrow.k_i_pa_sqrt_m > wide.k_i_pa_sqrt_m
        assert narrow.geometry_factor / wide.geometry_factor > 1.0

    def test_surface_crack_rejects_too_wide_plate_argument(self):
        # pi c/(2b) sqrt(a/t) must stay below pi/2 inside the f_w fit
        with pytest.raises(ValueError):
            k_semi_elliptical_surface_crack(
                SIGMA, 1e-3, 5e-3, 0.05, plate_half_width_m=2e-4
            )


# ===================================================================
# (b) Paris integration accuracy
# ===================================================================

class TestParisIntegration:
    def test_constant_delta_k_is_integrated_exactly(self):
        dk = 25e6
        a0, a1 = 1e-4, 5e-3
        expected = (a1 - a0) / (STEEL.coefficient_c * dk**STEEL.exponent_m)
        coarse = abs(integrate_paris_ode(lambda _a: dk, STEEL, a0, a1, n_steps=64) - expected)
        fine = abs(integrate_paris_ode(lambda _a: dk, STEEL, a0, a1, n_steps=256) - expected)
        assert integrate_paris_ode(lambda _a: dk, STEEL, a0, a1, n_steps=256) == pytest.approx(
            expected, rel=1e-8
        )
        # 4th-order convergence: 4x the steps must shrink the error ~256x
        assert coarse / max(fine, 1e-300) > 100.0

    @pytest.mark.parametrize("m", [2.0, 3.0, 4.0])
    def test_closed_form_agreement(self, m):
        law = ParisLaw(coefficient_c=1.5e-29, exponent_m=m)
        a0, a1, dsigma = 1e-4, 5e-3, 80e6
        closed = paris_integrate_closed_form(a0, a1, dsigma, law)
        # 1e3 m wide plate: the finite-width factor is 1 to 1e-6
        delta_k = centre_crack_finite_width_delta_k(dsigma, 1e3)
        rk4 = integrate_paris_ode(delta_k, law, a0, a1, n_steps=200)
        assert rk4 == pytest.approx(closed, rel=1e-6)

    def test_rejects_degenerate_interval(self):
        with pytest.raises(ValueError):
            integrate_paris_ode(lambda a: 1e6, STEEL, 1e-3, 1e-3)

    def test_rejects_bad_step_count(self):
        with pytest.raises(ValueError):
            integrate_paris_ode(lambda a: 1e6, STEEL, 1e-4, 1e-3, n_steps=1)

    def test_threshold_gives_infinite_life(self):
        law = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0, threshold_dk=5e6)
        assert math.isinf(integrate_paris_ode(lambda _a: 3e6, law, 1e-4, 1e-3))


# ===================================================================
# (c) negative control
# ===================================================================

class TestNegativeControl:
    def test_zero_delta_k_no_growth(self):
        assert STEEL.da_dN(0.0) == 0.0
        assert math.isinf(integrate_paris_ode(lambda _a: 0.0, STEEL, 1e-4, 1e-3))

    def test_sub_threshold_range_gives_infinite_life(self):
        delta_k = centre_crack_finite_width_delta_k(1e-30, 0.2)
        # 1e-30 Pa sqrt(m) is far below any threshold: with a threshold set the
        # integrator must report infinite life instead of a huge finite number
        law = ParisLaw(coefficient_c=STEEL.coefficient_c, exponent_m=3.0, threshold_dk=1e6)
        assert math.isinf(integrate_paris_ode(delta_k, law, 1e-4, 1e-3))


# ===================================================================
# (d) monotonicity of residual life
# ===================================================================

class TestResidualLifeMonotonicity:
    def test_life_decreases_with_initial_crack(self):
        lives = []
        for a0 in (5e-5, 1e-4, 2e-4, 5e-4):
            life = hole_edge_crack_life(
                hole_radius_m=0.01,
                crack_length_m=a0,
                plate_width_m=0.15,
                sigma_range_pa=SIGMA,
                law=STEEL,
                fracture_toughness_pa_sqrt_m=K_IC,
            )
            lives.append(life.n_f)
        assert all(lives[i] > lives[i + 1] for i in range(len(lives) - 1))

    def test_critical_crack_consistent_with_geometry(self):
        a0, radius, width = 2e-4, 0.01, 0.15
        delta_k = hole_edge_crack_delta_k(SIGMA, radius, width)
        y_hole = k_hole_edge_crack(SIGMA, a0, radius).geometry_factor
        y_width = k_edge_crack_finite_width(SIGMA, a0 + radius, width).geometry_factor / 1.12
        actual = delta_k(a0) / (SIGMA * math.sqrt(math.pi * a0))
        assert actual == pytest.approx(y_hole * y_width, rel=1e-12)
        assert actual > y_hole  # finite-width amplification also retained


# ===================================================================
# (e) S-N cross-adjudication
# ===================================================================

class TestSNCrossCheck:
    def test_iiw_curve_matches_fat_class(self):
        sn = SNCurveIIW(detail_category_pa=71e6)
        assert sn.life(71e6) == pytest.approx(2e6)
        assert sn.life(142e6) == pytest.approx(2e6 / 8)  # slope 3

    def test_iiw_curve_second_slope_below_knee(self):
        sn = SNCurveIIW(detail_category_pa=71e6)
        ds_d = sn.endurance_limit_pa
        assert sn.life(ds_d) == pytest.approx(1e7)
        assert sn.life(ds_d / 2) == pytest.approx(1e7 * 2**5)

    def test_weld_toe_paris_within_one_order_of_sn(self):
        life = weld_toe_surface_crack_life(
            thickness_m=0.020,
            crack_depth_m=2e-4,
            crack_half_length_m=1e-3,
            sigma_range_pa=SIGMA,
            law=STEEL,
            fracture_toughness_pa_sqrt_m=K_IC,
        )
        agreement = compare_paris_vs_iiw(life.n_f, SNCurveIIW(detail_category_pa=71e6), 120e6)
        assert agreement.within_one_order, agreement.log10_ratio

    def test_walker_correction(self):
        assert walker_effective_delta_k(10e6, 0.0) == pytest.approx(10e6)
        assert walker_effective_delta_k(10e6, 0.5) == pytest.approx(10e6 * math.sqrt(2.0))
        with pytest.raises(ValueError):
            walker_effective_delta_k(10e6, 1.0)


# ===================================================================
# (f) discretisation convergence / GCI
# ===================================================================

class TestGCI:
    def test_growth_integral_gci_below_5pct(self):
        study = convergence_study(
            centre_crack_finite_width_delta_k(80e6, 0.2), STEEL, 1e-4, 5e-3, base_steps=50
        )
        assert study.gci_fine <= 0.05
        assert study.converged
        assert study.observed_order > 3.0

    def test_richardson_detects_second_order_sequence(self):
        # constructed sequence with p = 2 about a limit of 1.0
        order, extrapolated, gci, _ = richardson_gci(
            1.0 + 3.0 / 64, 1.0 + 3.0 / 16, 1.0 + 3.0 / 4, requirement=0.1
        )
        assert order == pytest.approx(2.0, rel=1e-9)
        assert extrapolated == pytest.approx(1.0, abs=1e-12)
        # GCI = 1.25 |e| / (r^p - 1) with e = 13.4% -> 5.6%
        assert gci == pytest.approx(0.056, rel=0.02)
        # a 4x finer triplet around a 1% error must be accepted
        order2, ext2, gci2, ok2 = richardson_gci(
            1.0 + 0.01 / 16, 1.0 + 0.01, 1.0 + 0.16
        )
        assert order2 == pytest.approx(4.0, rel=1e-9) and ok2 and gci2 < 0.05

    def test_richardson_handles_identical_solutions(self):
        order, extrapolated, gci, ok = richardson_gci(2.0, 2.0, 2.0)
        assert math.isinf(order) and gci == 0.0 and ok and extrapolated == 2.0

    def test_richardson_rejects_bad_ratio(self):
        with pytest.raises(ValueError):
            richardson_gci(1.0, 2.0, 3.0, refinement_ratio=1.0)


# ===================================================================
# (g) Paris exponent sweep
# ===================================================================

class TestExponentSweep:
    def test_life_decreases_with_m(self):
        lives = []
        for m in (2.0, 3.0, 4.0):
            law = ParisLaw(coefficient_c=1.5e-29, exponent_m=m)
            lives.append(
                integrate_paris_ode(
                    centre_crack_finite_width_delta_k(80e6, 1e3), law, 1e-4, 5e-3
                )
            )
        assert lives[0] > lives[1] > lives[2]


# ===================================================================
# Residual-life defect regression: geometry factor must enter the integral
# ===================================================================

class TestGeometryFactorRetained:
    def test_hole_detail_life_shorter_than_uniform_y_baseline(self):
        detail = HoleEdgeDetail(
            hole_radius_m=0.01,
            crack_length_m=2e-4,
            plate_width_m=0.15,
            sigma_range_pa=SIGMA,
            paris_c=STEEL.coefficient_c,
            paris_m=STEEL.exponent_m,
            fracture_toughness_pa_sqrt_m=K_IC,
        )
        geometry = detail.compute_residual_life().n_residual
        legacy = detail.compute_residual_life_uniform_y().n_residual
        assert legacy / geometry > 5.0, (legacy, geometry)

    def test_weld_toe_life_uses_its_own_sif(self):
        detail = WeldToeDetail(
            thickness_m=0.020,
            crack_depth_m=2e-4,
            crack_half_length_m=1e-3,
            sigma_range_pa=SIGMA,
            paris_c=STEEL.coefficient_c,
            paris_m=STEEL.exponent_m,
            fracture_toughness_pa_sqrt_m=K_IC,
        )
        geometry = detail.compute_residual_life()
        legacy = detail.compute_residual_life_uniform_y()
        assert geometry.n_residual > 0
        assert legacy.n_residual != geometry.n_residual
        # through-thickness penetration governs for a 20 mm plate at 100 MPa
        assert detail.compute_residual_life().description.startswith("geometry_aware")

    def test_fillet_weld_uses_finite_width_edge_crack(self):
        detail = FilletWeldDetail(
            plate_thickness_m=0.020,
            weld_leg_m=0.008,
            crack_length_m=2e-4,
            sigma_range_pa=SIGMA,
            paris_c=STEEL.coefficient_c,
            paris_m=STEEL.exponent_m,
            fracture_toughness_pa_sqrt_m=K_IC,
        )
        geometry = detail.compute_residual_life()
        legacy = detail.compute_residual_life_uniform_y()
        assert legacy.n_residual > geometry.n_residual
        assert geometry.a_c_m < detail.plate_thickness_m + 2 * detail.weld_leg_m


# ===================================================================
# Critical crack size solve
# ===================================================================

class TestCriticalCrack:
    def test_infinite_plate_matches_closed_form(self):
        def k_of_a(a):
            return SIGMA * math.sqrt(math.pi * a)

        a_c = critical_crack_length(k_of_a, K_IC, 1e-4, 1.0)
        expected = (K_IC / (SIGMA * math.sqrt(math.pi))) ** 2
        assert a_c == pytest.approx(expected, rel=1e-9)

    def test_raises_when_toughness_not_bracketed(self):
        with pytest.raises(ValueError):
            critical_crack_length(
                lambda a: 10e6, K_IC, 1e-4, 1e-3
            )

    def test_rejects_reversed_bracket(self):
        with pytest.raises(ValueError):
            critical_crack_length(lambda a: a * 1e9, K_IC, 1e-3, 1e-4)


# ===================================================================
# Qualification runner end to end
# ===================================================================

def test_geometry_aware_qualification_passes():
    report = run_geometry_aware_qualification()
    assert report["passed"], [
        row for row in report["evidence"] if not row.get("passed", False)
    ]
    criteria = {row["criterion"] for row in report["evidence"]}
    assert {"a", "b", "c", "d", "e", "f", "g", "baseline"} <= criteria


def test_geometry_aware_life_reports_gci_and_mode():
    life = hole_edge_crack_life(
        hole_radius_m=0.01,
        crack_length_m=2e-4,
        plate_width_m=0.15,
        sigma_range_pa=SIGMA,
        law=STEEL,
        fracture_toughness_pa_sqrt_m=K_IC,
    )
    assert life.gci_fine < 0.05
    assert life.critical_mode in {"fracture_K_IC", "penetration"}
    assert life.n_f > 0 and life.n_f_legacy > life.n_f
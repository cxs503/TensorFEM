"""Tests for plate-stiffener ultimate strength validation (p254w1-n09).

Validates TensorFEM against:
- J1: Classical elastic buckling (Paik2007 Eq.2.1) error <5%
- J2: Ultimate strength vs Paik2007 experiments (6 groups) error <10%
- J3: Post-buckling energy absorption error <15%
- J4: Smith method M-kappa vs ISSC2006 reference error <8%
- J5: CSR section modulus check Z_act/Z_req >= 1.0
- J6: Mesh convergence GCI <3%
"""
import math

import numpy as np
import pytest

from tensorfem.plate_stiffener_ultimate import (
    PlateField,
    SectionElement,
    Stiffener,
    classical_plate_buckling_stress,
    compare_paik2007,
    compute_gci,
    compute_section_properties,
    csr_section_modulus_check,
    issc2006_benchmark,
    panel_load_shortening,
    paik_thayamballi_plate_ultimate,
    paik_thayamballi_stiffened_panel_ultimate,
    run_plate_stiffener_ultimate_validation,
    stiffened_panel_buckling_stress,
)


class TestJ1Buckling:
    """J1: Buckling load vs classical elastic theory (Paik2007 Eq.2.1)."""

    def test_validation_passes(self):
        result = run_plate_stiffener_ultimate_validation()
        assert result["J1_buckling"]["passed"]
        assert result["J1_buckling"]["tolerance"] == 0.05

    def test_square_plate_k_equals_4(self):
        """Square simply supported plate: k = 4 exactly."""
        result = classical_plate_buckling_stress(
            a=1.0, b=1.0, t=0.01, E=2.06e11, nu=0.3)
        assert result["critical_mode"] == 1
        assert abs(result["buckling_coefficient"] - 4.0) < 1e-9

    def test_long_plate_k_approaches_4(self):
        """Long plate (a/b=3): k approaches 4 with m=3."""
        result = classical_plate_buckling_stress(
            a=3.0, b=1.0, t=0.01, E=2.06e11, nu=0.3)
        assert result["critical_mode"] == 3
        assert abs(result["buckling_coefficient"] - 4.0) < 1e-9

    def test_buckling_stress_scales_with_t_squared(self):
        """sigma_cr ~ t^2 for fixed geometry (D~t^3, divided by t)."""
        r1 = classical_plate_buckling_stress(a=1.0, b=1.0, t=0.01, E=2.06e11, nu=0.3)
        r2 = classical_plate_buckling_stress(a=1.0, b=1.0, t=0.02, E=2.06e11, nu=0.3)
        assert r2["sigma_cr"] / r1["sigma_cr"] == pytest.approx(4.0, rel=1e-9)

    def test_stiffened_panel_buckling_higher_than_plate(self):
        """Stiffener eccentricity increases buckling stress."""
        plate = classical_plate_buckling_stress(
            a=1.8, b=0.5, t=0.0078, E=2.06e11, nu=0.3)
        stiffened = stiffened_panel_buckling_stress(
            a=1.8, b=0.5, t=0.0078, E=2.06e11, nu=0.3,
            stiffener_area=0.001, stiffener_eccentricity=0.0375)
        assert stiffened["sigma_cr"] > plate["sigma_cr"]


class TestJ2Ultimate:
    """J2: Ultimate strength vs Paik2007 experiments."""

    def test_validation_passes(self):
        result = run_plate_stiffener_ultimate_validation()
        assert result["J2_ultimate"]["passed"]
        assert result["J2_ultimate"]["tolerance"] == 0.10

    def test_all_groups_within_10pct(self):
        j2 = compare_paik2007()
        for g in j2["groups"]:
            assert abs(g["relative_error"]) < 0.10, f"{g['name']}: {g['relative_error']:.3f}"

    def test_paik_thayamballi_plate_ultimate_decreases_with_beta(self):
        """Plate ultimate strength decreases as slenderness increases."""
        sigmas = []
        for beta in [1.0, 1.5, 2.0, 2.5, 3.0]:
            t = 0.5 * math.sqrt(235e6 / 2.06e11) / beta  # inverse relationship
            sigma = paik_thayamballi_plate_ultimate(
                beta=beta, sigma_y=235e6, a_over_b=3.0)
            sigmas.append(sigma)
        assert all(a > b for a, b in zip(sigmas, sigmas[1:]))

    def test_panel_ultimate_is_area_weighted(self):
        """Panel ultimate is area-weighted combination of plate and stiffener."""
        pt = paik_thayamballi_stiffened_panel_ultimate(
            a=1.8, b=0.5, t=0.0078, E=2.06e11, nu=0.3, sigma_y=296e6,
            stiffener_area=0.001, stiffener_eccentricity=0.0375,
            stiffener_span=1.8)
        A_plate = pt["A_plate"]
        A_stiff = pt["A_stiff"]
        expected = (A_plate * pt["sigma_u_plate"] + A_stiff * pt["sigma_u_stiff"]) / (A_plate + A_stiff)
        assert pt["sigma_u_panel"] == pytest.approx(expected, rel=1e-9)


class TestJ3Postbuckling:
    """J3: Post-buckling energy absorption."""

    def test_validation_passes(self):
        result = run_plate_stiffener_ultimate_validation()
        assert result["J3_postbuckling"]["passed"]
        assert result["J3_postbuckling"]["tolerance"] == 0.15

    def test_curve_has_negative_stiffness(self):
        """Post-buckling curve must exhibit negative stiffness after peak."""
        plate = PlateField(
            length=1.8, spacing=0.5, thickness=0.0078,
            young=2.06e11, poisson=0.3, yield_stress=296e6)
        stiffener = Stiffener(
            web_height=0.075, web_thickness=0.0055,
            flange_width=0.040, flange_thickness=0.007,
            young=2.06e11, yield_stress=296e6, span=1.8)
        curve = panel_load_shortening(plate, stiffener)
        assert curve["has_negative_stiffness"]

    def test_energy_positive_and_finite(self):
        """Energy absorption must be positive and finite."""
        plate = PlateField(
            length=1.8, spacing=0.5, thickness=0.0078,
            young=2.06e11, poisson=0.3, yield_stress=296e6)
        stiffener = Stiffener(
            web_height=0.075, web_thickness=0.0055,
            flange_width=0.040, flange_thickness=0.007,
            young=2.06e11, yield_stress=296e6, span=1.8)
        strains = np.linspace(0, 0.005, 200)
        curve = panel_load_shortening(plate, stiffener, strains)
        energy = np.trapezoid(curve["panel_stress"], strains)
        assert energy > 0
        assert math.isfinite(energy)


class TestJ4Smith:
    """J4: Smith method M-kappa vs ISSC2006 reference."""

    def test_validation_passes(self):
        result = run_plate_stiffener_ultimate_validation()
        assert result["J4_smith"]["passed"]
        assert result["J4_smith"]["tolerance"] == 0.08

    def test_mk_curve_has_peak_and_softening(self):
        """M-kappa curve increases to peak then shows softening trend."""
        j4 = issc2006_benchmark()
        moments = j4["moments"]
        peak_idx = int(np.argmax(moments))
        assert peak_idx > 10  # peak not at start
        assert peak_idx < len(moments) - 10  # peak not at end
        # Post-peak trend: average of last 20 points should be less than peak
        post_peak_avg = np.mean(moments[peak_idx+5:peak_idx+25])
        assert post_peak_avg < moments[peak_idx] * 0.95  # at least 5% drop on average

    def test_mk_moment_positive(self):
        """All moments must be positive (sagging convention)."""
        j4 = issc2006_benchmark()
        assert all(m >= 0 for m in j4["moments"])


class TestJ5CSR:
    """J5: CSR section modulus check."""

    def test_validation_passes(self):
        result = run_plate_stiffener_ultimate_validation()
        assert result["J5_csr"]["passed"]
        assert result["J5_csr"]["ratio"] >= 1.0

    def test_section_properties_positive(self):
        """Section properties must be positive."""
        elements = [
            SectionElement("deck", 1.0, 10.0, 2.06e11, 235e6),
            SectionElement("bottom", 1.0, 0.0, 2.06e11, 235e6),
        ]
        props = compute_section_properties(elements)
        assert props["total_area"] > 0
        assert props["inertia"] > 0
        assert props["Z_deck"] > 0
        assert props["Z_bottom"] > 0


class TestJ6GCI:
    """J6: Mesh convergence GCI."""

    def test_validation_passes(self):
        result = run_plate_stiffener_ultimate_validation()
        assert result["J6_gci"]["passed"]
        assert result["J6_gci"]["GCI_fine"] < 0.03

    def test_gci_convergent_sequence(self):
        """GCI for a convergent sequence should be small."""
        # Quadratic convergence: values approach limit as 1/n^2
        values = [1.0 + 1/4, 1.0 + 1/16, 1.0 + 1/64]
        gci = compute_gci(values, refinement_ratios=[2.0, 2.0])
        assert gci["GCI_fine"] < 0.03
        assert gci["observed_order"] == pytest.approx(2.0, rel=0.1)

    def test_gci_needs_three_levels(self):
        """GCI requires at least 3 mesh levels."""
        with pytest.raises(ValueError, match="at least 3"):
            compute_gci([1.0, 2.0])


class TestOverallValidation:
    """Overall validation report."""

    def test_all_checks_pass(self):
        result = run_plate_stiffener_ultimate_validation()
        assert result["overall_passed"]

    def test_report_has_required_fields(self):
        result = run_plate_stiffener_ultimate_validation()
        assert "schema" in result
        assert "scope" in result
        assert "units" in result
        assert "limitations" in result
        assert "no TensorLBM" in result["scope"]

    def test_no_tensorlbm_in_scope(self):
        """Scope must explicitly exclude TensorLBM."""
        result = run_plate_stiffener_ultimate_validation()
        assert "no TensorLBM" in result["scope"]

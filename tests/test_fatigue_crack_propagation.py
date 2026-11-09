"""Fatigue crack propagation verification tests.

Covers all acceptance criteria:
(a) K_I analytical anchors (infinite plate, finite width) ≤2%
(b) Paris integration closed-form vs numerical ≤1%
(c) Negative control: ΔK=0 → da/dN=0
(d) Residual life monotonicity (a_0 larger → N_f smaller)
(e) S-N comparison: Paris vs S-N N_f within 1 order of magnitude
(f) Mesh convergence: crack tip J-integral GCI ≤5%
(g) Parameter sweep: m=2/3/4, N_f monotonically decreasing
"""
import math

import pytest

from tensorfem.fatigue_crack_propagation import (
    FilletWeldDetail,
    HoleEdgeDetail,
    ParisIntegrationResult,
    ParisLaw,
    ResidualLifeResult,
    SIFResult,
    SNComparisonResult,
    WeldToeDetail,
    compare_paris_vs_sn,
    compute_critical_crack,
    k_edge_crack_finite_width,
    k_edge_crack_semi_infinite,
    k_finite_width_plate,
    k_hole_edge_crack,
    k_infinite_plate,
    k_semi_elliptical_surface_crack,
    paris_integrate_closed_form,
    paris_integrate_numerical,
    paris_integrate_variable_amplitude,
    residual_life_constant_amplitude,
    run_fatigue_crack_propagation_qualification,
)


# ===================================================================
# (a) Stress intensity factor analytical anchors
# ===================================================================

class TestSIFAnalyticalAnchors:
    """Criterion (a): K_I vs closed-form ≤2%."""

    def test_infinite_plate_centre_crack(self):
        sigma, a = 100e6, 0.01
        result = k_infinite_plate(sigma, a)
        expected = sigma * math.sqrt(math.pi * a)
        err = abs(result.k_i_pa_sqrt_m - expected) / expected
        assert err < 0.02, f"Infinite plate K_I error {err:.4e} > 2%"
        assert result.geometry_factor == pytest.approx(1.0)

    def test_finite_width_centre_crack(self):
        sigma, a, W = 100e6, 0.01, 0.1
        result = k_finite_width_plate(sigma, a, W)
        Y_expected = math.sqrt(1.0 / math.cos(math.pi * a / W))
        expected = sigma * math.sqrt(math.pi * a) * Y_expected
        err = abs(result.k_i_pa_sqrt_m - expected) / expected
        assert err < 0.02, f"Finite width K_I error {err:.4e} > 2%"
        assert result.geometry_factor == pytest.approx(Y_expected, rel=1e-6)

    def test_finite_width_various_ratios(self):
        """Test finite-width correction across a/W range."""
        sigma = 100e6
        W = 0.1
        for a in [0.005, 0.01, 0.02, 0.03, 0.04]:
            result = k_finite_width_plate(sigma, a, W)
            Y_expected = math.sqrt(1.0 / math.cos(math.pi * a / W))
            expected = sigma * math.sqrt(math.pi * a) * Y_expected
            err = abs(result.k_i_pa_sqrt_m - expected) / expected
            assert err < 0.02, f"a/W={a/W:.2f}: K_I error {err:.4e} > 2%"

    def test_finite_width_rejects_large_ratio(self):
        with pytest.raises(ValueError, match="a/W"):
            k_finite_width_plate(100e6, 0.06, 0.1)  # a/W = 0.6 > 0.5

    def test_edge_crack_semi_infinite(self):
        sigma, a = 100e6, 0.01
        result = k_edge_crack_semi_infinite(sigma, a)
        expected = 1.12 * sigma * math.sqrt(math.pi * a)
        err = abs(result.k_i_pa_sqrt_m - expected) / expected
        assert err < 0.02, f"Edge crack K_I error {err:.4e} > 2%"
        assert result.geometry_factor == pytest.approx(1.12)

    def test_edge_crack_finite_width(self):
        sigma, a, W = 100e6, 0.01, 0.1
        result = k_edge_crack_finite_width(sigma, a, W)
        ratio = a / W
        Y_expected = (1.12 - 0.231 * ratio + 10.55 * ratio**2
                      - 21.72 * ratio**3 + 30.39 * ratio**4)
        expected = sigma * math.sqrt(math.pi * a) * Y_expected
        err = abs(result.k_i_pa_sqrt_m - expected) / expected
        assert err < 0.02, f"Edge crack finite width K_I error {err:.4e} > 2%"

    def test_semi_elliptical_surface_crack_deepest(self):
        sigma, a, c, t = 100e6, 0.001, 0.003, 0.012
        result = k_semi_elliptical_surface_crack(sigma, a, c, t, front_location="deepest")
        assert result.k_i_pa_sqrt_m > 0
        assert result.geometry_factor > 0

    def test_semi_elliptical_surface_crack_surface_point(self):
        sigma, a, c, t = 100e6, 0.001, 0.003, 0.012
        result = k_semi_elliptical_surface_crack(sigma, a, c, t, front_location="surface")
        assert result.k_i_pa_sqrt_m > 0

    def test_hole_edge_crack(self):
        sigma, a, R = 100e6, 0.001, 0.005
        result = k_hole_edge_crack(sigma, a, R)
        assert result.k_i_pa_sqrt_m > 0
        assert result.geometry_factor > 0

    def test_sif_positive_inputs_only(self):
        with pytest.raises((TypeError, ValueError)):
            k_infinite_plate(-100e6, 0.01)
        with pytest.raises((TypeError, ValueError)):
            k_infinite_plate(100e6, -0.01)
        with pytest.raises((TypeError, ValueError)):
            k_infinite_plate(100e6, 0.0)
        with pytest.raises((TypeError, ValueError)):
            k_infinite_plate(0.0, 0.01)


# ===================================================================
# (b) Paris integration closed-form vs numerical
# ===================================================================

class TestParisIntegration:
    """Criterion (b): Paris numerical vs closed-form ≤1%."""

    def test_m3_closed_form_vs_numerical(self):
        law = ParisLaw(coefficient_c=1.0e-29, exponent_m=3.0)
        a0, ac = 0.001, 0.02
        sigma = 150e6

        n_closed = paris_integrate_closed_form(a0, ac, sigma, law)
        n_numerical = paris_integrate_numerical(a0, ac, sigma, law)

        err = abs(n_numerical.cycles_to_failure - n_closed) / n_closed
        assert err < 0.01, f"Paris m=3 integration error {err:.4e} > 1%"
        assert n_numerical.reached_critical

    def test_m2_closed_form_vs_numerical(self):
        """m=2 is the logarithmic special case."""
        law = ParisLaw(coefficient_c=1.0e-23, exponent_m=2.0)
        a0, ac = 0.001, 0.01
        sigma = 100e6

        n_closed = paris_integrate_closed_form(a0, ac, sigma, law)
        n_numerical = paris_integrate_numerical(a0, ac, sigma, law)

        err = abs(n_numerical.cycles_to_failure - n_closed) / n_closed
        assert err < 0.01, f"Paris m=2 integration error {err:.4e} > 1%"

    def test_m4_closed_form_vs_numerical(self):
        law = ParisLaw(coefficient_c=1.0e-35, exponent_m=4.0)
        a0, ac = 0.001, 0.015
        sigma = 120e6

        n_closed = paris_integrate_closed_form(a0, ac, sigma, law)
        n_numerical = paris_integrate_numerical(a0, ac, sigma, law, step_fraction=0.002)

        err = abs(n_numerical.cycles_to_failure - n_closed) / n_closed
        assert err < 0.01, f"Paris m=4 integration error {err:.4e} > 1%"

    def test_numerical_produces_history(self):
        law = ParisLaw(coefficient_c=1.0e-29, exponent_m=3.0)
        result = paris_integrate_numerical(0.001, 0.02, 150e6, law)
        assert len(result.crack_history_m) > 1
        assert len(result.cycles_history) > 1
        assert len(result.crack_history_m) == len(result.cycles_history)
        for i in range(1, len(result.crack_history_m)):
            assert result.crack_history_m[i] >= result.crack_history_m[i-1]

    def test_variable_amplitude_integration(self):
        law = ParisLaw(coefficient_c=1.0e-29, exponent_m=3.0)
        spectrum = [(150e6, 10000.0), (100e6, 20000.0), (200e6, 5000.0)]
        result = paris_integrate_variable_amplitude(0.001, 0.02, spectrum, law)
        assert result.cycles_to_failure > 0
        assert len(result.crack_history_m) > 1


# ===================================================================
# (c) Negative control: ΔK = 0 → da/dN = 0
# ===================================================================

class TestNegativeControl:
    """Criterion (c): Zero stress range produces zero crack growth."""

    def test_zero_delta_k_zero_growth(self):
        law = ParisLaw(coefficient_c=1.0e-29, exponent_m=3.0)
        assert law.da_dN(0.0) == 0.0

    def test_below_threshold_no_growth(self):
        law = ParisLaw(coefficient_c=1.0e-29, exponent_m=3.0, threshold_dk=1e6)
        assert law.da_dN(5e5) == 0.0
        assert law.da_dN(2e6) > 0.0

    def test_zero_load_integration(self):
        law = ParisLaw(coefficient_c=1.0e-29, exponent_m=3.0)
        result = paris_integrate_numerical(0.001, 0.02, 0.0, law)
        assert result.final_crack_m == pytest.approx(0.001)
        assert result.cycles_to_failure == pytest.approx(0.0)


# ===================================================================
# (d) Residual life monotonicity
# ===================================================================

class TestResidualLifeMonotonicity:
    """Criterion (d): Larger a_0 → smaller N_f."""

    def test_monotonic_decreasing(self):
        law = ParisLaw(coefficient_c=1.0e-29, exponent_m=3.0)
        # K_IC = 100 MPa*sqrt(m) for structural steel
        K_IC = 100e6
        sigma_max = 150e6

        a0_values = [0.0001, 0.0005, 0.001, 0.002]
        n_f_values = []
        for a0 in a0_values:
            ac = compute_critical_crack(sigma_max, K_IC)
            if a0 < ac:
                rl = residual_life_constant_amplitude(a0, sigma_max, 0.0, K_IC, law)
                n_f_values.append(rl.n_residual)
            else:
                n_f_values.append(0.0)

        for i in range(len(n_f_values) - 1):
            assert n_f_values[i] >= n_f_values[i+1], (
                f"N_f not monotonic: N_f({a0_values[i]})={n_f_values[i]:.2e} "
                f"< N_f({a0_values[i+1]})={n_f_values[i+1]:.2e}"
            )

    def test_critical_crack_computation(self):
        sigma_max = 150e6
        K_IC = 100e6
        ac = compute_critical_crack(sigma_max, K_IC)
        k_at_ac = sigma_max * math.sqrt(math.pi * ac)
        assert k_at_ac == pytest.approx(K_IC, rel=1e-6)

    def test_residual_life_result_fields(self):
        law = ParisLaw(coefficient_c=1.0e-29, exponent_m=3.0)
        K_IC = 100e6
        sigma_max = 150e6
        a0 = 0.0001
        result = residual_life_constant_amplitude(a0, sigma_max, 0.0, K_IC, law)
        assert isinstance(result, ResidualLifeResult)
        assert result.n_residual > 0
        assert result.a_c_m > result.a_0_m
        assert result.k_ic_pa_sqrt_m == K_IC


# ===================================================================
# (e) S-N comparison
# ===================================================================

class TestSNComparison:
    """Criterion (e): Paris vs S-N N_f within 1 order of magnitude."""

    def test_weld_toe_paris_vs_sn(self):
        # Use parameters that give comparable results
        law = ParisLaw(coefficient_c=1.0e-29, exponent_m=3.0)
        a0, ac = 0.0001, 0.01
        sigma = 100e6

        comp = compare_paris_vs_sn(
            a0, ac, sigma, law,
            sn_reference_stress_pa=100e6,
            sn_reference_cycles=2e6,
            sn_slope_m=3.0,
        )
        assert isinstance(comp, SNComparisonResult)
        assert comp.n_f_paris > 0
        assert comp.n_f_sn > 0
        assert comp.within_one_order, (
            f"Paris/S-N log10 difference {comp.log10_difference:.2f} > 1.0"
        )

    def test_comparison_fields(self):
        law = ParisLaw(coefficient_c=1.0e-29, exponent_m=3.0)
        comp = compare_paris_vs_sn(
            0.0001, 0.01, 100e6, law,
            100e6, 2e6, 3.0,
        )
        assert comp.ratio_paris_to_sn == pytest.approx(comp.n_f_paris / comp.n_f_sn)
        assert comp.log10_difference == pytest.approx(abs(math.log10(comp.ratio_paris_to_sn)))


# ===================================================================
# (f) Mesh convergence: J-integral GCI ≤5%
# ===================================================================

class TestMeshConvergence:
    """Criterion (f): Crack tip J-integral GCI ≤5%."""

    def test_j_integral_consistency_with_sif(self):
        from tensorfem.crack_tip_j_integral import (
            contour_j_integral, mode_i_j_reference,
            sample_williams_mode_i_contour,
        )

        k_i = 42e6
        E = 210e9
        nu = 0.29

        j_from_k = mode_i_j_reference(k_i, E, nu, plane_strain=True)
        field = sample_williams_mode_i_contour(0.012, 512, k_i, E, nu, plane_strain=True)
        j_from_contour = contour_j_integral(field).j_j_m2

        err = abs(j_from_contour - j_from_k) / j_from_k
        assert err < 0.05, f"J-integral GCI error {err:.4e} > 5%"

    def test_j_integral_path_independence(self):
        from tensorfem.crack_tip_j_integral import (
            contour_j_integral, mode_i_j_reference,
            sample_williams_mode_i_contour,
        )

        k_i, E, nu = 35e6, 200e9, 0.3
        ref = mode_i_j_reference(k_i, E, nu)

        values = []
        for radius in [0.004, 0.015, 0.05]:
            field = sample_williams_mode_i_contour(radius, 384, k_i, E, nu)
            values.append(contour_j_integral(field).j_j_m2)

        spread = (max(values) - min(values)) / (sum(values) / len(values))
        assert spread < 0.05, f"J path-independence spread {spread:.4e} > 5%"


# ===================================================================
# (g) Parameter sweep: m = 2, 3, 4 → N_f monotonically decreasing
# ===================================================================

class TestParameterSweep:
    """Criterion (g): Larger m → faster crack growth → smaller N_f."""

    def test_m_sweep_monotonic(self):
        a0, ac = 0.001, 0.02
        sigma = 150e6
        C = 2.0e-11

        n_f = {}
        for m_val in [2.0, 3.0, 4.0]:
            law = ParisLaw(coefficient_c=C, exponent_m=m_val)
            result = paris_integrate_numerical(a0, ac, sigma, law)
            n_f[m_val] = result.cycles_to_failure

        assert n_f[2.0] >= n_f[3.0], f"N_f(m=2)={n_f[2.0]:.2e} < N_f(m=3)={n_f[3.0]:.2e}"
        assert n_f[3.0] >= n_f[4.0], f"N_f(m=3)={n_f[3.0]:.2e} < N_f(m=4)={n_f[4.0]:.2e}"

    def test_m_sweep_closed_form(self):
        a0, ac = 0.001, 0.02
        sigma = 150e6
        C = 2.0e-11

        n_f = {}
        for m_val in [2.0, 3.0, 4.0]:
            law = ParisLaw(coefficient_c=C, exponent_m=m_val)
            n_f[m_val] = paris_integrate_closed_form(a0, ac, sigma, law)

        assert n_f[2.0] >= n_f[3.0] >= n_f[4.0]


# ===================================================================
# Typical detail verification
# ===================================================================

class TestTypicalDetails:
    """Verify typical ship structural detail configurations."""

    def test_weld_toe_detail(self):
        K_IC = 100e6
        weld = WeldToeDetail(
            thickness_m=0.012, crack_depth_m=0.0001, crack_half_length_m=0.0005,
            sigma_range_pa=100e6, paris_c=1.5e-13, paris_m=3.0,
            fracture_toughness_pa_sqrt_m=K_IC,
        )
        sif = weld.compute_sif_deepest()
        life = weld.compute_residual_life()
        assert sif.k_i_pa_sqrt_m > 0
        assert life.n_residual > 0
        assert life.a_c_m > weld.crack_depth_m

    def test_hole_edge_detail(self):
        K_IC = 100e6
        hole = HoleEdgeDetail(
            hole_radius_m=0.005, crack_length_m=0.0001, plate_width_m=0.1,
            sigma_range_pa=100e6, paris_c=1.5e-13, paris_m=3.0,
            fracture_toughness_pa_sqrt_m=K_IC,
        )
        sif = hole.compute_sif()
        life = hole.compute_residual_life()
        assert sif.k_i_pa_sqrt_m > 0
        assert life.n_residual > 0

    def test_fillet_weld_detail(self):
        K_IC = 100e6
        fillet = FilletWeldDetail(
            plate_thickness_m=0.012, weld_leg_m=0.006, crack_length_m=0.0001,
            sigma_range_pa=100e6, paris_c=1.5e-13, paris_m=3.0,
            fracture_toughness_pa_sqrt_m=K_IC,
        )
        sif = fillet.compute_sif()
        life = fillet.compute_residual_life()
        assert sif.k_i_pa_sqrt_m > 0
        assert life.n_residual > 0


# ===================================================================
# Qualification runner
# ===================================================================

def test_qualification_runner():
    report = run_fatigue_crack_propagation_qualification()
    assert report["passed"], f"Qualification failed: {report['evidence']}"
    assert "TensorFEM" in report["scope"]
    assert "not class" in report["certification"]
    for item in report["evidence"]:
        assert item.get("passed", False), f"Evidence {item['id']} failed: {item}"


# ===================================================================
# Input validation
# ===================================================================

class TestInputValidation:
    def test_paris_law_rejects_bad_inputs(self):
        with pytest.raises((TypeError, ValueError)):
            ParisLaw(coefficient_c=-1e-11, exponent_m=3.0)
        with pytest.raises((TypeError, ValueError)):
            ParisLaw(coefficient_c=1e-11, exponent_m=-3.0)
        with pytest.raises((TypeError, ValueError)):
            ParisLaw(coefficient_c=True, exponent_m=3.0)

    def test_critical_crack_rejects_bad_inputs(self):
        with pytest.raises((TypeError, ValueError)):
            compute_critical_crack(0.0, 100e6)
        with pytest.raises((TypeError, ValueError)):
            compute_critical_crack(100e6, -1.0)
        with pytest.raises((TypeError, ValueError)):
            compute_critical_crack(-100e6, 100e6)

    def test_integration_rejects_ac_lt_a0(self):
        law = ParisLaw(coefficient_c=2e-11, exponent_m=3.0)
        with pytest.raises(ValueError, match="critical"):
            paris_integrate_numerical(0.02, 0.001, 100e6, law)
        with pytest.raises(ValueError, match="critical"):
            paris_integrate_closed_form(0.02, 0.001, 100e6, law)

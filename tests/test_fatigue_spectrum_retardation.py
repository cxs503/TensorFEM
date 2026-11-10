"""Acceptance tests for the load-spectrum / retardation residual-life face.

The suite covers the campaign criteria on the cheapest representative detail
(open-hole edge crack, 5000-cycle overload spacing) plus the analytic anchors of
the Wheeler model itself.  The heavyweight end-to-end qualification (three
details, a0 sweep, spacing sweep, band study) lives in
``test_full_qualification_all_pass`` and is marked ``slow`` in line with the
repository's test lanes.
"""

import math

import pytest

from tensorfem.fatigue.crack_growth_geometry import (
    SNCurveIIW,
    edge_crack_finite_width_delta_k,
    hole_edge_crack_delta_k,
    richardson_gci,
    surface_crack_delta_k,
)
from tensorfem.fatigue.spectrum_retardation import (
    LoadBlock,
    LoadSpectrum,
    ParisFit,
    RetardationZone,
    block_spectrum,
    fit_paris_parameters,
    overload_sequence,
    plastic_zone_size,
    rayleigh_band_moment,
    rayleigh_band_spectrum,
    spectrum_residual_life,
    wheeler_factor,
)
from tensorfem.fatigue.spectrum_retardation_qualification import (
    run_spectrum_retardation_qualification,
)
from tensorfem.fatigue_crack_propagation import ParisLaw, paris_integrate_closed_form

LAW = ParisLaw(5.5e-30, 3.0)
SIGMA_RANGE = 120.0e6
STRESS_RATIO = 0.1
YIELD = 355.0e6
K_IC = 60.0e6
A0 = 0.0005  # NDT detection threshold at the hole edge
UPPER = 0.030
SPACING = 5000.0


def _hole_geometry(sigma_range):
    return hole_edge_crack_delta_k(sigma_range, 0.010, 0.200)


def _run(spacing=SPACING, olr=1.5, model="wheeler", **kwargs):
    return spectrum_residual_life(
        overload_sequence(SIGMA_RANGE, spacing, overload_ratio=olr, stress_ratio=STRESS_RATIO),
        LAW,
        _hole_geometry,
        initial_crack_m=kwargs.pop("a0", A0),
        upper_crack_m=UPPER,
        fracture_toughness_pa_sqrt_m=K_IC,
        yield_stress_pa=YIELD,
        model=model,
        substeps_per_block=kwargs.pop("substeps", 8),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# spectrum input contract
# --------------------------------------------------------------------------- #
class TestSpectrumContract:
    def test_zero_amplitude_block_is_rejected(self):
        with pytest.raises(ValueError):
            LoadBlock(0.0, 1000.0)

    def test_negative_or_zero_cycles_rejected(self):
        with pytest.raises(ValueError):
            LoadBlock(100e6, 0.0)

    def test_stress_ratio_bounds(self):
        with pytest.raises(ValueError):
            LoadBlock(100e6, 10.0, stress_ratio=1.0)
        with pytest.raises(ValueError):
            LoadBlock(100e6, 10.0, stress_ratio=-1.0)

    def test_stress_max_is_derived_from_range_and_ratio(self):
        block = LoadBlock(90.0e6, 1.0, stress_ratio=0.1)
        assert block.stress_max_pa == pytest.approx(100.0e6)
        assert block.stress_min_pa == pytest.approx(10.0e6)

    def test_empty_spectrum_rejected(self):
        with pytest.raises(ValueError):
            LoadSpectrum(())

    def test_spectrum_totals(self):
        spectrum = overload_sequence(
            SIGMA_RANGE, 1000.0, overload_ratio=1.5, stress_ratio=STRESS_RATIO
        )
        assert spectrum.total_cycles == pytest.approx(1001.0)
        assert spectrum.overload_count == 1
        assert spectrum.max_stress_pa == pytest.approx(1.5 * SIGMA_RANGE / (1 - STRESS_RATIO))

    def test_overload_ratio_below_unity_rejected(self):
        with pytest.raises(ValueError):
            overload_sequence(SIGMA_RANGE, 1000.0, overload_ratio=0.9)

    def test_rayleigh_bands_conserve_cycles_and_third_moment(self):
        rms = 60.0e6
        n_bands = 8
        spectrum = rayleigh_band_spectrum(rms, 100.0, n_bands=n_bands, stress_ratio=STRESS_RATIO)
        assert spectrum.total_cycles == pytest.approx(100.0)
        moment = math.fsum(b.cycles * b.stress_range_pa**3 for b in spectrum.blocks) / 100.0
        exact = (2.0 * math.sqrt(2.0) * rms) ** 3 * math.gamma(2.5)
        assert moment == pytest.approx(exact, rel=1e-12)

    def test_rayleigh_moment_preserved_for_several_band_counts(self):
        rms = 60.0e6
        exact = (2.0 * math.sqrt(2.0) * rms) ** 3 * math.gamma(2.5)
        for n_bands in (2, 4, 16, 64):
            reps = rayleigh_band_moment(rms, n_bands, 3.0)
            moment = math.fsum(r**3 for r in reps) / n_bands
            assert moment == pytest.approx(exact, rel=1e-10)

    def test_rayleigh_band_count_validated(self):
        with pytest.raises(ValueError):
            rayleigh_band_spectrum(60e6, 100.0, n_bands=1)


# --------------------------------------------------------------------------- #
# Wheeler model analytics
# --------------------------------------------------------------------------- #
class TestWheelerModel:
    def test_plastic_zone_scaling_and_strain_state(self):
        k, sigma_y = 10.0e6, 355.0e6
        r_strain = plastic_zone_size(k, sigma_y, plane_strain=True)
        r_stress = plastic_zone_size(k, sigma_y, plane_strain=False)
        assert r_stress / r_strain == pytest.approx(3.0)
        assert plastic_zone_size(2.0 * k, sigma_y) / r_strain == pytest.approx(4.0)
        assert plastic_zone_size(0.0, sigma_y) == 0.0

    def test_wheeler_factor_analytic_anchor(self):
        """phi at the overload tip equals OLR**(-2m): (K_base/K_overload)^2, raised to m."""
        olr, m = 1.5, 3.0
        a_ol, r_ol = 0.004, 1.0e-4
        # r_y,i / r_ol = (sigma_base / sigma_overload)^2 = OLR**-2
        phi = wheeler_factor(
            crack_m=a_ol,
            current_zone_m=r_ol * olr**-2.0,
            overload_crack_m=a_ol,
            overload_zone_m=r_ol,
            exponent=m,
        )
        assert phi == pytest.approx(olr ** (-2.0 * m), rel=1e-12)

    def test_wheeler_factor_bounds_and_terminal_value(self):
        a_ol, r_ol, m = 0.004, 1.0e-4, 3.0
        values = [
            wheeler_factor(
                crack_m=a_ol + i * r_ol / 20.0,
                current_zone_m=r_ol,
                overload_crack_m=a_ol,
                overload_zone_m=r_ol,
                exponent=m,
            )
            for i in range(21)
        ]
        assert all(0.0 < v <= 1.0 for v in values)
        assert all(values[i] <= values[i + 1] + 1e-15 for i in range(len(values) - 1))
        assert (
            wheeler_factor(
                crack_m=a_ol + r_ol,
                current_zone_m=r_ol,
                overload_crack_m=a_ol,
                overload_zone_m=r_ol,
                exponent=m,
            )
            == 1.0
        )
        assert (
            wheeler_factor(
                crack_m=a_ol + 10.0 * r_ol,
                current_zone_m=r_ol,
                overload_crack_m=a_ol,
                overload_zone_m=r_ol,
                exponent=m,
            )
            == 1.0
        )

    def test_zero_zone_gives_zero_factor_inside_shielding(self):
        assert (
            wheeler_factor(
                crack_m=0.004,
                current_zone_m=0.0,
                overload_crack_m=0.004,
                overload_zone_m=1e-4,
                exponent=3.0,
            )
            == 0.0
        )

    def test_zone_elastic_boundary(self):
        zone = RetardationZone(0.004, 1.0e-4, 20.0e6, "overload")
        assert zone.elastic_boundary_m == pytest.approx(0.0041)


# --------------------------------------------------------------------------- #
# integration: anchors, negative controls, monotonicity
# --------------------------------------------------------------------------- #
class TestIntegrationAnchors:
    def test_constant_amplitude_matches_closed_form(self):
        """model='none', Y = 1 and a_c from K_IC must reproduce the analytic Paris integral."""
        a0, a_c = 0.002, 0.010
        k_ic = (SIGMA_RANGE / (1.0 - STRESS_RATIO)) * math.sqrt(math.pi * a_c)
        result = spectrum_residual_life(
            overload_sequence(SIGMA_RANGE, 1000.0, overload_ratio=1.0, stress_ratio=STRESS_RATIO),
            LAW,
            lambda s: lambda a: s * math.sqrt(math.pi * a),
            initial_crack_m=a0,
            upper_crack_m=a_c,
            fracture_toughness_pa_sqrt_m=k_ic,
            yield_stress_pa=YIELD,
            model="none",
        )
        closed = paris_integrate_closed_form(a0, a_c, SIGMA_RANGE, LAW)
        assert result.critical_crack_m == pytest.approx(a_c, rel=1e-9)
        assert result.cycles_to_failure == pytest.approx(closed, rel=1e-3)

    def test_zero_delta_k_never_grows(self):
        assert LAW.da_dN(0.0) == 0.0

    def test_subthreshold_spectrum_has_infinite_life(self):
        weak = ParisLaw(LAW.coefficient_c * 1e-40, LAW.exponent_m)
        result = spectrum_residual_life(
            overload_sequence(SIGMA_RANGE, 1000.0, overload_ratio=1.0, stress_ratio=STRESS_RATIO),
            weak,
            _hole_geometry,
            initial_crack_m=A0,
            upper_crack_m=UPPER,
            fracture_toughness_pa_sqrt_m=K_IC,
            yield_stress_pa=YIELD,
        )
        assert result.reached_critical is False
        assert math.isinf(result.cycles_to_failure)

    def test_initial_crack_size_monotonicity(self):
        lives = [_run(a0=a0).cycles_to_failure for a0 in (0.0005, 0.0008, 0.0012)]
        assert lives[0] > lives[1] > lives[2]

    def test_unit_overload_reproduces_unretarded_life(self):
        retarded = _run(olr=1.0)
        plain = _run(olr=1.0, model="none")
        assert retarded.cycles_to_failure == pytest.approx(plain.cycles_to_failure, rel=1e-9)
        assert retarded.retarded_cycles == 0.0
        assert retarded.retardation_life_factor == pytest.approx(1.0, rel=1e-9)

    def test_retardation_extends_life_monotonically_in_overload_ratio(self):
        factors = []
        for olr in (1.0, 1.25, 1.5):
            retarded = _run(olr=olr)
            plain = _run(olr=olr, model="none")
            factors.append(retarded.cycles_to_failure / plain.cycles_to_failure)
        assert factors[0] == pytest.approx(1.0, rel=1e-9)
        assert factors[0] < factors[1] < factors[2]

    def test_spectrum_cycle_bookkeeping_is_consistent(self):
        result = _run()
        total = result.spectrum_repetitions * result.spectrum_cycles
        assert result.cycles_to_failure <= total + 1e-6
        assert result.cycles_to_failure >= total - result.spectrum_cycles - 1e-6

    def test_retardation_events_recorded(self):
        result = _run()
        assert result.retardation_events
        first = result.retardation_events[0]
        assert 0.0 < first.crack_m < result.critical_crack_m
        assert first.zone_m > 0.0

    def test_critical_stress_pinning_shifts_critical_crack(self):
        low = _run(olr=1.5, critical_stress_pa=SIGMA_RANGE / (1 - STRESS_RATIO))
        high = _run(olr=1.5, critical_stress_pa=1.5 * SIGMA_RANGE / (1 - STRESS_RATIO))
        assert high.critical_crack_m < low.critical_crack_m

    def test_pinned_critical_crack_matches_closed_form(self):
        sigma_max = SIGMA_RANGE / (1 - STRESS_RATIO)
        result = spectrum_residual_life(
            block_spectrum([SIGMA_RANGE], [1000.0], stress_ratio=STRESS_RATIO),
            LAW,
            lambda s: lambda a: s * math.sqrt(math.pi * a),
            initial_crack_m=0.002,
            upper_crack_m=0.090,
            fracture_toughness_pa_sqrt_m=K_IC,
            yield_stress_pa=YIELD,
            model="none",
            critical_stress_pa=sigma_max,
        )
        expected = (K_IC / sigma_max) ** 2 / math.pi
        assert result.critical_crack_m == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- #
# discretisation
# --------------------------------------------------------------------------- #
class TestDiscretisation:
    def test_substep_convergence_of_retarded_life(self):
        live = [_run(spacing=20000.0, olr=1.5, substeps=s).cycles_to_failure for s in (4, 8, 16)]
        coarse, medium, fine = live[0], live[1], live[2]
        _order, _extrap, gci, ok = richardson_gci(fine, medium, coarse, requirement=0.05)
        assert gci <= 0.05, f"GCI {gci:.4f} above 5%: {live}"
        assert ok

    def test_band_count_convergence_with_pinned_peak(self):
        peak = 4.0 * 60e6 / (1.0 - STRESS_RATIO)
        lives = []
        for n_bands in (4, 8, 16):
            result = spectrum_residual_life(
                rayleigh_band_spectrum(60e6, 100.0, n_bands=n_bands, stress_ratio=STRESS_RATIO),
                LAW,
                _hole_geometry,
                initial_crack_m=A0,
                upper_crack_m=UPPER,
                fracture_toughness_pa_sqrt_m=K_IC,
                yield_stress_pa=YIELD,
                model="none",
                critical_stress_pa=peak,
            )
            lives.append(result.cycles_to_failure)
        _order, _extrap, gci, ok = richardson_gci(lives[2], lives[1], lives[0], requirement=0.05)
        assert gci <= 0.05, f"band GCI {gci:.4f}: {lives}"
        assert ok


# --------------------------------------------------------------------------- #
# inverse identification
# --------------------------------------------------------------------------- #
class TestParameterIdentification:
    def test_recover_paris_constants_from_synthetic_curve(self):
        a0, a_c = 0.002, 0.010
        n_points = 201
        cracks = [a0 + (a_c - a0) * i / (n_points - 1) for i in range(n_points)]
        cycles = [
            0.0 if a <= a0 else paris_integrate_closed_form(a0, a, SIGMA_RANGE, LAW) for a in cracks
        ]
        fit = fit_paris_parameters(cracks, cycles, lambda a: SIGMA_RANGE * math.sqrt(math.pi * a))
        assert isinstance(fit, ParisFit)
        assert fit.exponent_m == pytest.approx(LAW.exponent_m, rel=1e-3)
        assert fit.coefficient_c == pytest.approx(LAW.coefficient_c, rel=1e-3)
        assert fit.r_squared > 0.999

    def test_fit_rejects_undersized_input(self):
        with pytest.raises(ValueError):
            fit_paris_parameters(
                [0.002, 0.003], [0.0, 1.0], lambda a: SIGMA_RANGE * math.sqrt(math.pi * a)
            )


# --------------------------------------------------------------------------- #
# SIF anchors used by the spectrum chain
# --------------------------------------------------------------------------- #
class TestSifAnchors:
    def test_secant_finite_width_against_exact_array_factor(self):
        sigma, a, W = 100e6, 0.010, 0.100
        exact = sigma * math.sqrt(W * math.tan(math.pi * a / W))
        secant = sigma * math.sqrt(math.pi * a) / math.sqrt(math.cos(math.pi * a / W))
        assert secant == pytest.approx(exact, rel=2e-2)

    def test_edge_crack_one_point_one_two_limit(self):
        sigma, a = 100e6, 1.0e-7
        value = edge_crack_finite_width_delta_k(sigma, 0.1)(a)
        assert value == pytest.approx(1.12 * sigma * math.sqrt(math.pi * a), rel=2e-2)

    def test_surface_crack_semicircular_limit(self):
        sigma, a = 100e6, 1.0e-3
        value = surface_crack_delta_k(sigma, 0.05, a, a, front_location="deepest")(a)
        assert value == pytest.approx(0.6637 * sigma * math.sqrt(math.pi * a), rel=2e-2)


# --------------------------------------------------------------------------- #
# cross-adjudication with the S-N method
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_paris_vs_sn_within_one_order():
    result = _run(olr=1.0, model="none")
    n_sn = SNCurveIIW(detail_category_pa=90e6).life(SIGMA_RANGE)
    assert abs(math.log10(result.cycles_to_failure / n_sn)) <= 1.0


# --------------------------------------------------------------------------- #
# end-to-end qualification
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_full_qualification_all_pass():
    evidence = run_spectrum_retardation_qualification()
    assert evidence["all_passed"], f"failed criteria: {evidence['failed_ids']}"
    assert evidence["n_criteria"] >= 20
    for criterion in "abcdefghijklm":
        assert any(row["id"].startswith(criterion) for row in evidence["criteria"]), (
            f"criterion ({criterion}) missing from the evidence set"
        )

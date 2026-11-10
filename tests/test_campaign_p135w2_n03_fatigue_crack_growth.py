"""Campaign p135w2-n03: Full fatigue crack growth verification suite.

Runs all acceptance criteria (a)-(g) from the campaign DAG node and produces
a machine-readable evidence JSON.  This is the single entry point that the
campaign runner calls to decide whether the node passes.

Criteria:
  (a) SIF analytical anchors: K_I vs closed-form ≤2%
  (b) Paris integration: closed-form vs RK4 ≤1%
  (c) Negative control: ΔK=0 → da/dN=0, infinite life
  (d) Residual life monotonicity: a₀↑ → N_f↓
  (e) Paris vs S-N: within 1 order of magnitude for weld toe
  (f) GCI convergence: ≤5%
  (g) Paris exponent sweep: m=2/3/4 → N_f monotonically decreasing
"""
from __future__ import annotations

import json
import math
import os

import pytest

from tensorfem.fatigue_crack_propagation import (
    ParisLaw,
    k_edge_crack_finite_width,
    k_edge_crack_semi_infinite,
    k_finite_width_plate,
    k_hole_edge_crack,
    k_infinite_plate,
    k_semi_elliptical_surface_crack,
    paris_integrate_closed_form,
)
from tensorfem.fatigue.crack_growth_geometry import (
    SNCurveIIW,
    centre_crack_finite_width_delta_k,
    compare_paris_vs_iiw,
    convergence_study,
    critical_crack_length,
    edge_crack_finite_width_delta_k,
    fillet_weld_toe_life,
    hole_edge_crack_delta_k,
    hole_edge_crack_life,
    integrate_paris_ode,
    surface_crack_delta_k,
    weld_toe_surface_crack_life,
)
from tensorfem.fatigue.crack_growth_qualification import (
    run_geometry_aware_qualification,
)

# Shared parameters
SIGMA = 100.0e6
K_IC = 60.0e6
STEEL = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)
EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "fatigue_crack_growth")


class TestCampaignCriterionA:
    """(a) SIF analytical anchors: K_I vs closed-form ≤2%."""

    def test_a1_infinite_plate_limit(self):
        """Finite-width correction → 1 as a/W → 0."""
        dk = centre_crack_finite_width_delta_k(SIGMA, 0.2)
        a_tiny = 1e-8
        actual = dk(a_tiny)
        oracle = k_infinite_plate(SIGMA, a_tiny).k_i_pa_sqrt_m
        err = abs(actual - oracle) / oracle
        assert err < 1e-4, f"Finite-width → infinite plate error {err:.2e}"

    def test_a2_edge_crack_F_1_12(self):
        """Edge crack semi-infinite plate: F = 1.12."""
        a = 2e-3
        k = k_edge_crack_semi_infinite(SIGMA, a).k_i_pa_sqrt_m
        oracle = 1.12 * SIGMA * math.sqrt(math.pi * a)
        err = abs(k - oracle) / oracle
        assert err < 0.02, f"Edge crack F error {err:.2e}"

    def test_a3_newman_raju_semicircular(self):
        """Semicircular surface crack deepest point: F → 0.6637."""
        a = 1e-5
        sif = k_semi_elliptical_surface_crack(SIGMA, a, a, 1.0, front_location="deepest")
        F = sif.k_i_pa_sqrt_m / (SIGMA * math.sqrt(math.pi * a))
        err = abs(F - 0.6637) / 0.6637
        assert err < 0.02, f"Newman-Raju semicircular F error {err:.2e}"

    def test_a4_hole_edge_kirsch_limit(self):
        """Hole-edge crack a/R → 0: F → 1.12 K_t = 3.36."""
        f = k_hole_edge_crack(SIGMA, 1e-7, 5e-3).geometry_factor
        err = abs(f - 3.36) / 3.36
        assert err < 0.02, f"Hole-edge Kirsch limit error {err:.2e}"

    def test_a5_finite_width_vs_isida(self):
        """Finite-width centre crack vs Isida table ≤2%."""
        isida = {0.2: 1.0254, 0.4: 1.1118, 0.6: 1.2977}
        W = 0.2
        dk = centre_crack_finite_width_delta_k(SIGMA, W)
        for ratio, f_ref in isida.items():
            a = 0.5 * ratio * W
            k_inf = k_infinite_plate(SIGMA, a).k_i_pa_sqrt_m
            y_model = dk(a) / k_inf
            err = abs(y_model - f_ref) / f_ref
            assert err < 0.02, f"Isida 2a/W={ratio}: error {err:.2e}"


class TestCampaignCriterionB:
    """(b) Paris integration: closed-form vs RK4 ≤1%."""

    def test_b1_closed_form_vs_rk4(self):
        """RK4 in ln(a) matches closed-form for constant ΔK."""
        law = ParisLaw(coefficient_c=1.0e-32, exponent_m=3.0)
        a0, ac, dsigma = 1e-4, 5e-3, 80e6
        n_closed = paris_integrate_closed_form(a0, ac, dsigma, law)
        dk = centre_crack_finite_width_delta_k(dsigma, 1e3)
        n_rk4 = integrate_paris_ode(dk, law, a0, ac, n_steps=200)
        err = abs(n_rk4 - n_closed) / n_closed
        assert err < 0.01, f"Paris closed-form vs RK4 error {err:.2e}"

    def test_b2_constant_delta_k(self):
        """Truly constant ΔK: N = (a_c - a_0) / (C ΔK^m)."""
        law = ParisLaw(coefficient_c=1.0e-32, exponent_m=3.0)
        a0, ac = 1e-4, 5e-3
        dk_const = 25e6
        n_exact = (ac - a0) / (law.coefficient_c * dk_const ** law.exponent_m)
        n_rk4 = integrate_paris_ode(lambda _a: dk_const, law, a0, ac, n_steps=100)
        err = abs(n_rk4 - n_exact) / n_exact
        assert err < 0.01, f"Constant ΔK error {err:.2e}"


class TestCampaignCriterionC:
    """(c) Negative control: ΔK=0 → da/dN=0, infinite life."""

    def test_c1_zero_delta_k_zero_rate(self):
        law = ParisLaw(coefficient_c=1e-32, exponent_m=3.0)
        assert law.da_dN(0.0) == 0.0

    def test_c2_zero_delta_k_infinite_life(self):
        law = ParisLaw(coefficient_c=1e-32, exponent_m=3.0)
        life = integrate_paris_ode(lambda _a: 0.0, law, 1e-4, 1e-3)
        assert math.isinf(life)

    def test_c3_below_threshold_infinite_life(self):
        law = ParisLaw(coefficient_c=1e-32, exponent_m=3.0, threshold_dk=5e6)
        life = integrate_paris_ode(lambda _a: 3e6, law, 1e-4, 1e-3)
        assert math.isinf(life)
        assert law.da_dN(5e6) == 0.0  # at threshold, no growth


class TestCampaignCriterionD:
    """(d) Residual life monotonicity: a₀↑ → N_f↓."""

    def test_d1_monotone_decreasing(self):
        """Use centre crack (infinite plate limit) for clean monotonicity test."""
        law = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)
        # Use centre crack in wide plate (effectively infinite) for clean test
        dk = centre_crack_finite_width_delta_k(100e6, 1.0)  # W=1m, very wide
        a0_vals = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
        # Critical crack for centre crack: K_I = sigma*sqrt(pi*a)*Y = K_IC
        # For wide plate Y~1, so a_c = (K_IC/(sigma*sqrt(pi)))^2
        sigma = 100e6
        a_c = (K_IC / (sigma * math.sqrt(math.pi))) ** 2  # ~0.0115 m
        n_f_vals = []
        for a0 in a0_vals:
            assert a0 < a_c, f"a0={a0} >= a_c={a_c}"
            n_f = integrate_paris_ode(dk, law, a0, a_c, n_steps=200)
            n_f_vals.append(n_f)
        for i in range(len(n_f_vals) - 1):
            assert n_f_vals[i] > n_f_vals[i + 1], (
                f"N_f not monotone: a₀={a0_vals[i]:.1e} → {n_f_vals[i]:.2e}, "
                f"a₀={a0_vals[i+1]:.1e} → {n_f_vals[i+1]:.2e}"
            )


class TestCampaignCriterionE:
    """(e) Paris vs S-N: within 1 order of magnitude for weld toe."""

    def test_e1_paris_vs_iiw_weld_toe(self):
        law = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)
        sn = SNCurveIIW(detail_category_pa=71e6)
        life = weld_toe_surface_crack_life(
            thickness_m=0.020, crack_depth_m=2e-4, crack_half_length_m=1e-3,
            sigma_range_pa=100e6, law=law,
            fracture_toughness_pa_sqrt_m=K_IC, plate_half_width_m=1.0,
        )
        agreement = compare_paris_vs_iiw(life.n_f, sn, 100e6 * 1.2)
        assert agreement.within_one_order, (
            f"Paris N_f={life.n_f:.2e}, S-N N_f={agreement.n_f_sn:.2e}, "
            f"log₁₀ ratio={agreement.log10_ratio:.2f}"
        )


class TestCampaignCriterionF:
    """(f) GCI convergence: ≤5%."""

    def test_f1_gci_growth_integral(self):
        law = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)
        dk = centre_crack_finite_width_delta_k(80e6, 0.2)
        study = convergence_study(dk, law, 1e-4, 5e-3, base_steps=50)
        assert study.gci_fine <= 0.05, (
            f"GCI = {study.gci_fine * 100:.4f}% > 5%"
        )
        assert study.observed_order > 3.0, (
            f"Observed order {study.observed_order:.2f} < 3 (RK4 should be ~4)"
        )


class TestCampaignCriterionG:
    """(g) Paris exponent sweep: m=2/3/4 → N_f monotonically decreasing."""

    def test_g1_monotone_decreasing_with_m(self):
        dk = centre_crack_finite_width_delta_k(80e6, 1e3)
        n_f_vals = []
        for m in (2.0, 3.0, 4.0):
            law = ParisLaw(coefficient_c=1.5e-29, exponent_m=m)
            n_f = integrate_paris_ode(dk, law, 1e-4, 5e-3, n_steps=200)
            n_f_vals.append(n_f)
        assert n_f_vals[0] > n_f_vals[1] > n_f_vals[2], (
            f"N_f not monotone in m: {[f'{n:.2e}' for n in n_f_vals]}"
        )


class TestCampaignFullQualification:
    """Run the full qualification suite and save evidence JSON."""

    def test_full_qualification_all_pass(self):
        """Run the complete (a)-(g) qualification and verify all pass."""
        result = run_geometry_aware_qualification()
        evidence = result["evidence"]

        # Group by criterion
        by_criterion = {}
        for e in evidence:
            c = e["criterion"]
            by_criterion.setdefault(c, []).append(e)

        # Check all passed
        for criterion in ["a", "b", "c", "d", "e", "f", "g"]:
            items = by_criterion.get(criterion, [])
            assert len(items) > 0, f"No evidence for criterion ({criterion})"
            for item in items:
                assert item.get("passed", False), (
                    f"Criterion ({criterion}) {item['id']} FAILED: {item}"
                )

        # Save evidence JSON
        os.makedirs(EVIDENCE_DIR, exist_ok=True)
        evidence_path = os.path.join(EVIDENCE_DIR, "campaign_evidence.json")
        with open(evidence_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    def test_three_detail_verification(self):
        """Verify three typical ship structural details produce sensible N_f."""
        law = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)
        dsigma = 100e6

        # 1. Weld toe semi-elliptical surface crack
        weld_life = weld_toe_surface_crack_life(
            thickness_m=0.020, crack_depth_m=2e-4, crack_half_length_m=1e-3,
            sigma_range_pa=dsigma, law=law,
            fracture_toughness_pa_sqrt_m=K_IC, plate_half_width_m=1.0,
        )
        assert weld_life.n_f > 0
        assert math.isfinite(weld_life.n_f)

        # 2. Hole edge crack
        hole_life = hole_edge_crack_life(
            hole_radius_m=0.01, plate_width_m=0.15,
            crack_length_m=1e-4, sigma_range_pa=dsigma,
            law=law, fracture_toughness_pa_sqrt_m=K_IC,
        )
        assert hole_life.n_f > 0
        assert math.isfinite(hole_life.n_f)

        # 3. Fillet weld toe (edge crack)
        fillet_life = fillet_weld_toe_life(
            plate_thickness_m=0.010, weld_leg_m=0.008, crack_length_m=1e-4,
            sigma_range_pa=dsigma, law=law,
            fracture_toughness_pa_sqrt_m=K_IC,
        )
        assert fillet_life.n_f > 0
        assert math.isfinite(fillet_life.n_f)

        # All lives should be in a physically reasonable range (1e2 - 1e10 cycles)
        for name, life in [("weld_toe", weld_life), ("hole_edge", hole_life),
                           ("fillet_weld", fillet_life)]:
            assert 1e2 < life.n_f < 1e10, (
                f"{name} N_f = {life.n_f:.2e} outside reasonable range"
            )

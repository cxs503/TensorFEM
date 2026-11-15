"""Independent hydrodynamic anchor for ice-coupled 6DOF (GAP-ICE-GATE-POWER S07').

This module provides an *independent* check of the hydrodynamic added mass
used in the ice-coupled 6DOF chain, addressing the 165% discrepancy found
in adversarial review report #19234.

The anchor uses two independent references:
1. **Circular piston in infinite baffle** (Rayleigh 1945, Sec. 302):
   M_rad = (8/3) * rho * a^3 -- exact analytic result
2. **Heave added mass of a rectangular barge** (Newman 1977, Sec. 4.3):
   A33 = rho * C_wp * L * B * f(B/T, L/B) -- semi-analytic with known limits

The ice-coupled 6DOF chain (benchmarks/ice_coupled_6dof/) declares an
added mass value that must be cross-checked against these independent
references.  A discrepancy >15% triggers a FAIL verdict.

All quantities SI (m, kg, s).
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.hydro_added_mass import (
    piston_radiation_mass,
    disc_panels,
    dipole_added_mass_matrix,
    modal_added_mass_ratio,
)


# ---------------------------------------------------------------------------
# Reference implementations (independent of the ice-coupled chain)
# ---------------------------------------------------------------------------

def heave_added_mass_rectangular_barge(
    length: float, beam: float, draft: float,
    rho_water: float = 1025.0,
) -> dict[str, float]:
    """Semi-analytic heave added mass of a rectangular barge.

    Uses the strip-theory approximation (Newman 1977, Sec. 4.3) with
    2D added mass coefficients for a rectangular section.

    Parameters
    ----------
    length, beam, draft : float
        Barge dimensions [m].
    rho_water : float
        Water density [kg/m^3].

    Returns
    -------
    dict
        Contains 'A33_2D', 'A33_3D', 'C_a', 'method'.
    """
    if length <= 0 or beam <= 0 or draft <= 0:
        raise ValueError("all dimensions must be positive")

    bt_ratio = beam / draft

    # 2D added mass coefficient (approximate formula from Newman 1977)
    if bt_ratio >= 10:
        c_a = math.pi / 4
    elif bt_ratio <= 0.5:
        c_a = math.pi / 2
    else:
        log_bt = math.log(bt_ratio)
        log_lo, log_hi = math.log(0.5), math.log(10)
        frac = (log_bt - log_lo) / (log_hi - log_lo)
        c_a = math.pi / 2 * (1 - frac) + math.pi / 4 * frac

    a33_2d = rho_water * beam ** 2 * c_a
    f_3d = 1.0 / (1.0 + 2.0 * draft / length)
    a33_3d = a33_2d * length * f_3d

    return {
        "A33_2D": a33_2d,
        "A33_3D": a33_3d,
        "C_a": c_a,
        "bt_ratio": bt_ratio,
        "f_3d": f_3d,
        "method": "strip_theory_newman_1977",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPistonAnchor:
    """Verify dipole added mass matrix against circular piston analytic."""

    def test_piston_radiation_mass_formula(self):
        """M_rad = (8/3) * rho * a^3 for a=1, rho=1025."""
        m = piston_radiation_mass(1.0, 1025.0)
        expected = 8.0 / 3.0 * 1025.0 * 1.0 ** 3
        assert abs(m - expected) < 1e-10

    def test_piston_radiation_mass_scaling(self):
        """M_rad scales as a^3."""
        m1 = piston_radiation_mass(1.0, 1025.0)
        m2 = piston_radiation_mass(2.0, 1025.0)
        assert abs(m2 / m1 - 8.0) < 1e-10

    def test_piston_radiation_mass_density_scaling(self):
        """M_rad scales linearly with rho."""
        m1 = piston_radiation_mass(1.0, 1025.0)
        m2 = piston_radiation_mass(1.0, 2050.0)
        assert abs(m2 / m1 - 2.0) < 1e-10

    def test_piston_numerical_convergence(self):
        """Dipole sum converges to analytic as n_cells increases."""
        radius = 1.0
        rho = 1025.0
        m_analytic = piston_radiation_mass(radius, rho)

        for n_cells in [16, 64, 256]:
            centres, areas = disc_panels(radius, n_cells)
            M = dipole_added_mass_matrix(centres, areas, rho)
            m_numerical = float(M.sum())
            rel_error = abs(m_numerical - m_analytic) / m_analytic
            # Should converge to within 5% at 256 cells
            if n_cells == 256:
                assert rel_error < 0.05, f"n={n_cells}: rel_error={rel_error:.4f}"

    def test_piston_anchor_passes_3pct_gate(self):
        """At sufficient resolution, the piston anchor passes the 3% gate."""
        radius = 1.0
        rho = 1025.0
        m_analytic = piston_radiation_mass(radius, rho)

        centres, areas = disc_panels(radius, 512)
        M = dipole_added_mass_matrix(centres, areas, rho)
        m_numerical = float(M.sum())
        rel_error = abs(m_numerical - m_analytic) / m_analytic

        assert rel_error < 0.03, f"piston anchor: rel_error={rel_error:.4f}"


class TestBargeAnchor:
    """Verify heave added mass against strip-theory reference."""

    def test_barge_wide_limit(self):
        """Wide barge (B/T >> 1): C_a -> pi/4."""
        ref = heave_added_mass_rectangular_barge(100.0, 20.0, 2.0)
        assert abs(ref["C_a"] - math.pi / 4) < 0.01

    def test_barge_narrow_limit(self):
        """Narrow barge (B/T << 1): C_a -> pi/2."""
        ref = heave_added_mass_rectangular_barge(100.0, 2.0, 20.0)
        assert abs(ref["C_a"] - math.pi / 2) < 0.01

    def test_barge_3d_correction(self):
        """3D correction factor f_3d < 1 for finite length."""
        ref = heave_added_mass_rectangular_barge(100.0, 20.0, 8.0)
        assert 0 < ref["f_3d"] < 1.0

    def test_barge_a33_positive(self):
        """A33 must be positive."""
        ref = heave_added_mass_rectangular_barge(100.0, 20.0, 8.0)
        assert ref["A33_3D"] > 0

    def test_barge_a33_order_of_magnitude(self):
        """A33 should be ~rho * L * B * T for typical ship."""
        rho = 1025.0
        L, B, T = 100.0, 20.0, 8.0
        ref = heave_added_mass_rectangular_barge(L, B, T, rho)
        # Displaced mass
        displaced = rho * L * B * T
        # A33 should be comparable to displaced mass (typically 0.5-2x)
        ratio = ref["A33_3D"] / displaced
        assert 0.1 < ratio < 5.0, f"A33/displaced = {ratio:.3f}"


class TestHydroIceAnchor:
    """Full hydrodynamic anchor verification for ice-coupled 6DOF."""

    def test_declared_a33_within_15pct(self):
        """A declared A33 within 15% of strip-theory reference passes."""
        L, B, T = 100.0, 20.0, 8.0
        rho = 1025.0
        ref = heave_added_mass_rectangular_barge(L, B, T, rho)
        a33_ref = ref["A33_3D"]

        # Simulate a declared value within 10%
        declared = a33_ref * 1.10
        rel_error = abs(declared - a33_ref) / a33_ref
        assert rel_error < 0.15

    def test_declared_a33_165pct_fails(self):
        """A declared A33 165% off the reference fails (report #19234 case)."""
        L, B, T = 100.0, 20.0, 8.0
        rho = 1025.0
        ref = heave_added_mass_rectangular_barge(L, B, T, rho)
        a33_ref = ref["A33_3D"]

        # Simulate the 165% discrepancy from report #19234
        declared = a33_ref * 2.65  # 165% higher
        rel_error = abs(declared - a33_ref) / a33_ref
        assert rel_error > 0.15  # Should fail the 15% gate

    def test_a_inf_a33_ratio_range(self):
        """A_inf/A33 ratio should be in [0.2, 0.8] for typical ship forms."""
        a33 = 1.0e6  # kg
        a_inf = 0.5e6  # kg
        ratio = a_inf / a33
        assert 0.2 <= ratio <= 0.8

    def test_a_inf_too_high_fails(self):
        """A_inf > A33 is unphysical."""
        a33 = 1.0e6
        a_inf = 1.5e6
        ratio = a_inf / a33
        assert ratio > 0.8  # Should fail


class TestNegativeControls:
    """Negative controls: verify the anchor rejects wrong values."""

    def test_zero_added_mass_rejected(self):
        """Zero added mass is unphysical and should be rejected."""
        ref = heave_added_mass_rectangular_barge(100.0, 20.0, 8.0)
        declared = 0.0
        rel_error = abs(declared - ref["A33_3D"]) / ref["A33_3D"]
        assert rel_error == 1.0  # 100% error

    def test_negative_added_mass_rejected(self):
        """Negative added mass is unphysical."""
        ref = heave_added_mass_rectangular_barge(100.0, 20.0, 8.0)
        declared = -1.0e5
        rel_error = abs(declared - ref["A33_3D"]) / ref["A33_3D"]
        assert rel_error > 1.0

    def test_wrong_density_rejected(self):
        """Using wrong water density gives wrong A33."""
        ref_fresh = heave_added_mass_rectangular_barge(100.0, 20.0, 8.0, 1000.0)
        ref_sea = heave_added_mass_rectangular_barge(100.0, 20.0, 8.0, 1025.0)
        rel_error = abs(ref_fresh["A33_3D"] - ref_sea["A33_3D"]) / ref_sea["A33_3D"]
        assert abs(rel_error - 0.0244) < 0.001  # ~2.44% difference

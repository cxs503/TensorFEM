"""Coupled FSI validation: implicit IBM + Aitken relaxation vs Lamb (1932).

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION``.

This test exercises the *coupled* implicit-IBM + Aitken solver
(``IBMImplicitFSI``) on the spring-mounted plate configuration whose
analytic reference is Lamb (1932) art. 71:

    m_a = rho_f * pi * c^2          (2-D thin plate, broadside motion)
    f/f_vac = sqrt(m_s / (m_s + m_a))

The acceptance criterion is **< 15 %** on the coupled frequency ratio
``f/f_vac`` (owner precision standard).  The test runs at two lattice
resolutions (``c = 8, 16``) and verifies that the error decreases
with resolution, confirming convergence toward the analytic reference.

Why this test is needed
-----------------------
The existing ``test_ibm_aitken_convergence.py`` tests the *analysis
machinery* (fits, extrapolation, blind prediction) on synthetic data and
one smoke run.  What was missing is a test that actually runs the coupled
solver at multiple resolutions and asserts the frequency is within the
15 % acceptance band.  This test fills that gap.

The test is designed to run in < 90 s on a single core (smoke-size boxes
with 2 coupled periods per case).
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from tensorfem.ibm_aitken_added_mass import (  # noqa: E402
    IBMImplicitFSI,
    SpringMountedBody,
    analytic_added_mass,
    estimate_period_damped,
    plate_markers,
)
from tensorfem.ibm_aitken_implicit import LBMFluid2D  # noqa: E402

from xval_implicit_ibm_aitken_convergence import (  # noqa: E402
    MASS_RATIO,
    PERIOD_TARGET,
    run_case,
)

#: Owner acceptance: coupled frequency error < 15 %
TOL = 0.15

#: Steps per case: 2 coupled periods (enough for damped sine fit)
N_STEPS = 1200


def _run_coupled(c, L, n_steps=N_STEPS, omega_lbm=1.9, y0=0.2, ds=1.0):
    """Run one coupled free-vibration case and return the result dict."""
    cfg = {"name": f"c{c}_L{L}", "group": "T", "c": float(c),
           "lattice": [int(L), int(L)], "n_steps": int(n_steps),
           "omega_lbm": float(omega_lbm), "y0": float(y0),
           "ds": float(ds), "save_field": False}
    return run_case(cfg)


class TestCoupledFSIVsLamb:
    """Coupled implicit-IBM + Aitken solver vs Lamb (1932) thin-plate reference."""

    # -----------------------------------------------------------------------
    # Single-resolution acceptance tests (smoke-size grids)
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize("c,L", [
        (8, 72),     # smoke-size: ~11 s
        (16, 144),   # v6 reference geometry: ~31 s
    ])
    def test_coupled_frequency_within_15_pct(self, c, L):
        """The coupled frequency ratio must be within 15 % of the Lamb reference.

        This is the owner acceptance criterion for the anchor.
        """
        result = _run_coupled(c, L)
        assert not result["diverged"], f"Case diverged: {result.get('reason', '')}"
        assert np.isfinite(result["f_ratio_measured"])
        assert result["f_ratio_error"] < TOL, (
            f"f/f_vac error {100 * result['f_ratio_error']:.2f} % >= {100 * TOL:.0f} % "
            f"(c={c} dx, L={L})"
        )

    # -----------------------------------------------------------------------
    # Convergence: error must decrease with resolution
    # -----------------------------------------------------------------------
    def test_error_decreases_with_resolution(self):
        """The frequency error must decrease as the body is refined (c increases).

        This confirms the solver is converging toward the analytic reference,
        not just happening to land within the tolerance band by chance.
        """
        r_coarse = _run_coupled(8, 72)
        r_fine = _run_coupled(16, 144)

        assert not r_coarse["diverged"]
        assert not r_fine["diverged"]

        assert r_fine["f_ratio_error"] < r_coarse["f_ratio_error"], (
            f"Error did not decrease with resolution: "
            f"c=8: {100 * r_coarse['f_ratio_error']:.2f} %, "
            f"c=16: {100 * r_fine['f_ratio_error']:.2f} %"
        )

    # -----------------------------------------------------------------------
    # Aitken relaxation convergence
    # -----------------------------------------------------------------------
    def test_aitken_converges_every_step(self):
        """Aitken sub-iterations must converge without hitting max_iter."""
        result = _run_coupled(8, 72)
        assert result["maxit_hits"] == 0, (
            f"Aitken hit max_iter {result['maxit_hits']} times"
        )
        assert result["sub_iter_max"] <= 30

    # -----------------------------------------------------------------------
    # Period estimator consistency
    # -----------------------------------------------------------------------
    def test_damped_period_estimator_is_valid(self):
        """The damped sine least-squares estimator must be valid."""
        result = _run_coupled(8, 72)
        assert result["period_damped_valid"], (
            "Damped period estimator failed (insufficient cycles or bad fit)"
        )
        # The RMS of the fit should be small relative to the amplitude
        assert result["period_damped_rms_rel"] < 0.05, (
            f"Damped fit RMS too large: {100 * result['period_damped_rms_rel']:.2f} %"
        )

    # -----------------------------------------------------------------------
    # Control: viscosity should not affect the inertial added mass
    # -----------------------------------------------------------------------
    def test_viscosity_control(self):
        """Changing lattice viscosity (omega) should not significantly move the answer.

        The added mass is an inertial (potential-flow) quantity; viscosity
        should only affect the damping rate, not the frequency.
        """
        base = _run_coupled(8, 72, omega_lbm=1.9)
        low_visc = _run_coupled(8, 72, omega_lbm=1.98)

        delta = abs(low_visc["f_ratio_measured"] - base["f_ratio_measured"]) \
            / base["f_ratio_measured"]

        assert delta < 0.10, (
            f"Viscosity change moved frequency by {100 * delta:.2f} %"
        )

    # -----------------------------------------------------------------------
    # Control: amplitude linearity
    # -----------------------------------------------------------------------
    def test_amplitude_linearity(self):
        """Halving the amplitude should not change the frequency ratio.

        The added mass is a linear (small-amplitude) quantity; the frequency
        should be independent of amplitude in the linear regime.
        """
        base = _run_coupled(8, 72, y0=0.2)
        small = _run_coupled(8, 72, y0=0.1)

        delta = abs(small["f_ratio_measured"] - base["f_ratio_measured"]) \
            / base["f_ratio_measured"]
        assert delta < 0.02, (
            f"Amplitude change moved frequency by {100 * delta:.2f} %"
        )

    # -----------------------------------------------------------------------
    # Analytic reference sanity check
    # -----------------------------------------------------------------------
    def test_analytic_reference_matches_lamb(self):
        """The analytic reference f/f_vac must match the Lamb formula exactly."""
        c = 16.0
        rho_f = 1.0
        m_a = analytic_added_mass(c, rho_f)
        m_s = m_a / MASS_RATIO
        k = (m_s + m_a) * (2.0 * math.pi / PERIOD_TARGET) ** 2
        f_vac = math.sqrt(k / m_s) / (2.0 * math.pi)
        f_ref = math.sqrt(k / (m_s + m_a)) / (2.0 * math.pi)
        f_ratio_ref = f_ref / f_vac

        # The reference must be sqrt(1 / (1 + m_a/m_s)) = sqrt(1 / 1.6)
        expected = math.sqrt(1.0 / (1.0 + MASS_RATIO))
        assert f_ratio_ref == pytest.approx(expected, rel=1e-12)

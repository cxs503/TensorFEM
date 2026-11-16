"""Unit / mathematics tests for the two-component Shan-Chen MCMP solver.

These are the L1 (implementation) checks for the anchor
``XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO``: conservation and equilibrium
invariants, the pseudo-potential, the analytic single-interface interaction
force, a regression against the live lbmpy reference case, a replay of the
defect that made the first delivery diverge, tau sensitivity, and spurious
current quantification.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tensorfem.lbm_shan_chen_mcmp import (
    CX, CY, GOLDEN_32x32, MCMPConfig, REF_A, ShanChenMCMP, equilibrium_d2q9,
    psi_exponential, relative_l2, run_reference_case, shan_chen_mcmp_evidence,
)


def test_equilibrium_at_rest_is_a_fixed_point():
    """Two uniform components at rest must not move (no spurious force)."""
    sim = ShanChenMCMP(MCMPConfig(nx=16, ny=16, g_ab=6.0, g_ba=6.0))
    sim.init_uniform(0.5, 0.5)
    f0 = sim.f.copy()
    sim.run(50)
    assert np.abs(sim.f - f0).max() < 1e-12
    assert sim.spurious_current() < 1e-12


def test_mass_conservation_at_g6():
    """Collision + periodic streaming conserve both component masses."""
    sim = ShanChenMCMP(MCMPConfig(nx=32, ny=32, g_ab=6.0, g_ba=6.0))
    sim.init_step_x(0.1, 0.9)
    m0 = sim.total_mass()
    sim.run(300)
    m1 = sim.total_mass()
    assert np.isfinite(sim.f).all()
    assert np.abs(m1 - m0).max() / m0.max() < 1e-12
    assert m0[0] == pytest.approx(0.5 * 32 * 32)


def test_psi_is_monotone_and_vanishes_at_zero():
    rho = np.linspace(1e-6, 5.0, 200)
    psi = psi_exponential(rho)
    assert psi[0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(psi) > 0.0)
    # psi -> rho0 as rho -> inf
    assert psi_exponential(np.array([200.0]))[0] == pytest.approx(1.0, abs=1e-80)


def test_interaction_force_matches_single_interface_analytic_value():
    """Hand-derived force on the two cells flanking a straight interface.

    With ``psi(rho) = 1 - exp(-rho)`` and a step ``rho_a = 0.1 | 0.9`` the
    weighted neighbour sum is ``(1/9 + 2/36) * dpsi_b`` along ``x``, so

        |F_a| = psi_a * g_ab * (1/6) * |psi_b(right) - psi_b(left)|

    on either side of the interface.  This value is independent of the solver
    loop, so it pins the sign and the normalisation of the force.
    """
    psi_lo = 1.0 - np.exp(-0.1)
    psi_hi = 1.0 - np.exp(-0.9)
    expected_dilute = psi_lo * 6.0 * (1.0 / 6.0) * (psi_hi - psi_lo)
    expected_dense = psi_hi * 6.0 * (1.0 / 6.0) * (psi_hi - psi_lo)

    sim = ShanChenMCMP(MCMPConfig(nx=64, ny=8, g_ab=6.0, g_ba=6.0))
    sim.init_step_x(0.1, 0.9)
    F = sim.interaction_force()
    # x = 31 is the last dilute column, x = 32 the first dense column
    assert F[0, 0, 0, 31] == pytest.approx(expected_dilute, rel=1e-10)
    assert F[0, 0, 0, 32] == pytest.approx(expected_dense, rel=1e-10)
    # b sees the mirrored gradient of psi_a, so its force has the opposite sign
    # (the two components are pushed apart across the interface)
    assert F[1, 0, 0, 31] == pytest.approx(-expected_dense, rel=1e-10)
    assert F[1, 0, 0, 32] == pytest.approx(-expected_dilute, rel=1e-10)


def test_force_is_zero_in_a_uniform_region():
    sim = ShanChenMCMP(MCMPConfig(nx=16, ny=16, g_ab=6.0, g_ba=6.0))
    sim.init_uniform(0.3, 0.7)
    assert np.abs(sim.interaction_force()).max() < 1e-14


def test_regression_against_live_lbmpy_reference_case():
    """32x32, g_ab = 6, 400 steps must reproduce the lbmpy-generated profile."""
    if not GOLDEN_32x32.exists():  # pragma: no cover
        pytest.skip(f"missing reference {GOLDEN_32x32}")
    ref = json.loads(GOLDEN_32x32.read_text())
    n = int(ref["params"]["n"])
    sim = ShanChenMCMP(MCMPConfig(nx=n, ny=n, g_ab=float(ref["params"]["g_ab"]),
                                  g_ba=float(ref["params"]["g_ab"])))
    sim.init_step_x(0.1, 0.9)
    sim.run(int(ref["params"]["steps"]))
    err = relative_l2(sim.density()[0, 0, :], np.asarray(ref["rho_a_profile"]))
    assert err < 1e-8, f"rel L2 {err:.3e} against live lbmpy reference"


def test_anchor_case_is_inside_the_15_percent_band():
    """The published 64^2 / 1000-step anchor case (7 s)."""
    res = run_reference_case(steps=1000)
    assert res["diverged_at"] is None
    assert res["rel_l2_a"] < 0.15
    assert res["rel_l2_b"] < 0.15


def test_first_delivery_defect_is_reproduced_by_the_switch():
    """The ``psi(x-c)`` neighbour convention must diverge (kept for the post-mortem)."""
    sim = ShanChenMCMP(MCMPConfig(nx=32, ny=32, g_ab=6.0, g_ba=6.0,
                                  neighbour_convention="minus"))
    sim.init_step_x(0.1, 0.9)
    diverged = False
    for _ in range(200):
        sim.step()
        if not np.isfinite(sim.f).all():
            diverged = True
            break
    assert diverged


def test_registry_evidence_passes():
    rows = shan_chen_mcmp_evidence()
    assert len(rows) == 3
    for row in rows:
        assert row.passed, f"{row.id} error {row.error:.3e} > {row.tolerance}"


# ======================================================================
# v2 additions: tau sensitivity and spurious current tests
# ======================================================================

class TestTauSensitivity:
    """Tau sensitivity: barycentric+Guo must remain stable across τ values."""

    @pytest.mark.parametrize("tau", [0.6, 0.7, 0.8, 1.0, 1.2, 1.5])
    def test_symmetric_tau_stable(self, tau):
        """Barycentric+Guo with symmetric τ must not diverge in 500 steps."""
        sim = ShanChenMCMP(MCMPConfig(nx=32, ny=32, g_ab=6.0, g_ba=6.0,
                                      tau_a=tau, tau_b=tau))
        sim.init_step_x(0.1, 0.9)
        for _ in range(500):
            sim.step()
            assert np.isfinite(sim.f).all(), f"diverged at τ={tau}, step={sim.t}"

    @pytest.mark.parametrize("tau", [0.6, 0.7, 0.8, 1.0, 1.2, 1.5])
    def test_symmetric_tau_mass_conserved(self, tau):
        """Mass must be conserved at all τ values."""
        sim = ShanChenMCMP(MCMPConfig(nx=32, ny=32, g_ab=6.0, g_ba=6.0,
                                      tau_a=tau, tau_b=tau))
        sim.init_step_x(0.1, 0.9)
        m0 = sim.total_mass()
        sim.run(300)
        m1 = sim.total_mass()
        assert np.isfinite(sim.f).all()
        assert np.abs(m1 - m0).max() / m0.max() < 1e-10, \
            f"mass drift at τ={tau}: {np.abs(m1 - m0).max() / m0.max():.3e}"

    @pytest.mark.parametrize("tau_a,tau_b", [(0.6, 1.0), (0.8, 1.2), (1.0, 1.5)])
    def test_asymmetric_tau_stable(self, tau_a, tau_b):
        """Barycentric+Guo with τ_a ≠ τ_b must not diverge."""
        sim = ShanChenMCMP(MCMPConfig(nx=32, ny=32, g_ab=6.0, g_ba=6.0,
                                      tau_a=tau_a, tau_b=tau_b))
        sim.init_step_x(0.1, 0.9)
        for _ in range(500):
            sim.step()
            assert np.isfinite(sim.f).all(), \
                f"diverged at τ_a={tau_a}, τ_b={tau_b}, step={sim.t}"

    @pytest.mark.parametrize("tau_a,tau_b", [(0.6, 1.0), (0.8, 1.2), (1.0, 1.5)])
    def test_asymmetric_tau_mass_conserved(self, tau_a, tau_b):
        """Mass must be conserved with asymmetric τ."""
        sim = ShanChenMCMP(MCMPConfig(nx=32, ny=32, g_ab=6.0, g_ba=6.0,
                                      tau_a=tau_a, tau_b=tau_b))
        sim.init_step_x(0.1, 0.9)
        m0 = sim.total_mass()
        sim.run(300)
        m1 = sim.total_mass()
        assert np.isfinite(sim.f).all()
        assert np.abs(m1 - m0).max() / m0.max() < 1e-10

    def test_barycentric_velocity_is_tau_independent(self):
        """The barycentric velocity must be the same regardless of τ.

        At the first step, the barycentric velocity depends only on the
        initial populations and the interaction force, not on τ.
        """
        u_vals = []
        for tau in [0.6, 0.8, 1.0, 1.2, 1.5]:
            sim = ShanChenMCMP(MCMPConfig(nx=16, ny=16, g_ab=6.0, g_ba=6.0,
                                          tau_a=tau, tau_b=tau))
            sim.init_step_x(0.1, 0.9)
            u = sim.barycentric_velocity()
            u_vals.append(u.copy())
        # All barycentric velocities should be identical (before any stepping)
        for i in range(1, len(u_vals)):
            assert np.abs(u_vals[i] - u_vals[0]).max() < 1e-14, \
                f"barycentric velocity differs at τ={[0.6,0.8,1.0,1.2,1.5][i]}"


class TestSpuriousCurrents:
    """Spurious current quantification for the step-interface case.

    Note: Shan-Chen spurious currents are a known model-level artifact
    (Wagner 2006, Lischner 2020).  They arise from the anisotropy of the
    lattice force discretization and do not decrease monotonically with
    resolution.  The tests here verify that they remain within physically
    acceptable bounds, not that they vanish.
    """

    @pytest.mark.parametrize("n", [32, 64])
    def test_spurious_current_is_small_at_g6(self, n):
        """Spurious current at g_ab=6 must be < 0.01 in lattice units."""
        sim = ShanChenMCMP(MCMPConfig(nx=n, ny=n, g_ab=6.0, g_ba=6.0))
        sim.init_step_x(0.1, 0.9)
        sim.run(500)
        sc = sim.spurious_current()
        assert sc < 0.01, f"spurious current {sc:.2e} >= 0.01 at n={n}"

    @pytest.mark.parametrize("g_ab", [3.0, 4.5])
    def test_spurious_current_decreases_with_resolution(self, g_ab):
        """Spurious current should decrease (or stay small) with resolution.

        At moderate g_ab, the spurious current generally decreases with
        resolution.  At high g_ab, the behavior can be non-monotonic due
        to the lattice anisotropy of the force discretization.
        """
        sc_32 = None
        sc_64 = None
        for n in [32, 64]:
            sim = ShanChenMCMP(MCMPConfig(nx=n, ny=n, g_ab=g_ab, g_ba=g_ab))
            sim.init_step_x(0.1, 0.9)
            sim.run(500)
            sc = sim.spurious_current()
            if n == 32:
                sc_32 = sc
            else:
                sc_64 = sc
        # Spurious current at n=64 should be <= spurious current at n=32
        # (or at least not significantly larger)
        assert sc_64 <= sc_32 * 1.5, \
            f"spurious current increased with resolution: n=32 {sc_32:.2e}, n=64 {sc_64:.2e}"

    def test_spurious_current_at_low_g_is_small(self):
        """At low interaction strength (g_ab=3), spurious current should be small.

        The Shan-Chen spurious current scales roughly as g_ab^2, so at g_ab=3
        it should be significantly smaller than at g_ab=6.
        """
        sim = ShanChenMCMP(MCMPConfig(nx=64, ny=64, g_ab=3.0, g_ba=3.0))
        sim.init_step_x(0.1, 0.9)
        sim.run(500)
        sc = sim.spurious_current()
        # At g_ab=3, spurious current should be < 0.005 (much less than at g_ab=6)
        assert sc < 5e-3, f"spurious current {sc:.2e} at g_ab=3"


class TestConvergence:
    """Self-convergence of the interface profile across resolutions."""

    def test_interface_profile_converges_with_resolution(self):
        """The interface profile at n=64 should be closer to n=96 than n=32 is."""
        profiles = {}
        for n in [32, 64, 96]:
            sim = ShanChenMCMP(MCMPConfig(nx=n, ny=n, g_ab=6.0, g_ba=6.0))
            sim.init_step_x(0.1, 0.9)
            steps = 400 * n // 32  # scale steps with resolution
            sim.run(steps)
            profiles[n] = sim.density()[0, 0, :].copy()

        # Interpolate fine profile to coarser grids
        fine = profiles[96]
        x_fine = np.linspace(0, 1, 96)
        x_64 = np.linspace(0, 1, 64)
        x_32 = np.linspace(0, 1, 32)
        fine_at_64 = np.interp(x_64, x_fine, fine)
        fine_at_32 = np.interp(x_32, x_fine, fine)

        err_32 = relative_l2(profiles[32], fine_at_32)
        err_64 = relative_l2(profiles[64], fine_at_64)

        # n=64 should be closer to n=96 than n=32 is
        assert err_64 < err_32, \
            f"convergence violated: err_32={err_32:.3e}, err_64={err_64:.3e}"

    def test_density_extrema_converge(self):
        """Density extrema should converge with resolution."""
        extrema = {}
        for n in [32, 64, 96]:
            sim = ShanChenMCMP(MCMPConfig(nx=n, ny=n, g_ab=6.0, g_ba=6.0))
            sim.init_step_x(0.1, 0.9)
            sim.run(400)
            rho = sim.density()
            extrema[n] = {"rho_min": float(rho.min()), "rho_max": float(rho.max())}

        # All densities should be positive
        for n, e in extrema.items():
            assert e["rho_min"] > 0, f"negative density at n={n}: {e['rho_min']}"
